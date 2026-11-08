"""Fast reactive L: the voice's first words never wait for the heavy mind.

Measured on this machine (persona/voice/bench/*.py):

    DualLoop deep turn end to end (this box, 2026-11-08):  first token p50 ~8-12 s
    local llama.cpp qwen2.5-1.5b-instruct-q4_K_M, warm, 4 threads:
                                                     first token p50 307 ms (n=10)
                                                     first CLAUSE   p50 ~450 ms
    same model, cold process (load on critical path):            ~6.8 s
    Gemini gemini-3.5-flash-lite over the wire:     first token p50 587 ms

Only the FIRST CLAUSE is awaited (``stop_at_clause``): the whole 40-token reply
costs ~1.0-2.2 s of decode, and waiting for it moved measured first-audio from
the model's ~0.3 s TTFT out to 2.5-4.1 s. Speaking the clause is the difference.

So the fast path is a *separate, tiny* model that answers short spoken turns
outright, started at the same instant as the full DualLoop turn. The heavy turn
keeps running; when its first clause arrives it continues after what L already
said (``skip_restated`` in realtime.py drops the echo). One mind, two speeds.

This module owns no orchestration and no fallback policy - VoiceSession does.
It is a streaming client plus a decision about whether the fast answer is safe
to speak, and it fails OPEN (no speech) rather than delaying the deep path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    import httpx2 as httpx  # type: ignore

log = logging.getLogger(__name__)

# llama.cpp's OpenAI-compatible server is the measured winner here. Ollama is
# not installed on this box (winget download failed); any OpenAI-compatible URL
# works, so the same switch points at Ollama, llama.cpp or a remote tiny model.
DEFAULT_BASE = os.environ.get("SAWII_FAST_L_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_MODEL = os.environ.get("SAWII_FAST_L_MODEL", "qwen2.5-1.5b-instruct-q4_k_m")

# A spoken turn should be answered in one short breath. Long enough to be a
# real reply, short enough that TTS of the first clause starts immediately.
MAX_TOKENS = 40

# How much of this call's own conversation rides along with the question. Six
# messages is three exchanges: enough for "and you?" / "tell me more" to
# resolve, small enough to cost no measurable latency on a warm local model.
HISTORY_TURNS = max(0, int(os.environ.get("SAWII_FAST_L_HISTORY", "6")))

_SYSTEM = (
    "You are the fast voice of one shared mind. You speak first while the deeper "
    "part of the same mind keeps working on the full answer.\n"
    "Answer in ONE short spoken sentence, 4 to 14 words, warm and plain. No "
    "lists, no quotes, no emoji, no stage directions. Never invent a fact, a "
    "memory, a name, a number, or anything you did from the words alone. When "
    "the turn needs a fact, a memory, a tool, or more than one sentence, reply "
    "with exactly: PASS\n"
)

# The tiny model says PASS when the turn belongs to the deep path.
_PASS = re.compile(r"^\s*(pass|none|n/?a)\b", re.I)
_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*\s")
# Same, but also true at end-of-stream, where no trailing space ever arrives.
_SENTENCE_DONE = re.compile(r"[.!?…][\"')\]]*$")
# A comma is enough to speak safely; a run-on answer can wait for its period.
_CLAUSE_END = re.compile(r"[,;:—-]\s")
# Below this, a "clause" is just a hello and there is nothing worth speaking.
MIN_CLAUSE_CHARS = 14


def clause_boundary(text: str) -> int:
    """End offset of the first complete clause in ``text``, or 0 to keep it all.

    The clause-stop path breaks out of the stream as soon as a clause exists,
    but the text streamed so far runs past that clause into the next one, so
    speaking it verbatim produced cut-off lines - measured on this box:
    "Hello! How can", "I am fine, thank", "I'm here, just".

    A sentence end is always a safe cut. A comma is only trusted when BOTH
    sides are substantial: "Fine, thank you" must survive whole, while
    "I am fine, thank you for asking" should cut to "I am fine".
    """
    stripped = text.rstrip()
    m = _SENTENCE_END.search(text)
    if m:
        return m.end()
    if _SENTENCE_DONE.search(stripped):
        return len(stripped)
    m = _CLAUSE_END.search(text)
    if m:
        head = text[:m.start()].strip()
        tail = text[m.end():].strip()
        if len(head.split()) >= 3 and len(tail.split()) >= 2:
            return m.start()
    return 0

# Measured: a 1.5B model asked to abstain still invents. Asked "remind me what
# we talked about last week" it answered "We discussed the weather and plans for
# the weekend" - a memory it does not have. Self-restraint is not a safety
# property this model has, so the gate is code, not prompt: anything that needs
# memory, a fact, a tool, or the person's own data goes to the deep path.
_DEEP_CUES = re.compile(
    r"\b(remember|memory|memories|recall|forgot|last (time|week|night|month|tuesday)|"
    r"yesterday|earlier|before|again|did i|do i|what i said|we (talked|said|discussed)|"
    r"tell me about my|my (name|birthday|schedule|calendar|email|files?)|"
    r"(open|close|click|type|search|find|send|write|create|run|check|book|add|set|delete|play)\b|"
    r"(weather|time|date|temperature|news|price|score|stock|traffic|version)|"
    r"(plan|schedule|agenda|tomorrow|tonight|the weekend|idea|ideas|suggest|recommend)|"
    r"(how much|how many|how long|which|whose|capital|population|formula|translate)|"
    r"(about me|noticed|observed|what do you think of me|do you like me))\b",
    re.I)
# A question, or a turn that names the assistant's own state, still belongs to
# the fast voice when it is purely social ("how are you?", "what should I do?").
_SOCIAL = re.compile(
    r"\b(hi|hey|hello|good morning|good evening|how are you|how's it going|how are you feeling|"
    r"what's up|sup|thanks|thank you|okay|ok|cool|nice|bye|good night|morning|evening|yeah|no)\b",
    re.I)


def needs_deep_path(text: str) -> bool:
    """True when the fast responder must stay silent and let the mind answer.

    Deliberately over-broad: a missed fast answer costs a second of latency, a
    wrong fast answer costs the person's trust.
    """
    t = (text or "").strip()
    if not t or len(t) > 160:
        return True
    if _DEEP_CUES.search(t):
        return True
    # A question the fast model cannot know the answer to.
    if t.endswith("?") and not _SOCIAL.search(t):
        return True
    return False


def fast_l_enabled() -> bool:
    raw = (os.environ.get("SAWII_FAST_L") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def fast_l_config() -> tuple[str, str, str]:
    base = (os.environ.get("SAWII_FAST_L_BASE_URL") or DEFAULT_BASE).rstrip("/")
    model = os.environ.get("SAWII_FAST_L_MODEL") or DEFAULT_MODEL
    key = os.environ.get("SAWII_FAST_L_API_KEY") or "local"
    return base, model, key


@dataclass
class FastReply:
    """What the fast path produced. ``speakable`` is False when it abstained."""

    text: str = ""
    ok: bool = False
    abstained: bool = False
    model: str = ""
    host: str = ""
    first_token_ms: int | None = None
    total_ms: int | None = None
    error: str = ""
    clauses: list[str] = field(default_factory=list)


class FastL:
    """Streaming client for the tiny responder model.

    One persistent httpx client keeps the TCP+TLS handshake and, on llama.cpp,
    the ``keepalive`` slot hot, so the model is never re-loaded on the critical
    path. ``stream`` yields clause-sized pieces as they arrive.
    """

    def __init__(self, http: "httpx.AsyncClient | None" = None):
        self.base, self.model, self.key = fast_l_config()
        self._http = http
        self._owns = http is None
        self.last: FastReply | None = None

    def http(self) -> "httpx.AsyncClient":
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=2.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=8))
            self._owns = True
        return self._http

    @property
    def host_label(self) -> str:
        if "127.0.0.1" in self.base or "localhost" in self.base:
            return f"local:{self.model}"
        return f"http:{self.model}"

    async def warm(self) -> bool:
        """Load the model off the critical path. llama.cpp keeps it resident."""
        return await self._call("hi", max_tokens=6, system=_SYSTEM, temperature=0.0) is not None

    async def _call(self, user_text: str, *, max_tokens: int, system: str,
                    temperature: float, stop_at_clause: bool = False,
                    history: "list[dict] | None" = None) -> FastReply | None:
        msgs: list[dict] = [{"role": "system", "content": system}]
        # Oldest first, capped: a spoken call's recent exchanges are a few
        # hundred tokens, cheap against a warm local model, and enough to make
        # "and you?" resolve to something.
        for row in (history or [])[-HISTORY_TURNS:]:
            role = str(row.get("role") or "")
            content = str(row.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content[:600]})
        msgs.append({"role": "user", "content": user_text[:1200]})
        body = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if "127.0.0.1" in self.base or "localhost" in self.base:
            # llama.cpp only: hold the slot + the loaded weights between calls.
            # Sent ONLY to loopback - a remote OpenAI-compatible provider rejects
            # it outright (Gemini answers 400 "Unknown name \"keepalive\""), and
            # SAWII_FAST_L_BASE_URL must be able to repoint this client at any
            # of them with one variable.
            body["keepalive"] = "30m"
        t0 = time.perf_counter()
        first: int | None = None
        out = ""
        try:
            async with self.http().stream(
                    "POST", f"{self.base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"},
                    json=body, timeout=httpx.Timeout(15.0, connect=2.0)) as r:
                if r.status_code >= 400:
                    raw = (await r.aread())[:160]
                    return FastReply(ok=False, model=self.model, host=self.host_label,
                                     error=f"http{r.status_code}: "
                                           f"{raw.decode('utf-8', 'replace')[:100]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except ValueError:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        if first is None:
                            first = round((time.perf_counter() - t0) * 1000)
                        out += piece
                        if _PASS.match(out.strip()):
                            # It abstained; no reason to read the rest.
                            break
                        if stop_at_clause and len(out) >= MIN_CLAUSE_CHARS:
                            # One clause is enough to start speaking: the person
                            # hears this while the rest of the sentence is still
                            # being generated. Measured on this box the whole
                            # 40-token reply costs ~1.0-2.2 s, and waiting for it
                            # pushed first-audio from ~0.3 s (TTFT) to ~2.5-4 s.
                            # A comma only breaks when it yields a COMPLETE cut
                            # (both sides substantial); otherwise a two-word
                            # fragment like "Ok cool, let's" is spoken as if it
                            # were the whole line.
                            if (_SENTENCE_END.search(out) or _SENTENCE_DONE.search(out.strip())
                                    or clause_boundary(out)):
                                break
        except Exception as exc:  # noqa: BLE001
            return FastReply(ok=False, model=self.model, host=self.host_label,
                             error=f"{type(exc).__name__}")
        total = round((time.perf_counter() - t0) * 1000)
        text = " ".join(out.split())
        if _PASS.match(text):
            return FastReply(ok=True, abstained=True, model=self.model, host=self.host_label,
                             first_token_ms=first, total_ms=total)
        if stop_at_clause:
            # The stream ran past the clause we stopped for; speak only that
            # clause, otherwise the person hears "I am fine, thank".
            cut = clause_boundary(out)
            if cut:
                text = " ".join(out[:cut].split())
        return FastReply(ok=True, text=text, model=self.model, host=self.host_label,
                         first_token_ms=first, total_ms=total)

    async def reply(self, user_text: str,
                    history: "list[dict] | None" = None) -> FastReply:
        """One fast spoken answer, or an abstention. Never raises.

        Turns that need memory, a fact or a tool are abstained here, in code,
        before the model is even asked: measured on this box the 1.5B model
        happily invents a memory when prompted only by its instructions.

        ``history`` is the few most recent exchanges of THIS call, oldest first
        ([{"role","content"}...]). Without it the responder was deaf between
        turns: "tell me more about that" read as a long-term memory request and
        abstained, when the referent had been spoken three seconds earlier. This
        is short-term conversational context, deliberately not recall - a
        retrieval round trip is the very latency this path exists to avoid.
        """
        if needs_deep_path(user_text):
            return FastReply(ok=True, abstained=True, model=self.model,
                            host=self.host_label)
        res = await self._call(user_text, max_tokens=MAX_TOKENS, system=_SYSTEM,
                               temperature=0.4, stop_at_clause=True,
                               history=history)
        if res is None:
            res = FastReply(ok=False, model=self.model, host=self.host_label,
                            error="warm_failed")
        self.last = res
        if not res.ok:
            log.info("fast L unavailable (%s): %s", res.host, res.error)
        return res

    async def aclose(self) -> None:
        """Close only a client we opened ourselves; a shared one is the caller's."""
        if self._http is not None and self._owns:
            await self._http.aclose()
            self._http = None


async def _smoke() -> None:
    import json
    import sys

    fast = FastL()
    for text in ("hey, how is your day going?", "what's the capital of Peru?",
                 "remind me what we talked about last week"):
        r = await fast.reply(text)
        print(json.dumps({"in": text, **r.__dict__}), flush=True)
    await fast.aclose()
    sys.exit(0 if any(r.ok for r in [fast.last]) else 1)


if __name__ == "__main__":
    asyncio.run(_smoke())