"""Lobe A — fast micro-bridge via a real tiny local LLM (Ollama).

This is NOT DualLoop and NOT a canned phrase list. It calls an actual model
(HTTP chat completions against Ollama / OpenAI-compatible llama.cpp) and asks
for one short generic conversational bridge. The spoken text is whatever the
model returns.

Default trial host: Ollama ``qwen2.5:1.5b`` at localhost:11434 (~1–1.5GB Q4,
CPU, no GPU). Same OpenAI-compatible URL shape can later point at a Render
CPU container serving the same GGUF.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    import httpx2 as httpx  # type: ignore

log = logging.getLogger(__name__)

# Real model id — never a phrase table.
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_BASE = "http://127.0.0.1:11434/v1"

_A_SYSTEM = (
    "You are a fast voice micro-responder. The deep answer is being prepared "
    "in parallel. Reply with ONE short generic conversational bridge only "
    "(about 2–8 words). Acknowledge that you heard them and that you are "
    "about to answer. Stay generic and helpful — no persona, no character, "
    "no name, no jokes, no facts, no answer content. Output ONLY the spoken "
    "bridge phrase, nothing else."
)

_CLEAN = re.compile(r"[\s\"'`]+")


@dataclass
class LobeAResult:
    text: str
    model: str
    host: str
    ttft_ms: int | None
    total_ms: int | None
    ok: bool
    error: str = ""


def dual_lobe_voice_enabled() -> bool:
    """Experiment switch for the OLD lobe-A bridge overlay.

    Default OFF. This path targets Ollama at 127.0.0.1:11434, which is not
    installed on this box; leaving it ON by default made every voice turn take
    the `_dual_lobe_native_turn` branch, pay a connect timeout to a dead port,
    and never reach `fast_L.FastL` - a decoy fast path. It also forced
    native-audio mode. Set SAWII_DUAL_LOBE_VOICE=1 to re-enable it, and point
    SAWII_LOBE_A_BASE_URL at a live OpenAI-compatible server when you do.

    The fast reactive responder in use is persona/voice/fast_L.py
    (SAWII_FAST_L_BASE_URL), which needs no overlay.
    """
    raw = (os.environ.get("SAWII_DUAL_LOBE_VOICE")
           or os.environ.get("DUAL_LOBE_VOICE") or "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


def lobe_a_config() -> tuple[str, str, str]:
    base = (os.environ.get("SAWII_LOBE_A_BASE_URL")
            or os.environ.get("LOBE_A_BASE_URL")
            or DEFAULT_BASE).rstrip("/")
    model = (os.environ.get("SAWII_LOBE_A_MODEL")
             or os.environ.get("LOBE_A_MODEL")
             or DEFAULT_MODEL)
    key = (os.environ.get("SAWII_LOBE_A_API_KEY")
           or os.environ.get("LOBE_A_API_KEY")
           or os.environ.get("OLLAMA_API_KEY")
           or "ollama")
    return base, model, key


def _clean_bridge(text: str) -> str:
    t = (text or "").strip()
    t = t.split("\n", 1)[0].strip()
    # Strip wrapping quotes the tiny model sometimes adds.
    if len(t) >= 2 and t[0] in "\"'`“”" and t[-1] in "\"'`“”":
        t = t[1:-1].strip()
    # Hard cap — bridges must stay micro; truncate at a word boundary.
    if len(t) > 80:
        cut = t[:80]
        sp = cut.rfind(" ")
        t = (cut[:sp] if sp > 20 else cut).rstrip(" ,;:")
    return t.strip()


class FastLobeA:
    """HTTP chat client for the tiny bridge model (Ollama OpenAI API)."""

    def __init__(self, http: httpx.AsyncClient | None = None):
        self._http = http
        self._owns_http = http is None
        self.base, self.model, self.key = lobe_a_config()
        self.last: LobeAResult | None = None

    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=2.0))
            self._owns_http = True
        return self._http

    @property
    def host_label(self) -> str:
        if "11434" in self.base or "ollama" in self.base.lower():
            return f"ollama:{self.model}"
        return f"http:{self.model}"

    async def available(self) -> bool:
        """True when the OpenAI-compatible /models endpoint answers."""
        try:
            r = await self.http().get(
                f"{self.base}/models",
                headers={"Authorization": f"Bearer {self.key}"},
                timeout=2.0,
            )
            if r.status_code >= 400:
                return False
            data = r.json() if r.content else {}
            models = data.get("data") or data.get("models") or []
            ids = {str(m.get("id") or m.get("name") or "") for m in models}
            if not ids:
                return True  # endpoint up; pull may still be needed
            return self.model in ids or any(self.model in i for i in ids)
        except Exception:  # noqa: BLE001
            return False

    async def generate_bridge(self, user_text: str) -> LobeAResult:
        """Call the real model for one short bridge. Never returns a canned list pick."""
        user_text = " ".join((user_text or "").strip().split())[:500]
        messages = [
            {"role": "system", "content": _A_SYSTEM},
            {"role": "user", "content": (
                "The person just said:\n"
                + (user_text or "(speech captured; wording may be incomplete)")
                + "\n\nSpeak your bridge now."
            )},
        ]
        url = f"{self.base}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 24,
            "stream": False,
        }
        t0 = time.monotonic()
        ttft_ms: int | None = None
        try:
            r = await self.http().post(
                url,
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            ttft_ms = round((time.monotonic() - t0) * 1000)
            r.raise_for_status()
            data = r.json()
            raw = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                   or "")
            text = _clean_bridge(raw)
            if not text:
                raise RuntimeError("lobe_a_empty_response")
            result = LobeAResult(
                text=text, model=self.model, host=self.host_label,
                ttft_ms=ttft_ms, total_ms=ttft_ms, ok=True,
            )
            self.last = result
            return result
        except Exception as exc:  # noqa: BLE001
            total = round((time.monotonic() - t0) * 1000)
            result = LobeAResult(
                text="", model=self.model, host=self.host_label,
                ttft_ms=ttft_ms, total_ms=total, ok=False,
                error=f"{type(exc).__name__}:{exc}",
            )
            self.last = result
            log.warning("lobe A bridge failed via %s: %s", self.host_label, type(exc).__name__)
            return result

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None
