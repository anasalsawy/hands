"""OpenAI-compatible model clients (chat, streaming with tool calls, embeddings).

Every lobe talks through :class:`LLM`. Tests swap in :class:`ScriptedLLM`.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

try:  # production dependency
    import httpx  # type: ignore
except ImportError:  # pragma: no cover - sandbox fallback (same API)
    import httpx2 as httpx  # type: ignore

from .config import ModelEndpoint


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""

    def args(self) -> dict:
        try:
            v = json.loads(self.arguments or "{}")
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict | None = None
    # Token accounting. Previously the provider's `usage` block reached us inside
    # `raw` and was thrown away, so cost was unmeasurable and "is the picture
    # cheap?" was unanswerable. `cached_tokens` is the one that proves whether
    # prefix caching actually fires: it is billed at a discount and grows only if
    # the stable prefix stays byte-identical across turns.
    in_tokens: int = 0
    out_tokens: int = 0
    cached_tokens: int = 0


def usage_of(data: dict | None) -> tuple[int, int, int]:
    """(prompt, completion, cached) tokens from a chat completion body.

    Providers disagree about names and omit the block entirely when streaming
    without `stream_options: {include_usage}`. Missing is reported as 0 rather
    than guessed at - a wrong number here is worse than no number.
    """
    usage = (data or {}).get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return (int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            int(details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or usage.get("cache_read_input_tokens") or 0))


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    """Remove inline reasoning blocks some open models emit."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:  # unterminated opening already streamed
        text = text.split("</think>", 1)[1]
    return text.strip()


class LLM:
    """Base interface.

    ``on_health`` is an optional :class:`subsystems.SubsystemHealth` the LLM
    reports to on every call: :meth:`ok` on a successful response, :meth:`fail`
    with the exception text otherwise. Wiring is done at Mind construction so
    each lobe's calls land on their own named subsystem (a_llm, b_llm, c_llm).
    """

    name: str = "llm"
    on_health = None  # type: ignore[assignment]

    def with_model(self, model: str) -> "LLM":
        """Same client, same credentials, one different model id, as a copy.

        Only ever fed model ids from config.CURSOR_MODEL_ALIASES: the Cursor
        alias path swaps the cheap coding model for one host turn without
        mutating the shared adapter. Subclasses with an endpoint (the ones
        that put the id on the wire) override this.
        """
        twin = copy.copy(self)
        twin.model = model
        return twin

    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
                   temperature: float = 0.7, max_tokens: int | None = None,
                   json_mode: bool = False, model: str | None = None) -> ChatResult:
        raise NotImplementedError

    async def stream(self, messages: list[dict], *, tools: list[dict] | None = None,
                     temperature: float = 0.7, max_tokens: int | None = None,
                     model: str | None = None) -> AsyncIterator[dict]:
        result = await self.chat(messages, tools=tools, temperature=temperature,
                                 max_tokens=max_tokens, model=model)
        if result.text:
            yield {"type": "text", "text": result.text}
        for call in result.tool_calls:
            yield {"type": "tool_call", "call": call}
        yield {"type": "done", "finish_reason": result.finish_reason}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAICompatLLM(LLM):
    def __init__(self, endpoint: ModelEndpoint, *, name: str = "llm", client: "httpx.AsyncClient | None" = None):
        self.endpoint = endpoint
        self.name = name
        self._client = client
        # Optional runtime metrics. Reset between runs by callers who care.
        # Keys: n_calls, n_errors, total_ms, last_status.
        self.call_stats: dict = {"n_calls": 0, "n_errors": 0, "total_ms": 0.0,
                                 "last_status": 0, "last_error": "",
                                 "in_tokens": 0, "out_tokens": 0, "cached_tokens": 0}

    def reset_stats(self) -> None:
        self.call_stats = {"n_calls": 0, "n_errors": 0, "total_ms": 0.0,
                           "last_status": 0, "last_error": "",
                           "in_tokens": 0, "out_tokens": 0, "cached_tokens": 0}

    @property
    def model(self) -> str:
        return self.endpoint.model

    def with_model(self, model: str) -> "OpenAICompatLLM":
        """Endpoint snapshot with a new model id; the client and key are shared.

        The shallow copy reuses the same httpx pool, so per-alias workers cost
        nothing between turns. `model` on this class is a read-only property
        over `endpoint`, which is why the copy rebuilds the endpoint instead.
        """
        twin = copy.copy(self)
        twin.endpoint = dataclasses.replace(self.endpoint, model=model)
        return twin

    def _http(self) -> "httpx.AsyncClient":
        if self._client is None:
            # HTTP/2 so Cloudflare sees a modern-browser TLS/ALPN fingerprint
            # alongside the browser headers. If h2 isn't installed, httpx
            # silently falls back to HTTP/1.1, which still passes with the
            # richer headers in _headers().
            try:
                self._client = httpx.AsyncClient(timeout=self.endpoint.timeout, http2=True)
            except Exception:  # noqa: BLE001
                self._client = httpx.AsyncClient(timeout=self.endpoint.timeout)
        return self._client

    def _headers(self) -> dict:
        # Full browser-shaped headers, not vanity: Featherless sits behind
        # Cloudflare, and datacenter egress (Railway) with the default
        # `python-httpx/x.y` agent gets 403'd with error 1010 on /embeddings
        # even when /chat/completions passes. Cloudflare looks at the whole
        # header fingerprint (UA + Accept-Language + sec-ch-ua + Sec-Fetch-*),
        # not UA alone. These together get us through its bot score. This
        # cost us semantic recall in production — do not remove.
        # DeepInfra is a normal OpenAI-compatible service, not Featherless's
        # Cloudflare endpoint. Do not send unrelated website Origin/Referer
        # or browser-client headers to its authenticated API.
        if "api.deepinfra.com" in self.endpoint.base_url.lower():
            h = {"Content-Type": "application/json", "Accept": "application/json"}
            if self.endpoint.api_key:
                h["Authorization"] = f"Bearer {self.endpoint.api_key}"
            return h
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Origin": "https://featherless.ai",
            "Referer": "https://featherless.ai/",
        }
        if self.endpoint.api_key:
            h["Authorization"] = f"Bearer {self.endpoint.api_key}"
        return h

    def _body(self, messages, tools, temperature, max_tokens, *, stream: bool, json_mode: bool = False,
              model: str | None = None) -> dict:
        body: dict[str, Any] = {
            "model": model or self.endpoint.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        # ``max_tokens`` is optional at the wire level: pass None and the body
        # omits it, letting the provider's own ceiling (model context window)
        # apply. Callers that want a cap pass an int.
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if "api.deepinfra.com" in self.endpoint.base_url.lower():
            # Strip Persona-only WAV metadata: DeepInfra's OpenAI audio schema
            # accepts input_audio.data and input_audio.format.
            clean_messages = []
            for message in messages:
                copy = dict(message)
                if isinstance(copy.get("content"), list):
                    copy["content"] = [
                        {**part, "input_audio": {
                            "data": part["input_audio"]["data"],
                            "format": part["input_audio"]["format"]}}
                        if isinstance(part, dict) and part.get("type") == "input_audio"
                           and isinstance(part.get("input_audio"), dict)
                           and "data" in part["input_audio"] and "format" in part["input_audio"]
                        else part for part in copy["content"]]
                clean_messages.append(copy)
            body["messages"] = clean_messages
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if getattr(self.endpoint, "no_think", False) and "api.deepinfra.com" not in self.endpoint.base_url.lower():
            # Ask reasoning models (Qwen3, DeepSeek-R1, ...) to skip the hidden
            # "thinking" stream (delta.reasoning). Measured: 26-74s per reply with
            # thinking on vs ~3s with it off. Providers that do not understand the
            # field ignore it; ones that reject it are handled by the 400-fallback
            # in chat()/stream().
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    async def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=None, json_mode=False,
                   model: str | None = None, _kv_slot: int | None = None) -> ChatResult:
        import time as _time
        if not self.endpoint.configured:
            if self.on_health is not None:
                self.on_health.fail("endpoint not configured")
            self.call_stats["n_errors"] += 1
            self.call_stats["last_error"] = "endpoint not configured"
            raise LLMError(f"{self.name}: endpoint not configured")
        url = f"{self.endpoint.base_url}/chat/completions"
        body = self._body(messages, tools, temperature, max_tokens, stream=False, json_mode=json_mode, model=model)
        if _kv_slot is not None:
            body.update(id_slot=_kv_slot, cache_prompt=True)
        t0 = _time.monotonic()
        try:
            resp = await self._http().post(url, headers=self._headers(), json=body)
            if resp.status_code in (400, 422) and (json_mode or "chat_template_kwargs" in body):
                # Provider rejected an optional field we added (response_format
                # and/or the thinking-off hint). Drop them and retry once.
                body.pop("response_format", None)
                body.pop("chat_template_kwargs", None)
                resp = await self._http().post(url, headers=self._headers(), json=body)
            self.call_stats["last_status"] = resp.status_code
            if resp.status_code >= 400:
                if self.on_health is not None:
                    self.on_health.fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
                self.call_stats["n_errors"] += 1
                self.call_stats["last_error"] = f"HTTP {resp.status_code}"
                raise LLMError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        except LLMError:
            self.call_stats["total_ms"] += (_time.monotonic() - t0) * 1000
            self.call_stats["n_calls"] += 1
            raise
        except Exception as exc:  # noqa: BLE001
            self.call_stats["total_ms"] += (_time.monotonic() - t0) * 1000
            self.call_stats["n_calls"] += 1
            self.call_stats["n_errors"] += 1
            self.call_stats["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
            if self.on_health is not None:
                self.on_health.fail(f"{type(exc).__name__}: {exc}")
            raise
        self.call_stats["total_ms"] += (_time.monotonic() - t0) * 1000
        self.call_stats["n_calls"] += 1
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = [ToolCall(id=c.get("id") or f"call_{i}", name=(c.get("function") or {}).get("name", ""),
                          arguments=(c.get("function") or {}).get("arguments") or "")
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        if self.on_health is not None:
            self.on_health.ok(extra={"model": self.endpoint.model})
        pin, pout, pcached = usage_of(data)
        if _kv_slot is not None and data.get("timings"):
            pcached = int(data["timings"].get("cache_n", pcached))
            self.call_stats["kv_last_prefill_tokens"] = data["timings"].get("prompt_n")
        self.call_stats["in_tokens"] = self.call_stats.get("in_tokens", 0) + pin
        self.call_stats["out_tokens"] = self.call_stats.get("out_tokens", 0) + pout
        self.call_stats["cached_tokens"] = self.call_stats.get("cached_tokens", 0) + pcached
        return ChatResult(text=strip_reasoning(msg.get("content") or ""), tool_calls=calls,
                          finish_reason=choice.get("finish_reason") or "stop", raw=data,
                          in_tokens=pin, out_tokens=pout, cached_tokens=pcached)

    async def stream(self, messages, *, tools=None, temperature=0.7, max_tokens=None,
                     model: str | None = None, _kv_slot: int | None = None) -> AsyncIterator[dict]:
        if not self.endpoint.configured:
            if self.on_health is not None:
                self.on_health.fail("endpoint not configured")
            raise LLMError(f"{self.name}: endpoint not configured")
        url = f"{self.endpoint.base_url}/chat/completions"
        body = self._body(messages, tools, temperature, max_tokens, stream=True, model=model)
        # Without this the provider sends no `usage` block on a stream, so every
        # voice turn was invisible to cost accounting. Providers that do not
        # understand the field are handled by the 400-retry below.
        if _kv_slot is not None:
            body.update(id_slot=_kv_slot, cache_prompt=True)
        body["stream_options"] = {"include_usage": True}
        calls: dict[int, ToolCall] = {}
        finish = "stop"
        in_think = False
        ok = False
        usage: dict = {}
        timings: dict = {}
        try:
            # Two attempts at most: the second drops the thinking-off hint in case
            # the provider rejected the unknown field with a 400.
            for _attempt in (0, 1):
                if _attempt:
                    body.pop("chat_template_kwargs", None)
                    body.pop("stream_options", None)
                async with self._http().stream("POST", url, headers=self._headers(), json=body) as resp:
                    if resp.status_code in (400, 422) and ("chat_template_kwargs" in body
                                                     or "stream_options" in body):
                        await resp.aread()
                        continue
                    if resp.status_code >= 400:
                        text = (await resp.aread()).decode(errors="replace")
                        if self.on_health is not None:
                            self.on_health.fail(f"HTTP {resp.status_code}: {text[:200]}")
                        raise LLMError(f"{self.name}: HTTP {resp.status_code}: {text[:300]}")
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if _kv_slot is not None and chunk.get("timings"):
                            timings = chunk["timings"]
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            # drop <think> sections from models that inline reasoning
                            if "<think>" in piece:
                                in_think = True
                                piece = piece.split("<think>", 1)[0]
                            if in_think:
                                if "</think>" in piece:
                                    in_think = False
                                    piece = piece.split("</think>", 1)[1]
                                else:
                                    piece = ""
                            if piece:
                                yield {"type": "text", "text": piece}
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            call = calls.setdefault(idx, ToolCall(id=tc.get("id") or f"call_{idx}", name=""))
                            fn = tc.get("function") or {}
                            if tc.get("id"):
                                call.id = tc["id"]
                            if fn.get("name"):
                                call.name += fn["name"]
                            if fn.get("arguments"):
                                call.arguments += fn["arguments"]
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                ok = True
                break
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            if self.on_health is not None:
                self.on_health.fail(f"stream {type(exc).__name__}: {exc}")
            raise
        if ok and self.on_health is not None:
            self.on_health.ok(extra={"model": self.endpoint.model})
        if ok:
            pin, pout, pcached = usage_of({"usage": usage})
            if timings:
                pcached = int(timings.get("cache_n", pcached))
                self.call_stats["kv_last_prefill_tokens"] = timings.get("prompt_n")
            self.call_stats["in_tokens"] = self.call_stats.get("in_tokens", 0) + pin
            self.call_stats["out_tokens"] = self.call_stats.get("out_tokens", 0) + pout
            self.call_stats["cached_tokens"] = self.call_stats.get("cached_tokens", 0) + pcached
        for idx in sorted(calls):
            yield {"type": "tool_call", "call": calls[idx]}
        yield {"type": "done", "finish_reason": finish, "usage": usage}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.endpoint.configured:
            if self.on_health is not None:
                self.on_health.fail("endpoint not configured")
            raise LLMError(f"{self.name}: endpoint not configured")
        url = f"{self.endpoint.base_url}/embeddings"
        try:
            resp = await self._http().post(url, headers=self._headers(),
                                           json={"model": self.endpoint.model, "input": texts})
            if resp.status_code >= 400:
                if self.on_health is not None:
                    self.on_health.fail(f"HTTP {resp.status_code}: {resp.text[:200]}")
                raise LLMError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = sorted(resp.json().get("data", []), key=lambda d: d.get("index", 0))
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            if self.on_health is not None:
                self.on_health.fail(f"{type(exc).__name__}: {exc}")
            raise
        if self.on_health is not None:
            self.on_health.ok(extra={"model": self.endpoint.model})
        return [d["embedding"] for d in data]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------

Responder = Callable[[list[dict], dict], "ChatResult | str | Awaitable[ChatResult | str]"]


class ScriptedLLM(LLM):
    """Deterministic fake. ``responder(messages, opts)`` returns text or ChatResult."""

    def __init__(self, responder: Responder | str = "ok", *, name: str = "fake", delay: float = 0.0,
                 chunk: int = 12, chunk_delay: float = 0.0):
        self.responder = responder
        self.name = name
        self.delay = delay
        self.chunk_delay = chunk_delay
        self.chunk = chunk
        self.calls: list[dict] = []
        self.model = name

    async def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=None, json_mode=False,
                   model=None) -> ChatResult:
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode, "model": model})
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.responder, str):
            out: Any = self.responder
        else:
            out = self.responder(messages, {"tools": tools, "json_mode": json_mode, "n": len(self.calls)})
            if asyncio.iscoroutine(out):
                out = await out
        return out if isinstance(out, ChatResult) else ChatResult(text=str(out))

    async def stream(self, messages, *, tools=None, temperature=0.7, max_tokens=None, model=None):
        result = await self.chat(messages, tools=tools, temperature=temperature,
                                 max_tokens=max_tokens, model=model)
        text = result.text
        for i in range(0, len(text), self.chunk):
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield {"type": "text", "text": text[i:i + self.chunk]}
        for call in result.tool_calls:
            yield {"type": "tool_call", "call": call}
        yield {"type": "done", "finish_reason": "tool_calls" if result.tool_calls else "stop"}

    async def embed(self, texts):
        from .embedder import hash_embed
        return [hash_embed(t) for t in texts]


def extract_json(text: str) -> dict | None:
    """First balanced JSON object in a model reply (handles fences / prose)."""
    text = strip_reasoning(text or "")
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        val = json.loads(text[start:i + 1])
                        if isinstance(val, dict):
                            return val
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None
