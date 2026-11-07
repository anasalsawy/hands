"""The dual loop as an OpenAI-compatible endpoint.

This is the reason the kernel exists. A loop that only the companion app can
call is a feature; a loop that answers ``POST /v1/chat/completions`` is a model.
Anything that already talks to an OpenAI-compatible server - Open WebUI,
LangChain, an agent framework, a phone app - points its base URL at this and gets
the alternating mind without knowing there are two passes behind it.

Two things are deliberately *not* here:

* **No persona, no memory, no database.** Those are capabilities. A host that
  wants them builds a ``CapabilityBundle`` and passes a ``loop_factory``; a host
  that wants nothing gets a working mind from an API key alone.
* **No new wire format.** The chunks are ordinary ``chat.completion.chunk``
  objects. The handoff is invisible on the wire, which is the entire design goal
  expressed in bytes: the client cannot tell it is watching two passes.

The trace of who said what is offered as an *extra* field on the final chunk
(``sawii_dual``), which OpenAI clients ignore and curious hosts can read.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable

from .caps import CapabilityBundle
from .dual import DualLoop

log = logging.getLogger(__name__)

LoopFactory = Callable[[dict], DualLoop]


# ---------------------------------------------------------------------------
# request -> loop input
# ---------------------------------------------------------------------------
def _content_to_text(content: Any) -> str:
    """OpenAI content is a string or a list of parts. Both mean text to us."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Parts concatenate as written; a client that wants a space puts one in
        # the part. Joining with " " would invent one that was never typed.
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def split_messages(messages: list[dict]) -> tuple[str, list[dict], str]:
    """Return ``(user_text, history, system)`` from an OpenAI ``messages`` array.

    The last user message is what the loop thinks about; everything before it is
    history it inherits. System messages join the loop's own system prompt rather
    than replacing it - the writings are what make the mind, and a client must
    not be able to erase them by sending its own system line.
    """
    system_parts: list[str] = []
    history: list[dict] = []
    user_text = ""

    def flush() -> None:
        """Move the pending user turn into history, in order.

        A user message is only *pending* until we know whether another one
        follows. Flushing it lazily but at the wrong moment is what reorders a
        transcript, so every non-user turn flushes first.
        """
        nonlocal user_text
        if user_text:
            history.append({"role": "user", "content": user_text})
            user_text = ""

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = _content_to_text(m.get("content"))
        if role == "system":
            if text.strip():
                system_parts.append(text.strip())
        elif role == "user":
            flush()
            user_text = text
        elif role == "assistant":
            flush()
            if text.strip():
                history.append({"role": "assistant", "content": text})
        # tool / function messages: the loop drives its own hands, so a client's
        # tool results arrive as ordinary context instead of a second protocol.
        elif role in ("tool", "function"):
            flush()
            if text.strip():
                history.append({"role": "user", "content": f"[tool result] {text}"})
    return user_text, history, "\n\n".join(system_parts)


# ---------------------------------------------------------------------------
# loop output -> OpenAI chunks
# ---------------------------------------------------------------------------
def _chunk(cid: str, created: int, model: str, *, delta: dict, finish: str | None = None,
           extra: dict | None = None) -> dict:
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if extra:
        body.update(extra)
    return body


async def stream_chunks(loop: DualLoop, user_text: str, *, history: list[dict] | None = None,
                        model: str = "sawii-dual", cid: str | None = None,
                        created: int | None = None, trace: bool = True) -> AsyncIterator[dict]:
    """Yield OpenAI chunk objects as the dual loop speaks.

    The loop emits the *whole reply so far* each time (that is what keeps a
    handoff from looking like a restart). OpenAI deltas are increments, so we
    send only the new tail. If the buffer somehow got shorter, we send nothing
    rather than a negative delta.

    We drive ``run`` rather than ``stream`` so the finished ``DualResult`` is
    available for the last chunk - the beat trace rides there as ``sawii_dual``,
    which OpenAI clients ignore and curious hosts can read.
    """
    import asyncio

    cid = cid or ("chatcmpl-" + uuid.uuid4().hex[:24])
    created = created if created is not None else int(time.time())
    q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(loop.run(user_text, history=history, emit=q.put))
    sent = 0
    try:
        while True:
            item = await q.get()
            if item is None:
                break
            if not isinstance(item, str) or len(item) <= sent:
                continue
            delta, sent = item[sent:], len(item)
            yield _chunk(cid, created, model, delta={"content": delta})
    finally:
        res = await task

    extra = None
    if trace:
        extra = {"sawii_dual": {
            "switches": res.switches,
            "ms": round(res.ms),
            "beats": [{"writing": b.writing, "ms": round(b.ms), "ttft": round(b.ttft),
                       "tools": [c["name"] for c in b.tool_calls],
                       "verdict": bool(b.verdict)} for b in res.beats],
        }}
    yield _chunk(cid, created, model, delta={}, finish="stop", extra=extra)
    yield {"done": True}


def trace_of(res) -> dict:
    """The beat record, for hosts that want to see the two passes on the wire."""
    return {
        "switches": res.switches,
        "ms": round(res.ms),
        "beats": [{"writing": b.writing, "ms": round(b.ms), "ttft": round(b.ttft),
                   "tools": [c["name"] for c in b.tool_calls],
                   "verdict": bool(b.verdict)} for b in res.beats],
    }


async def completion_body(loop: DualLoop, user_text: str, *, history: list[dict] | None = None,
                          model: str = "sawii-dual", cid: str | None = None,
                          created: int | None = None, trace: bool = True) -> dict:
    """Non-streaming ``chat.completion``, with the beat trace attached."""
    cid = cid or ("chatcmpl-" + uuid.uuid4().hex[:24])
    created = created if created is not None else int(time.time())
    res = await loop.run(user_text, history=history)
    body = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": res.text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if trace:
        body["sawii_dual"] = trace_of(res)
    return body


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------
def create_app(*, loop_factory: LoopFactory, api_key: str = "", model_name: str = "sawii-dual",
               caps_factory: Callable[[dict], CapabilityBundle] | None = None):
    """A minimal ASGI app exposing the loop as a model.

    ``loop_factory(payload) -> DualLoop`` is the only required piece, and the
    simplest legal one is::

        lambda payload: DualLoop(llm, system=SYSTEM)

    which is a complete, self-sufficient product: an endpoint any client can use,
    with memory and tools available to hosts that ask for them via
    ``caps_factory``.

    Endpoints: ``GET /v1/models``, ``POST /v1/chat/completions`` (stream and
    non-stream). Auth is a single optional bearer key - this is a door, not an
    identity system; a host that needs real accounts puts one in front.
    """
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    def _err(status: int, msg: str) -> JSONResponse:
        return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status_code=status)

    def _authed(request: Request) -> bool:
        if not api_key:
            return True
        got = request.headers.get("authorization", "")
        return got.removeprefix("Bearer ").strip() == api_key

    async def models(request: Request):
        if not _authed(request):
            return _err(401, "invalid key")
        return JSONResponse({"object": "list",
                             "data": [{"id": model_name, "object": "model", "owned_by": "sawii"}]})

    async def chat_completions(request: Request):
        if not _authed(request):
            return _err(401, "invalid key")
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "invalid json")
        msgs = data.get("messages") or []
        user_text, history, client_system = split_messages(msgs)
        if not user_text.strip():
            return _err(400, "no user message")

        try:
            loop = loop_factory(data)
        except Exception as exc:  # noqa: BLE001
            log.exception("loop_factory failed")
            return _err(500, f"could not build the loop: {type(exc).__name__}")
        if client_system and getattr(loop, "system", ""):
            # Client context is appended to the mind's own writing, never swapped
            # in for it.
            loop.system = f"{loop.system}\n\n[from the app]\n{client_system}"
        if caps_factory is not None:
            try:
                loop.caps = caps_factory(data) or loop.caps
            except Exception as exc:  # noqa: BLE001 - a broken host wiring must not kill the reply
                log.warning("caps_factory failed: %s", exc)

        model = str(data.get("model") or model_name)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if data.get("stream"):
            async def gen():
                try:
                    async for ev in stream_chunks(loop, user_text, history=history,
                                                  model=model, cid=cid, created=created):
                        if ev.get("done"):
                            yield "data: [DONE]\n\n"
                        else:
                            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except Exception as exc:  # noqa: BLE001
                    log.exception("dual stream failed")
                    yield "data: " + json.dumps({"error": {"message": type(exc).__name__}}) + "\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        try:
            body = await completion_body(loop, user_text, history=history, model=model, cid=cid, created=created)
        except Exception as exc:  # noqa: BLE001
            log.exception("dual completion failed")
            return _err(500, f"internal error: {type(exc).__name__}")
        return JSONResponse(body)

    return Starlette(routes=[
        Route("/v1/models", models),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ])


__all__ = ["create_app", "split_messages", "stream_chunks", "completion_body", "trace_of"]