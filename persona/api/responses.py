"""OpenAI Responses API adapter backed by Persona's real dual-lobe chat kernel.

One inference service, no host patch. The adapter translates the Responses
message/tool protocol; the existing Mind owns reasoning, memory, C, and dispatch.
"""
from __future__ import annotations

import json
import time
import uuid

from starlette.responses import JSONResponse, StreamingResponse


def _content(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return "" if value is None else str(value)
    parts = []
    for part in value:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind in ("input_image", "image_url"):
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str) or not url.startswith(("data:image/", "https://")):
                continue
            parts.append({"type": "image_url", "image_url": {"url": url}})
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def from_responses_request(body: dict) -> dict:
    """Normalize Responses messages and flat tool declarations into Chat format.

    No silent guessing of previous_response_id transcripts: a host must supply
    the conversation including its own tool call and tool result.
    """
    if "sawii_host_mode" in body:
        raise ValueError("sawii_host_mode was removed; use model sawii-persona")
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    if not isinstance(inputs, list):
        raise ValueError("input must be an array or string")
    messages = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    for m in inputs:
        if not isinstance(m, dict):
            continue
        kind, role = m.get("type"), m.get("role")
        if kind == "function_call":
            name = str(m.get("name", ""))
            call_id = str(m.get("call_id") or m.get("id") or "")
            if name and call_id:
                messages.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": call_id, "type": "function", "function": {
                        "name": name, "arguments": str(m.get("arguments") or "{}")}}]})
        elif kind == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": str(m.get("call_id") or ""),
                             "content": _content(m.get("output", ""))})
        elif role in ("system", "developer", "user", "assistant", "tool"):
            msg = {"role": role, "content": _content(m.get("content", ""))}
            if role == "tool":
                msg["tool_call_id"] = str(m.get("tool_call_id") or m.get("call_id") or "")
            if role == "assistant" and isinstance(m.get("tool_calls"), list):
                msg["tool_calls"] = m["tool_calls"]
            messages.append(msg)
    if not any(m.get("role") == "user" for m in messages):
        raise ValueError("A complete input transcript with a user message is required")
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            tools.append(t)
        elif t.get("type") == "function" and t.get("name"):
            tools.append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
    out = {"model": body.get("model", "sawii-persona"),
           "messages": messages, "stream": bool(body.get("stream", False))}
    if tools:
        out["tools"] = tools
    for k in ("temperature", "sawii_lane", "sawii_session_id", "sawii_steer",
              "sawii_dual", "sawii_interrupt"):
        if k in body:
            out[k] = body[k]
    if "max_output_tokens" in body:
        out["max_tokens"] = body["max_output_tokens"]
    if isinstance(body.get("text"), dict):
        fmt = body["text"].get("format", {})
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            out["response_format"] = {"type": "json_schema", "json_schema": fmt}
    return out


class AdaptedRequest:
    """Expose the same auth/headers but replace only the JSON body."""
    def __init__(self, request, payload: dict):
        self.request = request
        self.headers = request.headers
        self.query_params = request.query_params
        self._body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    async def body(self):
        return self._body

    def __getattr__(self, attr):
        return getattr(self.request, attr)


def _base_response(rid: str, model: str, status: str, output: list[dict]):
    return {
        "id": rid, "object": "response", "created_at": int(time.time()),
        "status": status, "model": model, "output": output,
        "parallel_tool_calls": True,
        "error": None, "incomplete_details": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def _text_item(value: str):
    return {"id": "msg_" + uuid.uuid4().hex[:16], "type": "message",
            "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": value, "annotations": []}]}


def _call_item(call: dict):
    fn = call.get("function") or {}
    return {"id": "fc_" + uuid.uuid4().hex[:16], "type": "function_call",
            "name": str(fn.get("name") or ""), "call_id": str(call.get("id") or ""),
            "arguments": str(fn.get("arguments") or "{}"), "status": "completed"}


def from_chat_response(response: dict):
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = []
    if message.get("content"):
        output.append(_text_item(str(message["content"])))
    output.extend(_call_item(c) for c in message.get("tool_calls") or [])
    out = _base_response("resp_" + uuid.uuid4().hex[:20], response.get("model", "sawii-persona"),
                         "completed", output)
    # Same reason as in from_chat_stream: the verdict is for the user, and a
    # non-streaming client must be able to read it off the response itself.
    if response.get("sawii_verdict") is not None:
        out["sawii_verdict"] = response["sawii_verdict"]
    if response.get("sawii_metrics") is not None:
        out["sawii_metrics"] = response["sawii_metrics"]
    return out


def _event(kind, data):
    return "event: " + kind + "\ndata: " + json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n\n"


async def from_chat_stream(iterator, model="sawii-persona"):
    """Translate Persona's chat SSE as it arrives; preserve output text streaming."""
    rid = "resp_" + uuid.uuid4().hex[:20]
    output = []
    created = _base_response(rid, model, "in_progress", [])
    yield _event("response.created", {"response": created})
    yield _event("response.in_progress", {"response": created})
    text = ""
    text_id = ""
    text_started = False
    tools = {}
    done = False
    pending = ""
    async for chunk in iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", "replace")
        pending += chunk
        while "\n\n" in pending:
            frame, pending = pending.split("\n\n", 1)
            raw = next((line[5:].strip() for line in frame.splitlines()
                        if line.startswith("data:")), "")
            if not raw or raw == "[DONE]":
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            if "error" in data:
                failed = _base_response(rid, model, "failed", output)
                failed["error"] = data["error"]
                yield _event("response.failed", {"response": failed})
                return
            # Persona's own stream carries lanes, tools and the verifier on
            # sawii_* fields. A Responses client that never sees them is
            # structurally blind to the work; forward each as its own frame.
            for field, kind in (("sawii_event", "response.sawii_event"),
                                ("sawii_tool", "response.sawii_tool"),
                                ("sawii_verdict", "response.sawii_verdict")):
                if data.get(field) is not None:
                    yield _event(kind, {field: data[field]})
            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                if not text_started:
                    text_started = True
                    text_id = "msg_" + uuid.uuid4().hex[:16]
                    yield _event("response.output_item.added", {
                        "output_index": 0, "item": {"id": text_id, "type": "message",
                            "role": "assistant", "status": "in_progress", "content": []}})
                    yield _event("response.content_part.added", {
                        "output_index": 0, "content_index": 0, "item_id": text_id,
                        "part": {"type": "output_text", "text": "", "annotations": []}})
                text += piece
                yield _event("response.output_text.delta", {
                    "output_index": 0, "content_index": 0, "item_id": text_id, "delta": piece})
            for c in delta.get("tool_calls") or []:
                idx = int(c.get("index", 0))
                saved = tools.setdefault(idx, {"id": "", "function": {"name": "", "arguments": ""}})
                saved["id"] = saved["id"] or str(c.get("id") or "")
                fn = c.get("function") or {}
                if fn.get("name"):
                    saved["function"]["name"] += str(fn["name"])
                if fn.get("arguments"):
                    saved["function"]["arguments"] += str(fn["arguments"])
            if choice.get("finish_reason"):
                done = True
    if text_started:
        item = _text_item(text)
        item["id"] = text_id
        output.append(item)
        yield _event("response.output_text.done", {
            "output_index": 0, "content_index": 0, "item_id": text_id, "text": text})
        yield _event("response.content_part.done", {
            "output_index": 0, "content_index": 0, "item_id": text_id, "part": item["content"][0]})
        yield _event("response.output_item.done", {"output_index": 0, "item": item})
    for idx, c in sorted(tools.items()):
        item = _call_item(c)
        index = len(output)
        output.append(item)
        partial = {**item, "arguments": "", "status": "in_progress"}
        yield _event("response.output_item.added", {"output_index": index, "item": partial})
        yield _event("response.function_call_arguments.delta", {
            "output_index": index, "item_id": item["id"], "delta": item["arguments"]})
        yield _event("response.function_call_arguments.done", {
            "output_index": index, "item_id": item["id"], "arguments": item["arguments"]})
        yield _event("response.output_item.done", {"output_index": index, "item": item})
    status = "completed" if done else "incomplete"
    final = _base_response(rid, model, status, output)
    if status == "completed":
        yield _event("response.completed", {"response": final})
    else:
        yield _event("response.incomplete", {"response": final})


async def responses_endpoint(request, chat_completions):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("expected JSON object")
        chat_body = from_responses_request(body)
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}},
                            status_code=400)
    response = await chat_completions(AdaptedRequest(request, chat_body))
    if response.status_code != 200:
        return response
    if body.get("stream"):
        return StreamingResponse(from_chat_stream(response.body_iterator),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return JSONResponse(from_chat_response(json.loads(response.body)))
