"""Tinker SDK-based Inkling text/tool inference for Sawii's L/R kernel.

Base models and independently fine-tuned sampler checkpoints use the same
official sampling API. Completion-first until token streaming is supported by
Tinker's renderer/sampler; never claim this is server-token streaming.
"""
from __future__ import annotations

import asyncio
import json
import inspect
import logging
import os
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
import re

from .llm import LLM, LLMError, ChatResult, ToolCall

log = logging.getLogger(__name__)

_PRIVATE = re.compile(r"\u27e6[^\u27e7]*\u27e7")
_RAW_MESSAGE = "<|message_model|>"
_RAW_CONTENT = re.compile(r"<\|content_(\w+)\|>")
_RAW_TOKEN = re.compile(r"<\|[a-z_]+\|>")


def _field(obj, key: str):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _last_text(texts: list[str]) -> tuple[str, int]:
    """One sampled turn can hold several text messages; the cookbook joins them
    with no separator, so a draft and its rewrite reached the person as one
    run-on beat. Only the last is the answer; private markers in the earlier
    ones still reach the kernel, ahead of it."""
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return (texts[0] if texts else ""), len(texts)
    private = "".join(m for t in texts[:-1] for m in _PRIVATE.findall(t))
    return private + texts[-1], len(texts)


def final_text(parsed) -> tuple[str, int]:
    """The answer in a parsed Inkling turn, and how many text parts it had."""
    content = _field(parsed, "content") if not isinstance(parsed, (str, list)) else parsed
    if isinstance(content, str) or content is None:
        return content or "", 1 if (content or "").strip() else 0
    return _last_text([str(p.get("text") or "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text"])


def raw_turn(text: str) -> tuple[str, list[ToolCall], int] | None:
    """Text and tool calls from a turn the renderer could not parse.

    The renderer falls back to the decoded tokens when a turn is malformed (a
    tool call with one stray brace was enough), so the person saw
    ``<|message_model|>`` frames and the call inside them never ran.
    """
    if _RAW_MESSAGE not in text and not _RAW_CONTENT.search(text):
        return None
    texts: list[str] = []
    calls: list[ToolCall] = []
    for seg in text.split(_RAW_MESSAGE):
        kind = _RAW_CONTENT.search(seg)
        if kind is None:
            if seg.strip() and "<|" not in seg:
                texts.append(seg)
            continue
        head, body = seg[:kind.start()], seg[kind.end():]
        if kind.group(1) == "text":
            texts.append(_RAW_TOKEN.sub("", body))
        elif kind.group(1).startswith("invoke_tool"):
            try:
                obj, _ = json.JSONDecoder().raw_decode(_RAW_TOKEN.sub("", body).strip())
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name") or head.strip())
            args = obj.get("args", obj.get("arguments", {}))
            if name:
                calls.append(ToolCall(id=f"tinker_raw_{len(calls)}", name=name,
                                      arguments=args if isinstance(args, str) else json.dumps(args or {})))
    answer, parts = _last_text(texts)
    return answer.strip(), calls, parts


def parsed_tool_calls(parsed) -> list[ToolCall]:
    raw = parsed if isinstance(parsed, list) else _field(parsed, "tool_calls")
    calls = []
    for i, item in enumerate(raw or []):
        fn = _field(item, "function") or {}
        name = _field(item, "name") or _field(fn, "name")
        args = _field(item, "arguments") or _field(item, "args") or _field(fn, "arguments") or {}
        if not name:
            continue
        calls.append(ToolCall(id=str(_field(item, "id") or f"tinker_{i}"), name=str(name),
                              arguments=args if isinstance(args, str) else json.dumps(args)))
    return calls


class TinkerInklingLLM(LLM):
    provider = "tinker"
    sampling_mode = "completion"
    supports_native_audio = True
    def __init__(self, *, api_key: str, base_model: str = "thinkingmachines/Inkling-Small",
                 name: str = "Inkling", effort: float = 0.2,
                 sampler_dir: Path | None = None):
        self.name = name
        self.api_key = api_key
        self.base_model = base_model
        self.effort = effort
        self._service = None
        self._samplers: dict = {}
        self._sampler_dir = sampler_dir
        self._reconnected: dict = {}
        self._renderer = None
        self._lock = asyncio.Lock()
        self.on_health = None

    def _handle_path(self, model: str) -> Path | None:
        if self._sampler_dir is None:
            return None
        return self._sampler_dir / (sha256(model.encode()).hexdigest() + ".handle")

    def _save_handle(self, path: Path, sampler) -> None:
        get_handle = getattr(sampler, "get_sampler_handle", None)
        if get_handle is None:
            return
        temporary = None
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(get_handle())
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except (OSError, ValueError, TypeError):
            log.warning("Tinker sampler handle could not be saved")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def _sampler(self, tinker, model: str, rejected=None):
        existing = self._samplers.get(model)
        if existing is not None and existing is not rejected:
            return existing
        path = self._handle_path(model)
        reconnect = getattr(getattr(tinker, "SamplingClient", None), "from_sampler_handle", None)
        if path is not None and reconnect is not None and rejected is None:
            try:
                handle = await asyncio.to_thread(path.read_text)
                sampler = await asyncio.to_thread(reconnect, handle, api_key=self.api_key)
                self._samplers[model] = sampler
                self._reconnected[model] = sampler
                return sampler
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError):
                log.warning("Tinker sampler handle could not be reopened")
        source = {"model_path": model} if model.startswith("tinker://") else {"base_model": model}
        sampler = await self._service.create_sampling_client_async(**source)
        self._samplers[model] = sampler
        self._reconnected.pop(model, None)
        if path is not None:
            await asyncio.to_thread(self._save_handle, path, sampler)
        return sampler

    async def _components(self, model: str, rejected=None):
        try:
            import tinker
            from tinker_cookbook import model_info
            from tinker_cookbook.renderers import get_renderer, get_tokenizer
        except ImportError:
            # Some Cookbook releases export tokenizer only from tokenizer_utils.
            try:
                import tinker
                from tinker_cookbook import model_info
                from tinker_cookbook.renderers import get_renderer
                from tinker_cookbook.tokenizer_utils import get_tokenizer
            except ImportError as exc:
                raise LLMError("Install optional Tinker SDK: pip install -e '.[tinker]'") from exc
        if not self.api_key:
            raise LLMError("TINKER_API_KEY is missing from the Sawii deployment environment")
        async with self._lock:
            if self._renderer is None:
                self._renderer = await asyncio.to_thread(
                    lambda: get_renderer(model_info.get_recommended_renderer_name(self.base_model),
                                         get_tokenizer(self.base_model)))
            if self._service is None:
                self._service = tinker.ServiceClient(api_key=self.api_key)
            return tinker, self._renderer, await self._sampler(tinker, model, rejected)

    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
                   temperature: float = 0.7, max_tokens: int | None = None, json_mode: bool = False,
                   model: str | None = None) -> ChatResult:
        chosen = model or self.base_model
        tinker, renderer, sampler = await self._components(chosen)
        incoming = list(messages)
        if tools:
            from tinker_cookbook.renderers import ToolSpec
            system = str(incoming[0].get("content", "")) if incoming and incoming[0].get("role") == "system" else ""
            non_system = incoming[1:] if incoming and incoming[0].get("role") == "system" else incoming
            defs = [ToolSpec(name=fn["function"]["name"],
                             description=fn["function"].get("description", ""),
                             parameters=fn["function"].get("parameters", {"type": "object", "properties": {}}))
                    for fn in tools if fn.get("function", {}).get("name")]
            incoming = renderer.create_conversation_prefix_with_tools(defs, system_prompt=system)
            incoming.extend(non_system)
        elif json_mode:
            if incoming and incoming[0].get("role") == "system":
                incoming[0] = {**incoming[0], "content": str(incoming[0].get("content", "")) +
                               "\\nOutput one valid JSON object only."}
            else:
                incoming.insert(0, {"role": "system", "content": "Output one valid JSON object only."})
        if any(isinstance(msg.get("content"), list) and any(
                part.get("type") == "input_audio" for part in msg["content"])
               for msg in incoming):
            from tml_renderers import chat
            incoming = chat.OpenAIMessage.from_oss_messages(incoming)
        build_prompt = renderer.build_generation_prompt
        render_options = {"effort": self.effort} if "effort" in inspect.signature(build_prompt).parameters else {}
        prompt = await asyncio.to_thread(build_prompt, incoming, **render_options)
        parameters = tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=renderer.get_stop_sequences())

        async def sample(client):
            return await client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=parameters)

        try:
            response = await sample(sampler)
        except Exception as exc:
            if (self._reconnected.get(chosen) is not sampler
                    or getattr(exc, "status_code", None) not in (404, 410)):
                raise
            _, _, replacement = await self._components(chosen, rejected=sampler)
            response = await sample(replacement)
        tokens = response.sequences[0].tokens
        cached_tokens = getattr(response, "prompt_cache_hit_tokens", None)
        parsed, _ = renderer.parse_response(tokens)
        text, parts = final_text(parsed)
        calls = parsed_tool_calls(parsed)
        recovered = raw_turn(text)
        if recovered is not None:
            text, raw_calls, parts = recovered
            calls = calls or raw_calls
        if self.on_health is not None:
            self.on_health.ok(extra={"model": chosen, "provider": "tinker", "text_parts": parts,
                                     "recovered_raw": recovered is not None,
                                     "prompt_cache_hit_tokens": cached_tokens})
        return ChatResult(text=text, tool_calls=calls, out_tokens=len(tokens),
                          cached_tokens=cached_tokens or 0)

    async def stream(self, messages: list[dict], *, tools: list[dict] | None = None,
                     temperature: float = 0.7, max_tokens: int | None = None,
                     model: str | None = None):
        result = await self.chat(messages, tools=tools, temperature=temperature,
                                 max_tokens=max_tokens, model=model)
        if result.text:
            yield {"type": "text", "text": result.text}
        for call in result.tool_calls:
            yield {"type": "tool_call", "call": call}
        yield {"type": "done", "finish_reason": "tool_calls" if result.tool_calls else "stop",
               "usage": {"completion_tokens": result.out_tokens,
                         "prompt_tokens_details": {"cached_tokens": result.cached_tokens}}}

