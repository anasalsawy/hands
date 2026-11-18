from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field


class LaneConfig(BaseModel):
    """Persona root URL, identity key and a shared host-session identifier."""

    model_config = ConfigDict(frozen=True)
    base_url: str
    api_key: str = Field(repr=False, min_length=1)
    session_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    model: str = "sawii-persona"
    timeout: float = Field(default=120.0, gt=0)


class PersonaLanes:
    """No shared request lock: L and R may remain in flight simultaneously.

    Uses the existing /v1/chat/completions endpoint with explicit lane metadata.
    A missing lane acknowledgement is an error, never a silent mixed-lane fallback.
    """

    def __init__(
        self, config: LaneConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.config = config
        self._clients = {
            lane: httpx.AsyncClient(
                base_url=config.base_url.rstrip("/") + "/",
                timeout=config.timeout,
                transport=transport,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "X-Sawii-Session-Id": config.session_id,
                    "X-Sawii-Lane": lane,
                    "X-Sawii-Client": "inference",
                },
            )
            for lane in ("L", "R")
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()

    def _body(
        self, messages: list[dict], stream: bool, options: dict, lane: str
    ) -> dict:
        if "purpose" in options:
            options["sawii_purpose"] = options.pop("purpose")
        protected = {"messages", "stream", "model", "sawii_session_id", "sawii_lane"}
        if protected.intersection(options):
            raise ValueError("Lane identity and message fields cannot be overridden")
        return {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "sawii_lane": lane,
            "sawii_session_id": self.config.session_id,
            **options,
        }

    @staticmethod
    def _check(response: httpx.Response, lane: str) -> None:
        response.raise_for_status()
        if response.headers.get("X-Sawii-Lane") != lane:
            raise RuntimeError(
                "Persona did not acknowledge the requested lane; legacy mixed endpoint is unsupported"
            )

    async def execute(self, messages: list[dict], **options: Any) -> dict:
        """All planning, screenshots, extraction, verification and tools go to R."""
        response = await self._clients["R"].post(
            "v1/chat/completions", json=self._body(messages, False, options, "R")
        )
        self._check(response, "R")
        return response.json()

    async def conversation(
        self, messages: list[dict], **options: Any
    ) -> AsyncIterator[str]:
        """Stream only L's conversational text; cancellation closes only L."""
        if options.get("tools") or options.get("tool_choice"):
            raise ValueError("Conversation lane cannot own tools")
        async with self._clients["L"].stream(
            "POST", "v1/chat/completions", json=self._body(messages, True, options, "L")
        ) as response:
            self._check(response, "L")
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif not line and data_lines:
                    data = "\n".join(data_lines)
                    data_lines.clear()
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("tool_calls"):
                            raise RuntimeError("L returned tool calls")
                        if delta.get("content"):
                            yield delta["content"]

    async def interrupt(self, text: str, *, lane: str = "R") -> dict:
        """Send guidance via Persona's existing control endpoint."""
        if lane not in ("L", "R"):
            raise ValueError("lane must be L or R")
        response = await self._clients[lane].post(
            "v1/interrupt",
            json={
                "text": text,
                "sawii_lane": lane,
                "sawii_session_id": self.config.session_id,
            },
        )
        response.raise_for_status()
        return response.json()
