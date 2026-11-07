"""DeepInfra EmbeddingGemma 2: shared 768D text/audio semantic memory.

API reference: https://deepinfra.com/google/embeddinggemma-2/api
Do not log audio, private text, credentials, or full upstream error bodies.
"""
from __future__ import annotations

import base64
import math
import httpx

from .llm import LLM, LLMError

MODEL = "google/embeddinggemma-2"
URL = "https://api.deepinfra.com/v1/openai/embeddings"


class DeepInfraEmbeddingLLM(LLM):
    name = MODEL

    def __init__(self, api_key: str, *, timeout: float = 20.0):
        self.api_key = api_key
        self.model = MODEL
        self.timeout = timeout
        self.on_health = None
        self._client = None

    def _http(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
        return self._client

    @staticmethod
    def _values(body, count):
        data = sorted(body.get("data") or [], key=lambda item: item.get("index", 0))
        values = [item["embedding"] for item in data]
        if len(values) != count or any(
            len(vec) != 768 or not all(math.isfinite(float(v)) for v in vec)
            for vec in values
        ):
            raise ValueError("Invalid DeepInfra embedding dimensions or values")
        return values

    async def _post(self, body, count):
        if not self.api_key:
            raise LLMError("DeepInfra embedding API key missing")
        try:
            response = await self._http().post(URL, json=body)
            response.raise_for_status()
            values = self._values(response.json(), count)
        except Exception as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                category = f"HTTP {exc.response.status_code}"
            else:
                category = type(exc).__name__
            if self.on_health is not None:
                self.on_health.fail(category)
            raise LLMError(f"DeepInfra embedding failed: {category}") from None
        if self.on_health is not None:
            self.on_health.ok(extra={"model": MODEL})
        return values

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._post(
            {"model": MODEL, "input": [str(t) for t in texts],
             "task": "document", "dimensions": 768, "encoding_format": "float"},
            len(texts),
        )

    async def embed_audio(self, audio: bytes, mime: str = "audio/wav") -> list[float]:
        if not audio or mime != "audio/wav":
            raise ValueError("DeepInfra native audio embedding expects nonempty WAV audio")
        # The API accepts inline 16-kHz mono WAV. DeepInfra caps a clip at 30 seconds.
        content = [{"type": "input_audio", "input_audio": {
            "data": base64.b64encode(audio).decode("ascii"), "format": "wav"}}]
        values = await self._post(
            {"model": MODEL, "input": [content], "dimensions": 768,
             "encoding_format": "float"}, 1)
        return values[0]

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
