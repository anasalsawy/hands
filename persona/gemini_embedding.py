"""Gemini Embedding 2 native API client for shared text/audio semantic vectors.

No credentials are stored in source control. Uses the Google API key supplied by
the runtime environment. Implements the minimal LLM.embed interface expected
by Sawii's existing memory store, which tags saved vectors by model name.
"""
from __future__ import annotations

import base64
import math
import os

import httpx

from .llm import LLM, LLMError

MODEL = "gemini-embedding-2"
ROOT = "https://generativelanguage.googleapis.com/v1beta/models/" + MODEL


class GeminiEmbeddingLLM(LLM):
    name = "gemini-embedding-2"

    def __init__(self, api_key: str, *, timeout: float = 15.0):
        self.api_key = api_key
        self.model = MODEL
        self.timeout = timeout
        self.on_health = None
        self._client = None

    def _http(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise LLMError("Gemini embedding API key is not configured")
        if not texts:
            return []
        payload = {
            "requests": [
                {
                    "model": "models/" + MODEL,
                    "content": {"parts": [{"text": str(text)}]},
                    "outputDimensionality": 768,
                }
                for text in texts
            ]
        }
        try:
            response = await self._http().post(ROOT + ":batchEmbedContents", json=payload)
            response.raise_for_status()
            values = [part["values"] for part in response.json()["embeddings"]]
            if len(values) != len(texts) or any(
                len(v) != 768 or not all(math.isfinite(float(n)) for n in v) for v in values
            ):
                raise ValueError("Unexpected Gemini embedding dimensions or values")
        except Exception as exc:
            # Never print credentials, Google request details or user content.
            if self.on_health is not None:
                self.on_health.fail(type(exc).__name__)
            raise LLMError(f"Gemini embedding failed: {type(exc).__name__}") from None
        if self.on_health is not None:
            self.on_health.ok(extra={"model": MODEL})
        return values

    async def embed_audio(self, audio: bytes, mime: str = "audio/wav") -> list[float]:
        """Optional direct audio query; first-turn ASR text remains supported."""
        if not self.api_key:
            raise LLMError("Gemini embedding API key is not configured")
        if not audio or len(audio) > 8_000_000 or mime not in ("audio/wav", "audio/mpeg", "audio/mp3"):
            raise ValueError("Expected up to 8 MB of WAV or MPEG audio")
        payload = {
            "model": "models/" + MODEL,
            "content": {"parts": [{"inline_data": {
                "mime_type": "audio/mpeg" if mime == "audio/mp3" else mime,
                "data": base64.b64encode(audio).decode("ascii"),
            }}]},
            "outputDimensionality": 768,
        }
        try:
            response = await self._http().post(ROOT + ":embedContent", json=payload)
            response.raise_for_status()
            vector = response.json()["embedding"]["values"]
            if len(vector) != 768 or not all(math.isfinite(float(n)) for n in vector):
                raise ValueError("Unexpected Gemini audio embedding")
        except Exception as exc:
            if self.on_health is not None:
                self.on_health.fail(type(exc).__name__)
            raise LLMError(f"Gemini audio embedding failed: {type(exc).__name__}") from None
        if self.on_health is not None:
            self.on_health.ok(extra={"model": MODEL})
        return vector

    async def smoke_test(self) -> dict:
        """Probe both Google endpoints with synthetic inputs."""
        import io
        import struct
        import wave

        text_vector = (await self.embed(["A steady 440 Hz tone"]))[0]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"".join(
                struct.pack("<h", round(8000 * math.sin(math.tau * 440 * i / 16000)))
                for i in range(16000)
            ))
        audio_vector = await self.embed_audio(buf.getvalue(), "audio/wav")
        if len(text_vector) != 768 or len(audio_vector) != 768:
            raise LLMError("Gemini text/audio dimension mismatch")
        return {"model": self.model, "text_dimensions": 768, "audio_dimensions": 768,
                "native_audio": True}

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()


async def _run_smoke_test() -> None:
    import json

    key = os.getenv("SAWII_GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set SAWII_GOOGLE_API_KEY or GEMINI_API_KEY to run the live probe.")
    client = GeminiEmbeddingLLM(key)
    try:
        print(json.dumps(await client.smoke_test()))
    finally:
        await client.aclose()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_run_smoke_test())
