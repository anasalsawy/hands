"""Vocode Synthesizer: DeepInfra MiMo-V2.5-tts (speech + chat-audio fallbacks)."""
from __future__ import annotations

import base64
import logging
import time
from typing import Any

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    import httpx2 as httpx  # type: ignore

from ...config import ModelEndpoint, MIMO_TTS
from .base import BaseSynthesizer, SynthesisResult

log = logging.getLogger(__name__)


class DeepInfraMiMoSynthesizer(BaseSynthesizer):
    """OpenAI-compat ``/audio/speech`` first, then Xiaomi chat-completions audio."""

    def __init__(self, tts: ModelEndpoint, voice: str = "Mia",
                 http: "httpx.AsyncClient | None" = None):
        self.tts = tts
        self.voice = voice or "Mia"
        self._http = http
        self._retry_after = 0.0
        self._failures = 0

    def http(self) -> "httpx.AsyncClient":
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60)
        return self._http

    @property
    def available(self) -> bool:
        return bool(self.tts.base_url and self.tts.model and self.tts.api_key
                    and time.monotonic() >= self._retry_after)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.tts.api_key}",
            "Content-Type": "application/json",
        }

    def _model(self) -> str:
        return self.tts.model or MIMO_TTS

    async def create_speech(self, text: str, *, clone: dict | None = None) -> SynthesisResult:
        if not text.strip():
            return SynthesisResult(data=b"", content_type="audio/wav")
        errors: list[str] = []
        raw = await self._try_speech(text, clone=clone, errors=errors)
        if raw is None:
            raw = await self._try_chat_audio(text, errors=errors)
        if raw is None:
            self._failures += 1
            self._retry_after = time.monotonic() + min(300.0, 5.0 * 2 ** min(self._failures, 6))
            raise RuntimeError("MiMo TTS failed: " + " | ".join(errors[:4]))
        self._failures = 0
        self._retry_after = 0.0
        mime = "audio/wav" if raw[:4] == b"RIFF" else "audio/mpeg"
        return SynthesisResult(data=raw, content_type=mime)

    async def _try_speech(self, text: str, *, clone: dict | None, errors: list[str]) -> bytes | None:
        body: dict[str, Any] = {
            "model": self._model(),
            "input": text,
            "voice": self.voice or "Mia",
            "response_format": "wav" if clone else "mp3",
        }
        if clone and clone.get("data"):
            fmt = clone.get("format") or "wav"
            block: dict = {
                "audio": {
                    "data": base64.b64encode(clone["data"]).decode("ascii"),
                    "format": fmt,
                }
            }
            if clone.get("reference_text"):
                block["reference_text"] = clone["reference_text"]
            body["voice"] = {"clone": block}
        # OpenAI-compat base already ends with /v1/openai; append /audio/speech.
        url = f"{self.tts.base_url.rstrip('/')}/audio/speech"
        try:
            r = await self.http().post(url, headers=self._headers(), json=body)
            if r.status_code >= 400:
                errors.append(f"speech HTTP {r.status_code}: {r.text[:240]}")
                return None
            ctype = (r.headers.get("content-type") or "").lower()
            if "json" in ctype:
                errors.append(f"speech JSON: {r.text[:240]}")
                return None
            if r.content:
                return r.content
            errors.append("speech empty body")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"speech {type(exc).__name__}")
        return None

    async def _try_chat_audio(self, text: str, errors: list[str]) -> bytes | None:
        # Xiaomi MiMo TTS chat shape via DeepInfra OpenAI base.
        payload = {
            "model": self._model(),
            "messages": [
                {"role": "user", "content": "Speak the following clearly."},
                {"role": "assistant", "content": text},
            ],
            "audio": {"format": "wav", "voice": self.voice or "Mia"},
            "modalities": ["text", "audio"],
        }
        url = f"{self.tts.base_url.rstrip('/')}/chat/completions"
        try:
            r = await self.http().post(url, headers=self._headers(), json=payload)
            if r.status_code >= 400:
                errors.append(f"chat HTTP {r.status_code}: {r.text[:240]}")
                return None
            return _extract_chat_audio(r)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"chat {type(exc).__name__}")
        return None


def _extract_chat_audio(response: "httpx.Response") -> bytes | None:
    ctype = (response.headers.get("content-type") or "").lower()
    if "json" not in ctype:
        return response.content or None
    data = response.json()
    try:
        msg = data["choices"][0]["message"]
        audio = msg.get("audio") or {}
        if isinstance(audio, dict) and audio.get("data"):
            return base64.b64decode(audio["data"])
        if isinstance(audio, str):
            return base64.b64decode(audio)
    except Exception:  # noqa: BLE001
        pass
    if isinstance(data.get("audio"), str):
        return base64.b64decode(data["audio"])
    return None
