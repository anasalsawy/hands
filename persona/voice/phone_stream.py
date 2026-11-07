"""Bidirectional Twilio Media Streams bridge for the Companion phone surface.

Twilio sends 8 kHz mu-law audio over one websocket. We keep that websocket
open for the whole call, segment caller speech locally, feed utterances into
the existing VoiceSession on explicit lane L, and stream L's synthesized audio
back over the same Twilio stream.

R/Cline is intentionally not involved in media playback. It stays available for
work/tools/owner consultation while L remains attached to the caller.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import struct
import wave
from typing import Any

from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket, WebSocketDisconnect

from .realtime import VoiceSession, tts_body
from .native_input import AudioAttachedLLM, NativeAudio

_SILENCE_MS = 520
_MIN_SPEECH_MS = 120
_FRAME_MS = 20
_SPEECH_RMS = 420


def _mulaw_decode_byte(u: int) -> int:
    u = (~u) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if sign else sample


def mulaw_to_pcm16(data: bytes) -> bytes:
    return b"".join(struct.pack("<h", max(-32768, min(32767, _mulaw_decode_byte(b)))) for b in data)


def _linear2ulaw(sample: int) -> int:
    # G.711 mu-law encoder (14-bit magnitude with bias), compatible with
    # Twilio's audio/x-mulaw stream format.
    BIAS = 0x84
    CLIP = 32635
    sample = max(-CLIP, min(CLIP, int(sample)))
    sign = 0x80 if sample < 0 else 0
    if sample < 0:
        sample = -sample
    sample += BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _wav_to_mulaw8k(raw: bytes) -> bytes:
    with wave.open(io.BytesIO(raw), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("phone TTS requires 16-bit PCM WAV")
    samples = list(struct.unpack("<" + "h" * (len(frames) // 2), frames))
    if channels > 1:
        samples = [
            sum(samples[i:i + channels]) // channels
            for i in range(0, len(samples), channels)
        ]
    if rate != 8000 and samples:
        out_len = max(1, round(len(samples) * 8000 / rate))
        scale = (len(samples) - 1) / max(1, out_len - 1)
        resampled = []
        for i in range(out_len):
            pos = i * scale
            lo = int(pos)
            hi = min(lo + 1, len(samples) - 1)
            frac = pos - lo
            resampled.append(round(samples[lo] * (1 - frac) + samples[hi] * frac))
        samples = resampled
    return bytes(_linear2ulaw(s) for s in samples)


def _pcm_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    values = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _pcm_to_wav(pcm: bytes, rate: int = 8000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


class _PhoneSpeech:
    """SpeechIO-compatible adapter that forces WAV TTS for deterministic
    transcoding back into Twilio mu-law audio."""

    def __init__(self, speech: Any):
        self.base = speech

    @property
    def has_asr(self) -> bool:
        return self.base.has_asr

    @property
    def has_tts(self) -> bool:
        return self.base.has_tts

    async def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        return await self.base.transcribe(audio, mime)

    async def synthesize(self, text: str, *, clone: dict | None = None):
        body = tts_body(self.base.tts.model, text, voice=self.base.voice, clone=clone)
        body["response_format"] = "wav"
        r = await self.base.http().post(
            f"{self.base.tts.base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.base.tts.api_key}"},
            json=body,
        )
        r.raise_for_status()
        raw = r.content or b""
        if raw[:4] != b"RIFF":
            raise RuntimeError("phone TTS provider did not return PCM WAV")
        return raw, "audio/wav"



async def _native_r_listen(runtime: Any, session: Any, recording: NativeAudio) -> dict:
    """Silent R pass over the exact same original audio given to L."""
    model = getattr(getattr(runtime, "mind", None), "a", None)
    if model is None or not getattr(model, "supports_native_audio", False):
        raise RuntimeError("R native-audio listener is not configured")
    heard = AudioAttachedLLM(model, recording)
    models = getattr(getattr(runtime, "mind", None), "speaking", None)
    selected = models.models(session.persona_id).get("R") if models is not None and session.persona_id is not None else None
    result = await heard.chat([
        {"role": "system", "content": (
            "You are R, the silent worker lobe listening to a live phone call. "
            "You hear the original audio directly. Do not produce a transcript and do not speak to the caller. "
            "Extract only whether external/tool-backed work is required, the goal, constraints, conditions, "
            "approvals/denials, and a short semantic summary. Return JSON only with keys: "
            "needs_worker (boolean), goal (string), summary (string), constraints (array of strings), reason (string). "
            "For greetings, emotional conversation, or anything L can answer conversationally, needs_worker=false."
        )},
        {"role": "user", "content": recording.content()},
    ], temperature=0.0, max_tokens=500, json_mode=True, **({"model": selected} if selected else {}))
    raw = (getattr(result, "text", "") or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        return {"needs_worker": False, "goal": "", "summary": "", "constraints": [],
                "reason": "R native listener returned non-JSON"}
    return {
        "needs_worker": bool(data.get("needs_worker", data.get("needsWorker", False))),
        "goal": str(data.get("goal", ""))[:4000],
        "summary": str(data.get("summary", ""))[:2000],
        "constraints": [str(x)[:500] for x in (data.get("constraints") or [])[:20]],
        "reason": str(data.get("reason", ""))[:1200],
    }

class TwilioMediaStream(WebSocketEndpoint):
    encoding = "text"

    async def on_connect(self, websocket: WebSocket) -> None:
        hub = websocket.app.state.call_hub
        sid = str(websocket.path_params.get("sid", ""))
        token = str(websocket.path_params.get("token", ""))
        session = hub.sessions.get(sid)
        if session is None or not token or token != session.media_token:
            await websocket.close(code=4403)
            return
        runtime = getattr(websocket.app.state, "runtime", None)
        if runtime is None or session.persona_id is None:
            await websocket.close(code=4404)
            return
        try:
            from ..identity import get_persona
            persona = await get_persona(runtime.mind.store, session.persona_id)
        except Exception:
            persona = None
        if not persona:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        websocket.state.call = session
        websocket.state.runtime = runtime
        websocket.state.persona = persona
        websocket.state.stream_sid = ""
        websocket.state.pcm = bytearray()
        websocket.state.speech_ms = 0
        websocket.state.silence_ms = 0
        websocket.state.in_speech = False
        websocket.state.outbound_mime = None
        websocket.state.opening_sent = False
        websocket.state.listener_tasks = set()

        async def send_json(frame: dict) -> None:
            # VoiceSession emits text/metadata here. Twilio only needs stream
            # control; transcript events are mirrored into CallSession for UI.
            typ = frame.get("type")
            if typ == "delta" and frame.get("text"):
                session.push("you_partial", str(frame["text"]))
            elif typ == "transcript" and frame.get("text"):
                # Caption/history only. Native phone execution never routes
                # through this text; L and R already received original audio.
                session.push("them", str(frame["text"]))
            elif typ == "audio":
                websocket.state.outbound_mime = frame.get("mime")
            elif typ == "done" and frame.get("text"):
                session.push("you", str(frame["text"]))

        async def send_bytes(blob: bytes) -> None:
            stream_sid = websocket.state.stream_sid
            if not stream_sid:
                return
            audio = _wav_to_mulaw8k(blob)
            # Twilio accepts arbitrary media payload sizes, but 20 ms frames
            # keep interruption/clear latency predictable.
            for i in range(0, len(audio), 160):
                if websocket.client_state.name != "CONNECTED" or getattr(websocket.state, "voice", None).interrupted:
                    return
                payload = base64.b64encode(audio[i:i + 160]).decode("ascii")
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }))
                await asyncio.sleep(0.02)

        voice = VoiceSession(
            runtime.mind,
            persona,
            _PhoneSpeech(runtime.speech),
            send_json,
            send_bytes,
            lane="L",
            playback_ack=False,
            tz=getattr(session, "tz", ""),
            force_native_audio=True,
        )
        websocket.state.voice = voice
        session._voice = voice  # type: ignore[attr-defined] -- trusted worker result injection
        session.push("sys", "Twilio realtime audio websocket ready")

    async def _clear_playback(self, websocket: WebSocket) -> None:
        sid = websocket.state.stream_sid
        if sid:
            await websocket.send_text(json.dumps({"event": "clear", "streamSid": sid}))

    async def _flush_utterance(self, websocket: WebSocket) -> None:
        pcm = bytes(websocket.state.pcm)
        websocket.state.pcm.clear()
        websocket.state.speech_ms = 0
        websocket.state.silence_ms = 0
        websocket.state.in_speech = False
        if not pcm:
            return
        wav = _pcm_to_wav(pcm)
        recording = NativeAudio.from_wav(wav)
        voice = websocket.state.voice

        # True dual listening: L and R receive the same original NativeAudio.
        # L owns the caller-facing reply. R is silent and only extracts work.
        async def feed_l() -> None:
            await voice.on_audio(wav)
            await voice.on_end_utterance("audio/wav")

        async def feed_r() -> None:
            try:
                worker = await _native_r_listen(websocket.state.runtime, websocket.state.call, recording)
                websocket.state.call.push("r_listen", worker.get("summary") or worker.get("reason") or "listened")
                if not worker.get("needs_worker"):
                    return
                from .calls import _companion_phone_event
                await _companion_phone_event(
                    websocket, websocket.state.call, "call_worker_intent",
                    text=json.dumps(worker, ensure_ascii=False),
                    actor=websocket.state.call.peer_number or websocket.state.call.to_number,
                    metadata={"native_audio": True, "audio_sha256": recording.source["sha256"]},
                )
            except Exception as exc:  # noqa: BLE001
                websocket.state.call.push("sys", f"R native listener unavailable: {type(exc).__name__}")

        # Never block the websocket receive loop on inference. Both listeners
        # run concurrently while Twilio can immediately deliver more audio.
        for coro in (feed_l(), feed_r()):
            task = asyncio.create_task(coro)
            websocket.state.listener_tasks.add(task)
            task.add_done_callback(websocket.state.listener_tasks.discard)

    async def on_receive(self, websocket: WebSocket, data: str) -> None:
        try:
            frame = json.loads(data)
        except Exception:
            return
        event = frame.get("event")
        if event == "start":
            start = frame.get("start") or {}
            websocket.state.stream_sid = str(frame.get("streamSid") or start.get("streamSid") or "")
            session = websocket.state.call
            call_sid = str(start.get("callSid") or "")
            if call_sid and call_sid != session.sid:
                await websocket.close(code=4403)
                return
            if not websocket.state.opening_sent:
                websocket.state.opening_sent = True
                opener = session._next_say or f"Hi, it's {session.persona_name or 'Maya'}."
                asyncio.create_task(websocket.state.voice.speak_text(opener, kind="opening"))
            return

        if event == "media":
            payload = ((frame.get("media") or {}).get("payload") or "")
            if not payload:
                return
            try:
                mulaw = base64.b64decode(payload)
            except Exception:
                return
            pcm = mulaw_to_pcm16(mulaw)
            rms = _pcm_rms(pcm)
            speech = rms >= _SPEECH_RMS
            st = websocket.state
            if speech:
                if not st.in_speech:
                    st.in_speech = True
                    st.silence_ms = 0
                    voice = st.voice
                    if voice.speaking:
                        await voice.on_interrupt()
                        await self._clear_playback(websocket)
                        st.call.push("barge", "caller interrupted")
                st.speech_ms += _FRAME_MS
                st.pcm.extend(pcm)
            elif st.in_speech:
                st.silence_ms += _FRAME_MS
                st.pcm.extend(pcm)
                if st.silence_ms >= _SILENCE_MS:
                    if st.speech_ms >= _MIN_SPEECH_MS:
                        await self._flush_utterance(websocket)
                    else:
                        st.pcm.clear()
                        st.in_speech = False
                        st.speech_ms = st.silence_ms = 0
            return

        if event == "stop":
            if websocket.state.in_speech and websocket.state.speech_ms >= _MIN_SPEECH_MS:
                await self._flush_utterance(websocket)
            websocket.state.call.push("sys", "Twilio media stream stopped")

    async def on_disconnect(self, websocket: WebSocket, close_code: int) -> None:
        voice = getattr(websocket.state, "voice", None)
        if voice is not None:
            try:
                await voice.on_interrupt()
            except Exception:
                pass
        tasks = list(getattr(websocket.state, "listener_tasks", ()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        call = getattr(websocket.state, "call", None)
        if call is not None:
            call.push("sys", f"Twilio media websocket disconnected ({close_code})")
