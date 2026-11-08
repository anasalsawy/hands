"""Dual listen into ONE shared DualLoop mind.

Two ears (L listener + R provisional drafter) hear the same original audio
prefix while the mic is still open. Their private notes become host_system
context for a single DualLoop turn — never a second mind, never split agents.

This is progressive native-audio preparation, not a claim that the provider
supports token-by-token bidirectional audio streaming.
"""
from __future__ import annotations
import asyncio
import json
import time
from .native_input import AudioAttachedLLM, decode_native_audio


class DuplexVoice:
    """Two ears, one DualLoop mind.

    Listener and responder roles rotate after a usable prefix, but both always
    call the same ``mind.voice_native`` model and the final spoken turn still
    goes through DualLoop with both lobes attached to the same recording.
    """

    def __init__(self, mind, pid, emit):
        self.mind, self.pid, self.emit = mind, pid, emit
        self.listener = "L"
        self.responder = "R"
        self.latest = None
        self.task = None
        self.serial = 0
        self.note = None
        self.draft = None

    def offer(self, audio, mime):
        if not getattr(self.mind.voice_native or self.mind.a, "supports_native_audio", False):
            return
        self.serial += 1
        self.latest = (self.serial, bytes(audio), mime)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    async def _run(self):
        seen = 0
        while self.latest and self.latest[0] != seen:
            serial, audio, mime = self.latest
            seen = serial
            try:
                recording = await decode_native_audio(audio, mime)
                # Same native model for both ears — shared DualLoop mind later.
                model = AudioAttachedLLM(self.mind.voice_native or self.mind.a, recording)
                models = {} if self.mind.voice_native is not None else (
                    self.mind.speaking.models(self.pid) if hasattr(self.mind, "speaking") else {})

                async def infer(role, task):
                    options = {"max_tokens": 160, "temperature": 0.2}
                    if models.get(role):
                        options["model"] = models[role]
                    return await model.chat([
                        {"role": "system", "content": task},
                        {"role": "user", "content": (
                            "Listen directly to this original audio prefix. "
                            "It is incomplete and later audio remains authoritative."
                        )},
                    ], **options)

                listening, drafting = await asyncio.gather(
                    infer(self.listener, (
                        "Voice conversation exception: you are the listening lobe of ONE shared "
                        "DualLoop mind. Keep listening while the other prepares a response. "
                        "Return private JSON with intent_ready (boolean), brief (short string), "
                        "and keep_listening=true. Breathing/coughs or incomplete meaning are not "
                        "a ready request. Do not invent words, execute tools or speak publicly."
                    )),
                    infer(self.responder, (
                        "Voice conversation exception: prepare a brief provisional response from "
                        "the audio received so far while the other lobe listens. This is a private "
                        "draft for the shared DualLoop mind, not audible speech. If meaning is "
                        "incomplete or no intelligible speech exists, output only an empty string. "
                        "Never invent missing details or claim tool success."
                    )),
                )
                # A completed older prefix may be useful preparation, but can never replace final audio.
                note = json.loads(listening.text)
                if not isinstance(note, dict):
                    continue
                self.note = {
                    "serial": serial,
                    "at": time.monotonic(),
                    "brief": str(note.get("brief", ""))[:600],
                    "ready": note.get("intent_ready") is True,
                    "source": recording.source,
                }
                self.draft = {
                    "serial": serial,
                    "at": time.monotonic(),
                    "role": self.responder,
                    "text": str(drafting.text or "")[:1200],
                }
                await self.emit({
                    "type": "kernel",
                    "event": {
                        "type": "voice_roles",
                        "listener": self.listener,
                        "responder": self.responder,
                        "preview": True,
                        "intent_ready": self.note["ready"],
                        "shared_mind": "DualLoop",
                        "dual_listen": True,
                    },
                })
            except asyncio.CancelledError:
                raise
            except Exception:
                # A partial WebM fragment may not yet decode. The complete utterance remains intact.
                pass
            # Short pause so progressive dual-listen can start earlier than a 0.9s endpointer.
            await asyncio.sleep(0.05)

    @staticmethod
    def first_speakable(text: str, *, max_len: int = 90) -> str:
        """First short clause of a dual-listen draft for speculative TTS."""
        t = " ".join((text or "").strip().split())
        if not t:
            return ""
        for sep in (". ", "! ", "? ", "; ", ", ", " — ", " - "):
            if sep in t[: max_len + 10]:
                cut = t.find(sep) + (1 if sep.startswith((".", "!", "?")) else 0)
                if cut >= 8:
                    return t[:cut].strip()[:max_len]
        return t[:max_len].rstrip()

    async def finish_prefix(self):
        # Never stall DualLoop on a still-running prefix inference — cancel and
        # keep whatever note/draft already landed. Dual listen accelerates the
        # shared mind; it must not delay first speech.
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), timeout=0.05)
            except asyncio.TimeoutError:
                pass
        self.task = None
        self.latest = None
        note, draft = self.note, self.draft
        self.note = self.draft = None
        usable = bool(note and draft and note["ready"] and draft["text"] and time.monotonic() - draft["at"] < 5)
        role = draft["role"] if usable else "L"
        opener = self.first_speakable(draft["text"]) if usable else ""
        context = (
            "PRIVATE provisional preparation from an earlier audio prefix for the shared DualLoop mind; "
            "the COMPLETE attached audio overrides it:\n"
            + note["brief"]
            + "\nProvisional draft: "
            + draft["text"]
            + ("\nYou already began speaking aloud: " + repr(opener)
               + ". Continue without repeating that opener." if opener else
               "\nPrefer continuing this draft in the first spoken clause when it still fits the complete audio.")
        ) if usable else ""
        self.listener = role
        self.responder = "R" if role == "L" else "L"
        return role, context, opener

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.task = None
        self.latest = None
        self.note = self.draft = None
