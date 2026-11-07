"""Latest-audio listener + responder preparation, isolated from task execution.

This is progressive native-audio inference, not an ASR transcript or a claim
that the model provider supports token-by-token bidirectional audio streaming.
"""
from __future__ import annotations
import asyncio
import json
import time
from .native_input import AudioAttachedLLM, decode_native_audio


class DuplexVoice:
    def __init__(self, mind, pid, emit):
        self.mind,self.pid,self.emit=mind,pid,emit
        self.listener='L';self.responder='R'
        self.latest=None;self.task=None;self.serial=0
        self.note=None;self.draft=None

    def offer(self, audio, mime):
        if not getattr(self.mind.voice_native or self.mind.a,'supports_native_audio',False):return
        self.serial+=1
        self.latest=(self.serial,bytes(audio),mime)
        if self.task is None or self.task.done():
            self.task=asyncio.create_task(self._run())

    async def _run(self):
        seen=0
        while self.latest and self.latest[0]!=seen:
            serial,audio,mime=self.latest;seen=serial
            try:
                recording=await decode_native_audio(audio,mime)
                model=AudioAttachedLLM(self.mind.voice_native or self.mind.a,recording)
                models={} if self.mind.voice_native is not None else (self.mind.speaking.models(self.pid) if hasattr(self.mind,'speaking') else {})
                async def infer(role,task):
                    options={'max_tokens':160,'temperature':0.2}
                    if models.get(role):options['model']=models[role]
                    return await model.chat([
                        {'role':'system','content':task},
                        {'role':'user','content':'Listen directly to this original audio prefix. It is incomplete and later audio remains authoritative.'}],**options)
                listening,drafting=await asyncio.gather(
                    infer(self.listener,'Voice conversation exception: you are the listening lobe. Keep listening while the other prepares a response. Return private JSON with intent_ready (boolean), brief (short string), and keep_listening=true. Breathing/coughs or incomplete meaning are not a ready request. Do not invent words, execute tools or speak publicly.'),
                    infer(self.responder,'Voice conversation exception: prepare a brief provisional response from the audio received so far while the other lobe listens. This is a private draft, not audible speech. If meaning is incomplete or no intelligible speech exists, output only an empty string. Never invent missing details or claim tool success.'))
                # A completed older prefix may be useful preparation, but can never replace final audio.
                note=json.loads(listening.text)
                if not isinstance(note,dict):continue
                self.note={'serial':serial,'at':time.monotonic(),'brief':str(note.get('brief',''))[:600],
                           'ready':note.get('intent_ready') is True,'source':recording.source}
                self.draft={'serial':serial,'at':time.monotonic(),'role':self.responder,
                            'text':str(drafting.text or '')[:1200]}
                await self.emit({'type':'kernel','event':{'type':'voice_roles','listener':self.listener,
                    'responder':self.responder,'preview':True,'intent_ready':self.note['ready']}})
            except asyncio.CancelledError:raise
            except Exception:
                # A partial WebM fragment may not yet decode. The complete utterance remains intact.
                pass
            await asyncio.sleep(.25)

    async def finish_prefix(self):
        if self.task and not self.task.done():
            self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        self.task=None;self.latest=None
        note,draft=self.note,self.draft
        self.note=self.draft=None
        usable=bool(note and draft and note['ready'] and draft['text'] and time.monotonic()-draft['at']<5)
        role=draft['role'] if usable else 'L'
        context=('PRIVATE provisional preparation from an earlier audio prefix; the COMPLETE attached audio overrides it:\n'
                 +note['brief']+'\nProvisional draft: '+draft['text']) if usable else ''
        self.listener=role
        self.responder='R' if role=='L' else 'L'
        return role,context

    async def close(self):
        if self.task:self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        self.task=None;self.latest=None;self.note=self.draft=None
