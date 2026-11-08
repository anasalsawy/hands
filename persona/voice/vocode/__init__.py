"""Vocode-faithful conversation architecture for Persona companion voice.

Approach (documented for operators): the upstream ``vocode`` PyPI package is
not used. It pulls a heavy Windows-hostile dependency tree and would replace
Persona's DualLoop mind with its own agent stack. Instead this package mirrors
Vocode's core orchestration shapes:

  Transcriber  →  Agent  →  Synthesizer
         \\          |          /
          StreamingConversation

Backends are DeepInfra (MiMo-V2.6-Flash native hear + thinking off,
Inworld realtime-tts-1.5-mini speak by default; TTS_MODEL overrideable).
The Agent is Persona's shared DualLoop mind — dual listen feeds one mind, never
two independent agents.
"""
from .agent import DualLoopAgent
from .base import SynthesisResult, Transcription
from .conversation import StreamingConversation, VoicePipeline, build_pipeline, pipeline_status
from .synthesizer import DeepInfraMiMoSynthesizer
from .transcriber import DeepInfraNativeHear, DualListenTranscriber, HeardUtterance

__all__ = [
    "DeepInfraMiMoSynthesizer",
    "DeepInfraNativeHear",
    "DualListenTranscriber",
    "DualLoopAgent",
    "HeardUtterance",
    "StreamingConversation",
    "SynthesisResult",
    "Transcription",
    "VoicePipeline",
    "build_pipeline",
    "pipeline_status",
]
