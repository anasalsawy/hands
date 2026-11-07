"""Legacy opt-in transcription helper; normal native turns never call this."""
from __future__ import annotations

import asyncio

from .native_input import NativeAudio


def native_audio_part(audio: bytes, *, mime: str = "audio/wav") -> dict:
    return NativeAudio.from_wav(audio, mime).part()


async def transcribe_native(audio: bytes, *, model: str, api_key: str,
                            mime: str = "audio/wav", max_tokens: int = 256) -> str:
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is not configured")
    recording = NativeAudio.from_wav(audio, mime)
    try:
        import tinker
        from tinker_cookbook import model_info
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from tml_renderers import chat
    except ImportError as exc:
        raise RuntimeError("Install the optional Tinker audio dependencies: pip install -e '.[tinker]'") from exc

    renderer = await asyncio.to_thread(
        lambda: get_renderer(model_info.get_recommended_renderer_name(model), get_tokenizer(model)))
    messages = chat.OpenAIMessage.from_oss_messages([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this speech exactly. Return only the transcript."},
            recording.part(),
        ],
    }])
    prompt = await asyncio.to_thread(renderer.build_generation_prompt, messages)
    # Pass the configured secret directly. Never mutate process-wide env or
    # send the credential in a model prompt, tool result or log.
    service = tinker.ServiceClient(api_key=api_key)
    sampler = await service.create_sampling_client_async(base_model=model)
    response = await sampler.sample_async(
        prompt=prompt, num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens, temperature=0.0, stop=renderer.get_stop_sequences()))
    message, _ = renderer.parse_response(response.sequences[0].tokens)
    return str(get_text_content(message)).strip()
