"""Which provider keys are actually present in this environment (no secrets printed)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

KEYS = [
    "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ELEVENLABS_API_KEY", "AZURE_OPENAI_API_KEY",
    "TINKER_API_KEY", "FEATHERLESS_API_KEY", "DEEPINFRA_API_KEY", "COMPUSE_LLM_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "FIREWORKS_API_KEY",
    "TOGETHER_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY",
    "COHERE_API_KEY", "DEEPSEEK_API_KEY", "NVIDIA_API_KEY", "CEREBRAS_API_KEY",
]

from persona.config import load_dotenv  # noqa: E402

load_dotenv()
print("cwd:", os.getcwd())
for k in KEYS:
    v = os.environ.get(k) or ""
    print(f"{k:24s} {'SET len=' + str(len(v)) + ' prefix=' + v[:7] if v else '-'}")
print()
print("--- SAWII_ vars ---")
for k in sorted(os.environ):
    if k.startswith("SAWII_") or "LOBE" in k or "DUAL" in k or "VOICE" in k or "TTS" in k:
        print(f"{k}={os.environ[k]}")