"""All runtime configuration, read from the environment (prefix SAWII_).

Every model is a plain OpenAI-compatible endpoint, so any lobe can be moved to
another provider (Featherless, Fireworks, Together, a local vLLM) by changing
three variables and nothing else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


_QUOTES = chr(34) + chr(39)


def load_dotenv(path: str = "") -> None:
    """Populate os.environ from a .env file. Existing env vars always win.

    Minimal on purpose: no dependency, no interpolation, no shell semantics.
    Lines are ``KEY=value``; ``#`` starts a comment, surrounding quotes are
    stripped, and an ``export `` prefix is tolerated.
    """
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        for candidate in (
            os.path.join(os.getcwd(), ".env"),
            os.path.abspath(os.path.join(here, "..", "..", ".env")),
        ):
            if os.path.isfile(candidate):
                path = candidate
                break
        else:
            return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value

def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"SAWII_{name}", default)


def _bool(name: str, default: bool) -> bool:
    raw = _env(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = _env(name, "")
    return int(raw) if raw.strip() else default


def _float(name: str, default: float) -> float:
    raw = _env(name, "")
    return float(raw) if raw.strip() else default


FEATHERLESS = "https://api.featherless.ai/v1"
DEEPINFRA = "https://api.deepinfra.com"
DEEPINFRA_OPENAI = "https://api.deepinfra.com/v1/openai"
TINKER_OAI = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
INKLING_SMALL = "thinkingmachines/Inkling-Small"
MIMO_HEAR = "XiaomiMiMo/MiMo-V2.6-Flash"
MIMO_TTS = "XiaomiMiMo/MiMo-V2.5-tts"
QWEN3_TTS = "Qwen/Qwen3-TTS"
# Lowest practical DeepInfra TTS RTT in bake-off (warm p50 ~0.8s).
# Override with SAWII_TTS_MODEL=... to force MiMo/Qwen/etc.
INWORLD_TTS = "inworld-ai/realtime-tts-1.5-mini"

# ------------------------------------------------------------------ provider choice
# Featherless is the authoritative default provider for every Persona lobe.
# DeepInfra is now an EXPLICIT opt-in, never a side effect of a credential being
# present in the environment: the mere existence of DEEPINFRA_API_KEY used to
# repoint TTS, native audio and the Cursor worker at api.deepinfra.com, which
# silently moved lobes onto a provider nobody chose (and onto a suspended
# account). Set SAWII_PROVIDER=deepinfra to route the defaults back there - no
# code change needed - or point one lobe at it with its own SAWII_<PREFIX>_BASE_URL.
PROVIDER_FEATHERLESS = "featherless"
PROVIDER_DEEPINFRA = "deepinfra"


def provider() -> str:
    """The explicitly chosen default provider: featherless unless SAWII_PROVIDER says otherwise."""
    wanted = _env("PROVIDER", "").strip().lower()
    return PROVIDER_DEEPINFRA if wanted == PROVIDER_DEEPINFRA else PROVIDER_FEATHERLESS


def deepinfra_preferred() -> bool:
    """True only when DeepInfra was chosen on purpose (SAWII_PROVIDER=deepinfra).

    A present-but-unused DEEPINFRA_API_KEY means nothing here. An endpoint whose
    own SAWII_<PREFIX>_BASE_URL names api.deepinfra.com still uses DeepInfra for
    that one lobe regardless of this flag - per-lobe override always wins.
    """
    return provider() == PROVIDER_DEEPINFRA


def is_deepinfra_base(base_url) -> bool:
    return "api.deepinfra.com" in (base_url or "").lower()

# ------------------------------------------------------------------ Cursor aliases
# Hard-coded alias table for the Cursor adapter - the CURSOR-DEEPINFRA host
# worker built in mind.py from settings.infer_a (INFER_A envs, already
# DEEPINFRA_API_KEY-aware in _endpoint below). Every id here is served through
# https://api.deepinfra.com/v1/openai and is cheap at coding ("super coding",
# but At DeepSeek-V4.1-Flash money): prices per 1M tokens, verified on
# deepinfra.com 2026-11-06.
#
# The table is a whitelist, not a suggestion: resolve_cursor_alias() answers
# "" for anything that is not one of these exact keys, so a model string
# chosen by an attacker or a host can never be passed through to the provider.
#
# Aliases route ONLY the host/execution lane (R-side motor work) of a turn
# tagged with the persona-cursor-deepinfra surface. Persona's own conversational
# voice and reasoning keep their normal model (lobe A) - the alias changes which
# cheap worker executes, exactly as choose_host_inference's docstring says:
# "An explicit Cursor alias changes model provider, not the Persona mind."
#
# Railway env vars a restart needs to apply this:
#   DEEPINFRA_API_KEY=<key>        required - the service-wide DeepInfra secret.
#                                  Any base_url containing api.deepinfra.com
#                                  takes this key (and it outranks a stale
#                                  provider key). Never IDC /active2maas.
#   SAWII_INFER_A_BASE_URL=https://api.deepinfra.com/v1/openai
#                                  required to turn the feature on at all: unset,
#                                  settings.infer_a stays unconfigured, the
#                                  CURSOR-DEEPINFRA worker is None, and every
#                                  host (cursor included) keeps A exactly as today.
#   SAWII_CURSOR_ALIAS=<alias>     optional - the default for that surface when a
#                                  request names no alias. Must be a key below;
#                                  unset or unknown falls back to sawii-fast
#                                  (deepseek-ai/DeepSeek-V4.1-Flash), never to
#                                  an arbitrary model.
#   SAWII_INFER_A_MODEL=<id>       optional - overrides the worker's own model
#                                  id (what "sawii-persona" on this surface
#                                  gets); an alias in the request still resolves
#                                  strictly through the table, never through it.
CURSOR_DEEPINFRA_BASE = DEEPINFRA_OPENAI
# Price comments per 1M tokens, verified on deepinfra.com 2026-11-06.
CURSOR_MODEL_IDS = (
    "deepseek-ai/DeepSeek-V4.1-Flash",          # $0.098 in / $0.294 out (51% off) - the baseline
    "deepseek-ai/DeepSeek-V4-Flash",            # $0.09 in / $0.18 out
    "deepseek-ai/DeepSeek-V4-Flash-0731",       # $0.06 in / $0.18 out
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",     # fresher cut of the baseline: $0.098 in / $0.294 out
    "XiaomiMiMo/MiMo-V2.6-Flash",               # $0.14 in / $0.28 out
    "zai-org/GLM-5.3-Flash",                    # $0.15 in / $0.50 out
)
# Friendly short names for the same six, so a host can pick without spelling the
# org/model id. Both spellings resolve; nothing outside these rows does.
CURSOR_MODEL_ALIASES = {
    # alias              -> DeepInfra model id                     price / 1M tokens
    "sawii-fast":      "deepseek-ai/DeepSeek-V4.1-Flash",          # $0.098 in / $0.294 out (51% off)
    "sawii-fast-v4":   "deepseek-ai/DeepSeek-V4-Flash",            # $0.09 in / $0.18 out
    "sawii-fast-0731": "deepseek-ai/DeepSeek-V4-Flash-0731",       # $0.06 in / $0.18 out
    "sawii-fast-1017": "deepseek-ai/DeepSeek-V4.1-Flash-1017",     # fresher cut of the default: $0.098 in / $0.294 out
    "mimo-fast":       "XiaomiMiMo/MiMo-V2.6-Flash",               # $0.14 in / $0.28 out
    "glm-flash":       "zai-org/GLM-5.3-Flash",                    # $0.15 in / $0.50 out
}
CURSOR_DEFAULT_ALIAS = "sawii-fast"
CURSOR_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
CURSOR_PUBLIC_MODEL = "sawii-persona"

# The same closed deal sheet, priced for Featherless, which is now the default
# provider. Every id below was read out of GET https://api.featherless.ai/v1/models
# (10,245 served models, 2026-11-08) - none are invented. XiaomiMiMo/MiMo-V2.6-Flash
# is NOT served by Featherless, so "mimo-fast" keeps its alias key and re-points to
# the cheapest fast reasoner Featherless does have; a host naming "mimo-fast" still
# gets a table row, never a passthrough.
CURSOR_FEATHERLESS_MODEL_IDS = (
    "deepseek-ai/DeepSeek-V3.2",               # $0.264 in / $0.41 out - the baseline
    "deepseek-ai/DeepSeek-V3.1-Terminus",      # $0.285 in / $1.00 out
    "deepseek-ai/DeepSeek-V3-0324",            # $0.385 in / $1.625 out
    "Qwen/Qwen3-Coder-Next",                   # $0.16 in / $1.15 out - coding-tuned
    "openai/gpt-oss-120b",                     # $0.15 in / $0.60 out
    "zai-org/GLM-4.6",                         # $0.55 in / $2.20 out
)
CURSOR_FEATHERLESS_ALIASES = {
    # alias              -> Featherless model id
    "sawii-fast":      "deepseek-ai/DeepSeek-V3.2",
    "sawii-fast-v4":   "deepseek-ai/DeepSeek-V3.1-Terminus",
    "sawii-fast-0731": "deepseek-ai/DeepSeek-V3-0324",
    "sawii-fast-1017": "Qwen/Qwen3-Coder-Next",
    "mimo-fast":       "openai/gpt-oss-120b",          # MiMo is not on Featherless
    "glm-flash":       "zai-org/GLM-4.6",
}
CURSOR_FEATHERLESS_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"


def cursor_table(which: str = "") -> tuple[tuple[str, ...], dict[str, str]]:
    """(model ids, alias map) for a provider - featherless unless told otherwise.

    Two closed tables, one per provider, sharing one alias-key space so a host
    never has to know which provider is live. Selection follows SAWII_PROVIDER;
    it is NOT chosen by whichever credential happens to be in the environment.
    """
    which = (which or provider()).strip().lower()
    if which == PROVIDER_DEEPINFRA:
        return CURSOR_MODEL_IDS, CURSOR_MODEL_ALIASES
    return CURSOR_FEATHERLESS_MODEL_IDS, CURSOR_FEATHERLESS_ALIASES


def resolve_cursor_alias(name, which: str = "") -> str:
    """Alias key or table model id -> a model id on the ACTIVE provider. "" = not in the table.

    Strictly a lookup against one closed table (see cursor_table): a friendly alias
    resolves to its row, a model id that is one of the active rows resolves to
    itself, and every other string - any live model id this deal sheet does not
    name, including a row belonging to the OTHER provider - resolves to "".
    Nothing is ever passed through, and switching the default provider does not
    widen what a host can name.
    """
    if not isinstance(name, str):
        return ""
    ids, aliases = cursor_table(which)
    key = name.strip()
    hit = aliases.get(key, "")
    if hit:
        return hit
    return key if key in ids else ""


def cursor_default_alias(which: str = "") -> str:
    """SAWII_CURSOR_ALIAS when it names a real row of the active table, else the baseline."""
    wanted = _env("CURSOR_ALIAS", "").strip()
    return wanted if resolve_cursor_alias(wanted, which) else CURSOR_DEFAULT_ALIAS


def cursor_default_model(which: str = "") -> str:
    """The model id behind the default alias - the cheap coding baseline of the active table."""
    which = (which or provider()).strip().lower()
    fallback = (CURSOR_DEFAULT_MODEL if which == PROVIDER_DEEPINFRA
                else CURSOR_FEATHERLESS_DEFAULT_MODEL)
    return resolve_cursor_alias(cursor_default_alias(which), which) or fallback


def cursor_alias_for(value) -> str:
    """A cursor request's model string -> a table model id, "" for anything else.

    "sawii-persona" (or no model at all) is NOT run through here: the gate in
    api/app.py accepts the public pin outright and the worker keeps its own
    configured model, which config already defaults to the baseline alias
    (deepseek-ai/DeepSeek-V4.1-Flash) - the "default alias when unset" leg.
    What reaches here is an explicit model string: only a row of the table
    resolves, an arbitrary provider id returns "" and the caller 400s it.
    """
    return resolve_cursor_alias(value)


def cursor_alias_for(value, base_url: str = "") -> str:
    """A cursor request's model string -> a table model id, "" for anything else.

    "sawii-persona" (or no model at all) is NOT run through here: the gate in
    api/app.py accepts the public pin outright and the worker keeps its own
    configured model, which config already defaults to the baseline alias of the
    active table - the "default alias when unset" leg.
    What reaches here is an explicit model string: only a row of the table
    resolves, an arbitrary provider id returns "" and the caller 400s it.
    Pass the worker's base_url to resolve against that provider's table.
    """
    which = PROVIDER_DEEPINFRA if is_deepinfra_base(base_url) else ""
    return resolve_cursor_alias(value, which)


def cursor_worker_endpoint(inherit: ModelEndpoint) -> ModelEndpoint:
    """The endpoint the Cursor host worker is built from (INFER_A prefix).

    The R-side execution lane now runs on Featherless by default, exactly like
    every other lobe. Activation no longer depends on naming an
    api.deepinfra.com base: the worker is configured whenever a base resolves
    (SAWII_INFER_A_BASE_URL, else A's own provider), and its model comes from
    the ACTIVE alias table - deepseek-ai/DeepSeek-V3.2 on Featherless,
    deepseek-ai/DeepSeek-V4.1-Flash under SAWII_PROVIDER=deepinfra - unless
    SAWII_CURSOR_ALIAS names another row or SAWII_INFER_A_MODEL overrides it.
    The whitelist property is unchanged: only table rows can ever be selected.
    SAWII_INFER_A_BASE_URL=https://api.deepinfra.com/v1/openai still pins this
    one lobe to DeepInfra on its own, with no other lobe following.
    """
    base_url = _env("INFER_A_BASE_URL", "") or (inherit.base_url if inherit else "")
    which = PROVIDER_DEEPINFRA if is_deepinfra_base(base_url) else provider()
    return _endpoint("INFER_A", model=cursor_default_model(which),
                     base=(CURSOR_DEEPINFRA_BASE if which == PROVIDER_DEEPINFRA else FEATHERLESS),
                     timeout=90)


@dataclass
class ModelEndpoint:
    """One OpenAI-compatible model: base URL, key, model id."""

    base_url: str
    api_key: str
    model: str
    timeout: float | None = None
    # Reasoning models (Qwen3, DeepSeek-R1, ...) spend tens of seconds emitting a
    # hidden "thinking" stream (delta.reasoning) before the answer. That thinking
    # is the single biggest latency cost in a turn - measured at 26-74s per reply
    # vs ~3s with it off. We ask the provider to skip it; if the provider does not
    # support the flag we silently fall back (see llm.py).
    no_think: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


def _endpoint(prefix: str, *, model: str, base: str = FEATHERLESS, timeout: float | None = None,
              inherit: "ModelEndpoint | None" = None) -> ModelEndpoint:
    base_url = _env(f"{prefix}_BASE_URL", "") or (inherit.base_url if inherit else base)
    supplied_key = _env(f"{prefix}_API_KEY", "")
    # Keys never cross providers. A base naming api.deepinfra.com takes the
    # dedicated DeepInfra credential (it outranks a stale provider-specific key
    # or an unresolved Railway reference string); any other base is treated as
    # Featherless and takes FEATHERLESS_API_KEY. The old rule fired on the mere
    # presence of DEEPINFRA_API_KEY and could hand that secret to a Featherless
    # base, so the URL - not the environment - now decides, and a Featherless
    # call can never carry the DeepInfra key nor a DeepInfra call the Featherless one.
    if is_deepinfra_base(base_url):
        api_key = os.environ.get("DEEPINFRA_API_KEY", "") or supplied_key
    else:
        # Only borrow an inherited credential when the inherited endpoint is on
        # this same (non-DeepInfra) provider.
        inherit_key = "" if inherit is None or is_deepinfra_base(inherit.base_url) else inherit.api_key
        api_key = supplied_key or inherit_key or _env("FEATHERLESS_API_KEY", "")
    return ModelEndpoint(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=_env(f"{prefix}_MODEL", model),
        timeout=(float(_env(f"{prefix}_TIMEOUT")) if _env(f"{prefix}_TIMEOUT").strip() else timeout),
        no_think=_bool("DISABLE_THINKING", True),
    )


def _tts_endpoint() -> ModelEndpoint:
    """Companion speak path. Featherless Chatterbox unless DeepInfra is chosen.

    Featherless serves ResembleAI/chatterbox and chatterbox-turbo (both probed
    live 2026-11-08), so the default needs no DeepInfra at all. Chatterbox takes
    a named voice and has no "alloy": SAWII_TTS_VOICE selects one (Aaron, Anaya,
    ...). The DeepInfra bake-off still stands for that provider -
      inworld-ai/realtime-tts-1.5-mini ~0.83s p50 (DeepInfra only)
      ResembleAI/chatterbox-turbo      ~0.84s
      Qwen/Qwen3-TTS                   ~1.5s
      XiaomiMiMo/MiMo-V2.5-tts         ~3.7s
    DeepInfra is reached with SAWII_PROVIDER=deepinfra or SAWII_TTS_BASE_URL
    naming api.deepinfra.com. Override the id with SAWII_TTS_MODEL.
    """
    on_deepinfra = deepinfra_preferred()
    default_base = DEEPINFRA_OPENAI if on_deepinfra else FEATHERLESS
    default_model = INWORLD_TTS if on_deepinfra else "ResembleAI/chatterbox"
    base_url = (_env("TTS_BASE_URL", "") or default_base).rstrip("/")
    if is_deepinfra_base(base_url):
        # DeepInfra's own credential only, never the Featherless one.
        api_key = os.environ.get("DEEPINFRA_API_KEY", "") or _env("TTS_API_KEY", "")
    else:
        api_key = _env("TTS_API_KEY", "") or _env("FEATHERLESS_API_KEY", "")
    return ModelEndpoint(base_url, api_key, _env("TTS_MODEL", default_model))


def _native_audio_host() -> str:
    """Which provider hosts the audio-input ("hearing") model: deepinfra | featherless | off.

    Capability, not credential, decides this one. Featherless serves no
    audio-input chat model as of 2026-11-08 (checked all 10,245 served ids: no
    XiaomiMiMo, no Qwen2-Audio, no MiniCPM-o, no whisper/asr), so native audio
    cannot follow the Featherless default without being switched to a text-only
    model that would silently stop hearing her. SAWII_NATIVE_AUDIO_HOST overrides:
    "off" disables the DualLoop lobe outright, "featherless" forces the default
    provider (use once Featherless serves an audio model, e.g. with
    SAWII_VOICE_NATIVE_MODEL naming it).
    """
    wanted = _env("NATIVE_AUDIO_HOST", "").strip().lower()
    if wanted in ("off", "none", "no", "false", "disabled"):
        return "off"
    if wanted in (PROVIDER_DEEPINFRA, PROVIDER_FEATHERLESS):
        return wanted
    if deepinfra_preferred():
        return PROVIDER_DEEPINFRA
    # CAPABILITY EXCEPTION, and the only place a bare credential still decides
    # anything. Featherless has no audio-input model, so the ears have nowhere to
    # go on the default provider; if a DeepInfra credential exists we keep them
    # where they work rather than silently deafening her. This host is pinned to
    # this one lobe - it cannot move the voice, the verifier, the observer or the
    # R worker, and its key can never ride along on a Featherless request.
    # Turn it off with SAWII_NATIVE_AUDIO_HOST=off.
    return PROVIDER_DEEPINFRA if os.environ.get("DEEPINFRA_API_KEY", "") else "off"


def _native_audio_available() -> bool:
    """True when a hosted audio-input model exists, so /app may default to DualLoop."""
    host = _native_audio_host()
    if host == "off":
        return False
    return bool(_env("VOICE_NATIVE_MODEL", "") or host == PROVIDER_DEEPINFRA)


def _voice_native_endpoint() -> ModelEndpoint:
    """The audio-input lobe - the one endpoint deliberately allowed off Featherless."""
    host = _native_audio_host()
    if host == "off":
        return _endpoint("VOICE_NATIVE", model="")
    base = DEEPINFRA_OPENAI if host == PROVIDER_DEEPINFRA else FEATHERLESS
    model = MIMO_HEAR if host == PROVIDER_DEEPINFRA else ""
    return _endpoint("VOICE_NATIVE", model=model, base=base, timeout=90)


def _asr_endpoint() -> ModelEndpoint:
    """Speech-to-text. Unconfigured unless an operator names a base: Featherless has no ASR model.

    The credential follows the base that was named, so pointing this at
    api.deepinfra.com takes DEEPINFRA_API_KEY and nothing else can leak in.
    """
    base_url = _env("ASR_BASE_URL", "").rstrip("/")
    if not base_url:
        return ModelEndpoint("", "", _env("ASR_MODEL", "whisper-1"))
    api_key = (os.environ.get("DEEPINFRA_API_KEY", "") if is_deepinfra_base(base_url)
               else _env("ASR_API_KEY", "") or _env("FEATHERLESS_API_KEY", ""))
    return ModelEndpoint(base_url, api_key, _env("ASR_MODEL", "whisper-1"))


def _a_endpoint() -> ModelEndpoint:
    """A speaks from Featherless unless the mouth host is selected.

    SAWII_A_HOST=coreweave is Qwen/Qwen3.8-27B on CoreWeave serverless.
    Both lobes call that model. AL and AR, once published, replace it.
    """
    from .life.mouth import BASE_MODEL, COREWEAVE

    if _env("A_HOST", "").strip().lower() == "tinker":
        return ModelEndpoint(
            base_url=_env("A_BASE_URL", TINKER_OAI).rstrip("/"),
            api_key=_env("A_API_KEY", "") or _env("TINKER_API_KEY", "") or os.environ.get("TINKER_API_KEY", ""),
            model=_env("A_MODEL", INKLING_SMALL),
            timeout=_float("A_TIMEOUT", 120),
            no_think=False,
        )
    if _env("A_HOST", "").strip().lower() == "coreweave":
        # The Featherless stand-in is not on this host. Naming the host
        # selects the mouth unless a different model was set on purpose.
        chosen = _env("A_MODEL", "")
        model = BASE_MODEL if chosen in ("", "Qwen/Qwen3.8-Flash-Next") else chosen
        return ModelEndpoint(
            base_url=_env("A_BASE_URL", COREWEAVE).rstrip("/"),
            api_key=_env("A_API_KEY", "") or _env("COREWEAVE_API_KEY", ""),
            model=model,
            timeout=_float("A_TIMEOUT", 90),
            no_think=_bool("DISABLE_THINKING", True),
        )
    return _endpoint("A", model="Qwen/Qwen3.8-Flash-Next", timeout=90)


@dataclass
class Settings:
    # --- storage -----------------------------------------------------------
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "./data"))
    db_path: str = ""
    # The diagnostic trace: one JSONL line for every model call, tool, memory
    # read and lane step, with the full prompt. Off by default because it
    # writes the whole conversation to disk; this is the switch for a diagnosis
    # run. SAWII_TRACE=1 turns it on, SAWII_TRACE_DIR moves it elsewhere.
    trace: bool = field(default_factory=lambda: _bool("TRACE", False))
    trace_dir: str = ""

    # --- security ----------------------------------------------------------
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL", "http://localhost:8800"))
    # Comma-separated allowed origins for the native shells (Capacitor/Tauri
    # load from capacitor://localhost, tauri://localhost, http://localhost) and
    # any separately hosted web build. "*" is safe here because auth is always a
    # bearer persona key - never a cookie. Empty string = no CORS middleware.
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))

    # --- the three lobes + helpers ----------------------------------------
    # A: the voice. Featherless Qwen3.8-Flash-Next until SAWII_A_HOST=coreweave,
    # which is Qwen/Qwen3.8-27B per token. AL learns everything and cools
    # toward that base. AR learns facts only and does not cool. README.md
    a: ModelEndpoint = field(default_factory=_a_endpoint)
    # B: the adjutant (private verifier, loops keeper, conscience).
    b: ModelEndpoint = field(default_factory=lambda: _endpoint("B", model="openai/gpt-oss-120b", timeout=90))
    # C: the psych observer.
    c: ModelEndpoint = field(default_factory=lambda: _endpoint("C", model="Compumacy/Psych_Qwen_32B", timeout=120))
    # Semantic grounding: explicit provider selection; all secrets stay in the environment.
    # Featherless Qwen/Qwen3-Embedding-8B is the default now that Featherless is the
    # provider (probed live: 200, 4096-dim, ~3.0s cold). "gemini" (Gemini Embedding 2,
    # 768-dim, needs GOOGLE_API_KEY/GEMINI_API_KEY) and "deepinfra" (768-dim) remain
    # one env var away; note the vector dimension differs per provider, so switching
    # re-embeds rather than reuses. An Actions secret is NOT a Render environment key.
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "featherless").lower())
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", ""))
    require_initial_semantic: bool = field(default_factory=lambda: _bool("REQUIRE_INITIAL_SEMANTIC", True))
    session_grounded_memory: bool = field(default_factory=lambda: _bool("SESSION_GROUNDED_MEMORY", True))
    # Picture-first turns: the turn is grounded on the picture of them, which is
    # already in the prompt for free, plus the loops and rules that need no
    # embedding. The broad semantic search then only runs when she actually
    # reaches for it (search_memory / try_to_remember) or on a turn that names
    # her past. Off means every turn pays the embedding before the first word.
    picture_first_memory: bool = field(default_factory=lambda: _bool("PICTURE_FIRST_MEMORY", True))
    # A brand-new conversation has no picture yet, so its first turn still
    # grounds semantically; after this many turns the picture is trusted.
    picture_first_after: int = field(default_factory=lambda: _int("PICTURE_FIRST_AFTER", 0))
    memory_session_idle_seconds: int = field(default_factory=lambda: _int("MEMORY_SESSION_IDLE_SECONDS", 1200))
    embed: ModelEndpoint = field(default_factory=lambda: _endpoint("EMBED", model="Qwen/Qwen3-Embedding-8B", timeout=20))
    # Consultants: comma separated model ids on the consult endpoint (default: B's provider).
    consult_models: list[str] = field(default_factory=lambda: [m.strip() for m in _env("CONSULT_MODELS", "").split(",") if m.strip()])
    consult_base_url: str = field(default_factory=lambda: _env("CONSULT_BASE_URL", ""))
    consult_api_key: str = field(default_factory=lambda: _env("CONSULT_API_KEY", ""))
    consult_wait_ms: int = field(default_factory=lambda: _int("CONSULT_WAIT_MS", 300))

    # Steer mode (2026-10-21): an expensive model that only directs. A (cheap)
    # does all speaking + tools; STEER writes short directives at critical
    # moments. Off unless SAWII_STEER_MODEL is set. Inherits A's provider when
    # SAWII_STEER_BASE_URL / _API_KEY are not given.
    steer: ModelEndpoint = field(default_factory=lambda: _endpoint("STEER", model="", timeout=60,
                                                                   inherit=_endpoint("A", model="")))
    steer_max_tokens: int = field(default_factory=lambda: _int("STEER_MAX_TOKENS", 90))
    # Inference (visits only) may use a different worker than the companion's A -
    # e.g. a cheap DeepSeek under an expensive director. This is the R-side
    # execution lane (the CURSOR host worker). It runs on Featherless by default
    # now (deepseek-ai/DeepSeek-V3.2 via the alias table, FEATHERLESS_API_KEY);
    # SAWII_INFER_A_BASE_URL naming DeepInfra, or SAWII_PROVIDER=deepinfra, moves
    # just this lane back with DEEPINFRA_API_KEY. SAWII_INFER_A_MODEL still
    # overrides the worker's own id. The alias table stays closed either way.
    infer_a: ModelEndpoint = field(default_factory=lambda: cursor_worker_endpoint(
        _endpoint("A", model="")))

    # --- behaviour ---------------------------------------------------------
    # The mind is the two lobes. One answers what was asked. The other, at
    # the same time, brings what that answer leaves out. SAWII_DUAL=0 exists
    # so the test suite can run the older speak() path. A product request
    # cannot turn the lobes off.
    dual: bool = field(default_factory=lambda: _bool("DUAL", True))
    # What a new command mid-reply does (dual only): "steer" = re-aim the task
    # without stopping (the silent lobe takes it at once), "stop" = end the reply
    # after the current sentence and take the words as the next question.
    # Computer use: send screenshots to the model as images (only if A can see).
    vision: bool = field(default_factory=lambda: _bool("VISION", False))
    # A20: after a reply the silent side pre-answers the likely next question (one extra call).
    anticipate: bool = field(default_factory=lambda: _bool("ANTICIPATE", True))
    interrupt_mode: str = field(default_factory=lambda: (_env("INTERRUPT_MODE", "steer") or "steer").lower())
    a_tools: bool = field(default_factory=lambda: _bool("A_TOOLS", True))
    # When nobody else is running her tool calls - at home, or any client that
    # sends no tools - she runs them herself and keeps the lanes moving. This
    # is how many rounds one reply may take before it hands back. A host that
    # runs the calls always gets them after one round; it is the host's loop.
    lane_rounds: int = field(default_factory=lambda: _int("LANE_ROUNDS", 6))
    # The anti-deception meter is on every reply (2026-10-22). This flag no
    # longer hides it. Kept so an old env line does not crash a boot.
    show_verdict_to_user: bool = field(default_factory=lambda: _bool("SHOW_VERDICT", True))
    # Publish the lobes' actual words - not just `from`/`kind` - on the live
    # channel, so the companion UI can show the inter-lobe dialogue in the open.
    # This deliberately reverses the old "they cannot see this" rule: the owner
    # wants the minds visible. Set SAWII_SHOW_INNER=false to go back to kinds-only.
    show_inner_to_user: bool = field(default_factory=lambda: _bool("SHOW_INNER", True))
    tail_turns: int = field(default_factory=lambda: _int("TAIL_TURNS", 0))  # 0 = full available history
    recall_budget_chars: int = field(default_factory=lambda: _int("RECALL_BUDGET_CHARS", 0))  # 0 = no app truncation
    recall_band: float = field(default_factory=lambda: _float("RECALL_BAND", 0.30))  # loose = associative
    # Open sesame: only a near match is named unasked. 0.78 is the semantic bar
    # (README.md). Ordinary recall stays at recall_band.
    open_sesame_threshold: float = field(default_factory=lambda: _float("OPEN_SESAME_THRESHOLD", 0.78))
    unaddressed_days: float = field(default_factory=lambda: _float("UNADDRESSED_DAYS", 3.0))
    live_checks_max: int = field(default_factory=lambda: _int("LIVE_CHECKS_MAX", 3))  # B checks per reply, live
    psyche_half_life_days: float = field(default_factory=lambda: _float("PSYCHE_HALF_LIFE_DAYS", 21.0))

    # --- initiative ---------------------------------------------------------
    heartbeat_seconds: int = field(default_factory=lambda: _int("HEARTBEAT_SECONDS", 60))
    reflect_every_ticks: int = field(default_factory=lambda: _int("REFLECT_EVERY_TICKS", 30))
    quiet_hours: str = field(default_factory=lambda: _env("QUIET_HOURS", "22:30-08:00"))
    daily_outreach_cap: int = field(default_factory=lambda: _int("DAILY_OUTREACH_CAP", 4))
    min_gap_between_outreach_minutes: int = field(default_factory=lambda: _int("OUTREACH_GAP_MIN", 90))
    background_enabled: bool = field(default_factory=lambda: _bool("BACKGROUND", True))
    # The persona service also ships a developer console at /app. In production
    # the user-facing app is the gateway (companion-app), and two UIs on two
    # domains is how people end up editing the wrong one. Set SERVE_CONSOLE=false
    # on the deployed backend so exactly one interface exists.
    serve_console: bool = field(default_factory=lambda: _bool("SERVE_CONSOLE", True))

    # --- sleep / subconscious ----------------------------------------------
    sleep_hour: int = field(default_factory=lambda: _int("SLEEP_HOUR", 3))
    juice_floor: float = field(default_factory=lambda: _float("JUICE_FLOOR", 0.55))
    buffer_rows: int = field(default_factory=lambda: _int("BUFFER_ROWS", 2000))
    bowl_ratio: float = field(default_factory=lambda: _float("BOWL_RATIO", 0.9))

    # --- the overflowing cup --------------------------------------------------
    # The raw-record budget per persona. Nothing evaporates until the cup is over
    # it, and only records that are already consolidated (see life/evaporate.py).
    cup_chars: int = field(default_factory=lambda: _int("CUP_CHARS", 400_000))
    cup_protect_days: float = field(default_factory=lambda: _float("CUP_PROTECT_DAYS", 14.0))
    replay_ratio: float = field(default_factory=lambda: _float("REPLAY_RATIO", 0.2))
    finetune_min_examples: int = field(default_factory=lambda: _int("FINETUNE_MIN_EXAMPLES", 40))
    finetune_provider: str = field(default_factory=lambda: _env("FINETUNE_PROVIDER", "none"))  # none|fireworks|command
    fireworks_api_key: str = field(default_factory=lambda: _env("FIREWORKS_API_KEY", ""))
    fireworks_account: str = field(default_factory=lambda: _env("FIREWORKS_ACCOUNT", ""))
    fireworks_base_model: str = field(default_factory=lambda: _env("FIREWORKS_BASE_MODEL", "accounts/fireworks/models/qwen3-8b"))
    finetune_command: str = field(default_factory=lambda: _env("FINETUNE_COMMAND", ""))

    # --- aging: the per-turn weight update ---------------------------------
    # Where A's weights actually live. When A runs on a hosted/frozen endpoint
    # this stays empty and the update module is an honest no-op. Point it at a
    # self-hosted server (vLLM + LoRA, or the transformers/PEFT wrapper) that
    # accepts POST {persona_id, turn_id, level, reward, prompt, response}.
    # The two copies she speaks from. AL is the fast lobe, AR the slow one.
    # Empty until the open-weight server has those names.
    al_model: str = field(default_factory=lambda: _env("AL_MODEL", ""))
    ar_model: str = field(default_factory=lambda: _env("AR_MODEL", ""))
    update_url: str = field(default_factory=lambda: _env("UPDATE_URL", ""))
    update_timeout: float = field(default_factory=lambda: _float("UPDATE_TIMEOUT", 20.0))
    update_max_chars: int = field(default_factory=lambda: _int("UPDATE_MAX_CHARS", 6000))

    tinker_effort: float = field(default_factory=lambda: _float("TINKER_EFFORT", 0.2))
    tinker_native_audio: bool = field(default_factory=lambda: _bool(
        "TINKER_NATIVE_AUDIO", _env("A_HOST", "").strip().lower() == "tinker"))
    # Inkling hears audio but writes text; the voice line still speaks it
    # through TTS (or the device's voice). "text" is an explicit opt-out.
    voice_output: str = field(default_factory=lambda: _env("VOICE_OUTPUT", "speech"))
    # "stream": each sentence is spoken as it is written (narration during
    # tool waits included) and C reviews the whole reply as it ends; a RED
    # with findings is corrected aloud. "whole": nothing is heard until C
    # has passed the whole reply GREEN. README.md
    voice_gate: str = field(default_factory=lambda: _env("VOICE_GATE", "stream"))
    # --- voice -------------------------------------------------------------
    # CAPABILITY EXCEPTION (verified 2026-11-08 against
    # GET https://api.featherless.ai/v1/models, 10,245 ids): Featherless serves no
    # audio-input chat model - no MiMo, no Qwen2-Audio, no MiniCPM-o, no
    # whisper/asr id - so "hearing" cannot move to Featherless without losing the
    # capability. Native audio therefore stays on the host that does serve it
    # (DeepInfra MiMo-V2.6-Flash) and is deliberately the ONE lobe allowed to
    # resolve to api.deepinfra.com. It is opt-in per lobe, never a side effect of
    # a stray credential: SAWII_NATIVE_AUDIO_HOST=off (or a DeepInfra key that is
    # absent) leaves it unconfigured and the companion falls back to ASR + text.
    # Opt out of the DualLoop default with SAWII_VOICE_NATIVE_AUDIO=0.
    native_audio_default: bool = field(default_factory=lambda: _bool(
        "VOICE_NATIVE_AUDIO", _native_audio_available()))
    voice_native: ModelEndpoint = field(default_factory=_voice_native_endpoint)
    # ASR likewise: no speech-to-text model on Featherless. Unconfigured unless an
    # operator names a base, and the key then follows that base's provider.
    asr: ModelEndpoint = field(default_factory=_asr_endpoint)
    # Featherless Chatterbox by default; DeepInfra MiMo/inworld under
    # SAWII_PROVIDER=deepinfra. SAWII_TTS_* still overrides per lobe.
    tts: ModelEndpoint = field(default_factory=_tts_endpoint)
    # Named Mia is MiMo-specific; Chatterbox voices are named (Aaron, Anaya, ...)
    # and the provider default applies when this is empty.
    tts_voice: str = field(default_factory=lambda: _env(
        "TTS_VOICE",
        "Mia" if "mimo" in _env("TTS_MODEL", "").lower() else ""))

    # --- channels ----------------------------------------------------------
    twilio_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID", ""))
    twilio_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN", ""))
    twilio_from: str = field(default_factory=lambda: _env("TWILIO_FROM", ""))
    twilio_whatsapp_from: str = field(default_factory=lambda: _env("TWILIO_WHATSAPP_FROM", ""))
    telegram_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_secret: str = field(default_factory=lambda: _env("TELEGRAM_WEBHOOK_SECRET", ""))
    smtp_host: str = field(default_factory=lambda: _env("SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: _int("SMTP_PORT", 587))
    smtp_user: str = field(default_factory=lambda: _env("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: _env("SMTP_PASSWORD", ""))
    smtp_from: str = field(default_factory=lambda: _env("SMTP_FROM", ""))
    email_inbound_secret: str = field(default_factory=lambda: _env("EMAIL_INBOUND_SECRET", ""))

    # --- deep lane ---------------------------------------------------------
    deep_max_lanes: int = field(default_factory=lambda: _int("DEEP_MAX_LANES", 0))
    workspace_root: str = ""

    def __post_init__(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        if not self.trace_dir:
            self.trace_dir = _env("TRACE_DIR", os.path.join(self.data_dir, "trace"))
        if self.trace:
            # The logger is a singleton the whole process writes to, so it is
            # opened here, once, where the data dir is known. It cannot raise.
            from . import trace as _trace
            _trace.setup(self.trace_dir, enabled=True)
        if not self.db_path:
            self.db_path = _env("DB_PATH", os.path.join(self.data_dir, "persona.db"))
        if not self.workspace_root:
            self.workspace_root = _env("WORKSPACE_ROOT", os.path.join(self.data_dir, "workspaces"))
        if not self.consult_base_url:
            self.consult_base_url = self.b.base_url
        if not self.consult_api_key:
            self.consult_api_key = self.b.api_key


load_dotenv()


def load_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    s.__post_init__()
    return s
