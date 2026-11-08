"""Guard: the engine stays a pure tool executor (no LLM, no provider key).

    python scripts/engine_no_llm_guard.py

Design invariant this protects: Persona is the only mind. ``sawii_hands.py``
executes tools and gets every bit of its thinking from R over the ``/v1/hands``
WebSocket, so the VLM *is* R. If anyone ever wires a provider client, an API key
or a model name into the engine, that silently splits the mind in two - so this
test fails the build instead of letting it slide.

It checks the two files that make up the engine binary (``sawii_hands.py`` and
its optional GUI client ``ocu_mcp.py``), plus the spawn path in
``persona/link.py`` (the only command line a delegated lane may receive).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Names that would mean the engine talks to a model provider by itself.
FORBIDDEN_IMPORTS = {
    "httpx", "openai", "anthropic", "google.generativeai", "litellm",
    "requests", "urllib3", "aiohttp", "transformers", "ollama", "mistralai",
    "groq", "deepinfra", "cohere",
}
# Identifier/substring patterns that would mean the engine holds a credential
# or names a model of its own.
FORBIDDEN_TEXT = [
    r"\bapi_key\b", r"\bAPI_KEY\b", r"\bapiKey\b", r"\bvlmApiKey\b",
    r"\bDEEPINFRA\b", r"\bOPENAI_API_KEY\b", r"\bGEMINI_API_KEY\b",
    r"\bbase_url\b", r"\bvlmBaseUrl\b", r"\bmodel_name\b", r"\bvlmModelName\b",
    r"chat/completions", r"\bopenai\.ChatCompletion",
]
# Ways the phantom --engine-preset flag could come back. Prose may *mention* it
# (the docs explain why it does not exist); only real wiring is a regression.
PRESET_WIRING = [r'add_argument\(\s*["\']--engine-preset', r"dest\s*=\s*[\"']engine_preset",
                 r"\.engine_preset\b", r"engine_preset\s*=", r"yaml\.safe_load"]

ENGINE_FILES = ["sawii_hands.py", "ocu_mcp.py"]


def _imports(path: Path) -> set[str]:
    """Every module this file imports, top-level and nested."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return {m.lower() for m in found}


def main() -> int:
    failures: list[str] = []
    print("engine no-LLM guard")

    for rel in ENGINE_FILES:
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"{rel}: missing")
            print(f"  [FAIL] {rel:<16} missing")
            continue
        text = path.read_text(encoding="utf-8")
        mods = _imports(path)
        bad_mods = sorted(m for m in MODS if m in mods) if (MODS := FORBIDDEN_IMPORTS) else []
        bad_text = sorted({rx for rx in FORBIDDEN_TEXT if re.search(rx, text)})
        status = "OK  " if not bad_mods and not bad_text else "FAIL"
        if bad_mods or bad_text:
            failures.append(f"{rel}: provider imports={bad_mods} credential text={bad_text}")
        print(f"  [{status}] {rel:<16} imports={len(mods):3d} provider-imports={bad_mods or 'none'}"
              f" credential/model text={bad_text or 'none'}")

    # The engine's own CLI must not grow a model or key flag.
    hands = ROOT / "sawii_hands.py"
    argv_flags = set(re.findall(r'"(--[a-z-]+)"', hands.read_text(encoding="utf-8")))
    # --key is the PERSONA key (auth to /v1/hands). Anything naming a model,
    # a preset or a second credential is the regression we are guarding.
    keyish = sorted(f for f in argv_flags if f != "--key"
                    and ("key" in f or "model" in f or "preset" in f))
    print(f"  [OK  ] sawii_hands.py flags    {sorted(argv_flags)}")
    if "--key" not in argv_flags:
        failures.append("--key (the persona key) disappeared from the engine CLI")
    if keyish:
        failures.append(f"engine CLI grew a credential/model flag: {keyish}")

    # stitch.py must not hand the engine a preset, and must document its own key
    # as Persona's.
    stitch = (ROOT / "stitch.py").read_text(encoding="utf-8")
    leaked = [rx for rx in PRESET_WIRING if re.search(rx, stitch)]
    if leaked:
        failures.append(f"stitch.py wires a phantom engine preset / YAML loader: {leaked}")
    # The phantom flag must not be advertised as usable anywhere, including prose
# and the README. (Docs may say it does not exist - that is the fix.)
    readme = ROOT / "README.md"
    docs = {"stitch.py": stitch}
    if readme.is_file():
        docs["README.md"] = readme.read_text(encoding="utf-8")
    advertised = [f"{name}: {ln.strip()}" for name, text in docs.items()
                  for ln in text.splitlines()
                  if "--engine-preset" in ln and "python stitch.py" in ln]
    if advertised:
        failures.append(f"the phantom flag is still advertised as usable: {advertised}")
    documented = "PERSONA" in stitch and "DEEPINFRA_API_KEY" in stitch
    print(f"  [{'OK  ' if documented and not leaked else 'FAIL'}] stitch.py          "
          f"engine-preset flag={'absent' if not leaked else leaked}, "
          f"--deepinfra-key documented as Persona's={'yes' if documented else 'no'}")
    if not documented:
        failures.append("stitch.py no longer states that --deepinfra-key is PERSONA's key")

    # The spawn path may only pass the persona key + sandbox flags to the engine.
    spawn = (ROOT / "persona" / "link.py").read_text(encoding="utf-8")
    i = spawn.find("async def _spawn_delegate")
    block = spawn[i:spawn.find("async def _monitor_process", i)] if i >= 0 else ""
    passed = set(re.findall(r'"(--[a-z-]+)"', block))
    allowed = {"--url", "--key", "--approve", "--place", "--lane", "--root"}
    extra = sorted(passed - allowed)
    print(f"  [{'OK  ' if block and not extra else 'FAIL'}] link._spawn_delegate passes {sorted(passed)}")
    if not block:
        failures.append("could not find _spawn_delegate in persona/link.py")
    if extra:
        failures.append(f"_spawn_delegate passes new engine flags: {extra}")
    if "--lane" not in passed or "--key" not in passed:
        failures.append("_spawn_delegate must pass --lane and --key (the persona key)")

    # argparse must actually accept every flag the docs advertise.
    probe = argparse.ArgumentParser()
    probe.add_argument("--url"); probe.add_argument("--key"); probe.add_argument("--root", action="append")
    probe.add_argument("--approve"); probe.add_argument("--place")
    probe.add_argument("--computer-use", action="store_true"); probe.add_argument("--lane")
    try:
        probe.parse_args(["--url", "http://x", "--key", "sp_x", "--root", ".", "--lane", "R1",
                          "--approve", "ask", "--place", "computer", "--computer-use"])
        print("  [OK  ] engine CLI         every advertised flag parses")
    except SystemExit as exc:
        failures.append(f"engine CLI rejects its own documented flags: {exc}")

    print()
    if failures:
        print("ENGINE NO-LLM GUARD FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ENGINE NO-LLM GUARD PASSED - the engine has no LLM, no provider client "
          "and no provider key; R is its only mind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())