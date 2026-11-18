"""Run the Hands checks that exist in the repo (persona/testing), for real.

    python scripts/run_checks.py

Builds a Mind with scripted lobes (no network, no keys) on a throwaway store,
creates a persona, then runs the instrumented harness:

    run_test_turn   - one real turn through Mind.respond, with metrics
    run_benchmark   - N real turns, aggregate latency numbers

This is the same harness the product uses; the only substitution is the model
behind the lobes (ScriptedLLM) so the run is deterministic and offline.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Offline: no remote embedder, no semantic-required gate.
os.environ["SAWII_REQUIRE_INITIAL_SEMANTIC"] = "0"


async def main() -> int:
    from persona.config import load_settings
    from persona.identity import create_persona
    from persona.llm import ScriptedLLM
    from persona.mind import Mind
    from persona.testing import run_benchmark, run_test_turn

    tmp = Path(tempfile.mkdtemp(prefix="hands_checks_"))
    os.environ["SAWII_DATA_DIR"] = str(tmp)
    os.environ["SAWII_DB_PATH"] = str(tmp / "sawii.db")
    settings = load_settings()
    settings.data_dir = str(tmp)
    settings.db_path = str(tmp / "sawii.db")

    def lobe(text: str) -> ScriptedLLM:
        return ScriptedLLM(text, name="scripted")

    mind = Mind(
        settings,
        llms={
            "a": lobby_reply(),
            "b": lobe("Noted."),
            "c": lobe("Okay."),
            "embed": None,  # hash-only embeddings: fully offline
        },
    )

    pid, key = await create_persona(mind.store, name="Check")
    print(f"persona: id={pid} key={key[:10]}…")
    print(f"store:   {settings.db_path}\n")

    # ── one instrumented turn ─────────────────────────────────────────
    turn = await run_test_turn(mind, persona_id=pid, user_message="hey, how is your day going?")
    print("run_test_turn")
    print(f"  reply:     {turn.reply[:120]!r}")
    print(f"  wall_ms:   {turn.wall_ms}")
    print(f"  metrics:   {json.dumps(turn.metrics)}")
    print(f"  kernel:    {len(turn.kernel)} events")
    print(f"  error:     {turn.error or '-'}")
    if turn.error:
        print("\n(turn reported an error; benchmark still runs to show the path)")

    # ── a small benchmark ─────────────────────────────────────────────
    bench = await run_benchmark(mind, persona_id=pid, iterations=3)
    print("\nrun_benchmark (3 iterations)")
    print(f"  ttft_ms:        {bench['ttft_ms']}")
    print(f"  reply_ms:       {bench['reply_ms']}")
    print(f"  dead_air_max:   {bench['dead_air_max_ms']}")
    print(f"  wall_ms:        {bench['wall_ms']}")
    print(f"  errors:         {bench['errors'] or '-'}")
    for i, r in enumerate(bench["replies"]):
        print(f"  reply[{i}]:       {r[:100]!r}")

    ok = bool(turn.reply) and not turn.error and not bench["errors"] and all(bench["replies"])
    print("\nhands checks:", "PASSED" if ok else "COMPLETED WITH ERRORS")
    return 0 if ok else 1


def lobby_reply() -> "object":
    # The 'A' lobe answers; keep it short and warm so the reply path is exercised.
    from persona.llm import ChatResult, ScriptedLLM

    def responder(messages, opts):  # noqa: ANN001
        # The kernel may ask for JSON decisions; give plain text so it falls
        # back to a normal reply rather than a malformed instruction.
        return ChatResult(text="I'm good, thanks for asking. What's on your mind?")

    return ScriptedLLM(responder, name="scripted-a")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
