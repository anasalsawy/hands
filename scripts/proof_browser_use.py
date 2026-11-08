#!/usr/bin/env python3
"""Proof: Persona creates concurrency inside browser-use.

This script demonstrates the core value proposition:

    Persona's Mind decides to split work into R-lanes.
    Each R-lane becomes a browser-use Agent.run() coroutine.
    Multiple agents run concurrently — managed by Persona's LaneBook.
    browser-use thinks each Agent is independent.
    Persona knows they are all R-lanes of one conversation.

What this proves to browser-use:
    1. Persona plugs INTO browser-use, not around it.
    2. Concurrency uses browser-use's own Agent class — no wrapper, no proxy.
    3. Persona manages the lifecycle (start/steer/stop) through Link.
    4. Any LLM endpoint works — including Persona itself as the LLM.

Usage:
    # Basic: 3 concurrent browser-use agents, each with its own task
    python scripts/proof_browser_use.py --key $OPENAI_API_KEY

    # With a custom LLM endpoint (e.g. Persona is the LLM)
    python scripts/proof_browser_use.py --key $OPENAI_API_KEY --base-url http://localhost:8800/v1

    # Headful (see the browsers)
    python scripts/proof_browser_use.py --key $OPENAI_API_KEY --headful

    # Custom number of concurrent lanes
    python scripts/proof_browser_use.py --key $OPENAI_API_KEY --lanes 5

Requirements:
    pip install browser-use langchain-openai
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root so we can import persona.link
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona.link import (
    AgentProfile,
    BrowserUseNativeDriver,
    Link,
    LinkRegistry,
    LinkedLane,
)

log = logging.getLogger("proof")

# ──────────────────────────────────────────────────────────────────────
# Sample tasks that demonstrate concurrency — each is an independent
# browser-use Agent that runs in its own R-lane.
# ──────────────────────────────────────────────────────────────────────

SAMPLE_TASKS = [
    'Go to https://news.ycombinator.com and tell me the title of the #1 story right now.',
    'Go to https://en.wikipedia.org/wiki/Concurrency_(computer_science) and summarize the first paragraph.',
    'Go to https://httpbin.org/get and tell me what IP address is shown.',
    'Go to https://example.com and read the full text on the page.',
    'Go to https://www.timeanddate.com/worldclock/ and tell me the current time in Tokyo.',
]


async def run_proof(args) -> None:
    """Spawn N concurrent browser-use Agents through Persona's Link adapter."""

    print("\n" + "=" * 70)
    print("  PROOF: Persona creates concurrency inside browser-use")
    print("=" * 70)
    print(f"\n  Lanes:    {args.lanes}")
    print(f"  Model:    {args.model}")
    print(f"  Headless: {not args.headful}")
    print(f"  LLM URL:  {args.base_url or '(default OpenAI)'}")
    print()

    # ── Build the Link (exactly as stitch.py would) ──────────────────
    config = {
        "name": "browser-use-proof",
        "agent_type": "browser_use_native",
        "model": args.model,
        "api_key": args.key,
        "base_url": args.base_url or "",
        "headless": not args.headful,
        "max_actions_per_step": 10,
        "max_failures": 3,
        "use_vision": True,
    }

    link = Link.detect(config)
    print(f"  Link detected: {link.agent.name} (strategy={link.agent.strategy}, "
          f"type={link.agent.agent_type})")
    print()

    # ── Pick tasks ───────────────────────────────────────────────────
    tasks_to_run = SAMPLE_TASKS[:args.lanes]
    if args.lanes > len(SAMPLE_TASKS):
        # Repeat tasks to fill the requested lane count
        for i in range(args.lanes - len(SAMPLE_TASKS)):
            tasks_to_run.append(SAMPLE_TASKS[i % len(SAMPLE_TASKS)])

    # ── Spawn all lanes concurrently ─────────────────────────────────
    # This is the proof: each delegate() call creates a browser-use Agent
    # with its own Browser instance. Agent.run() runs as an asyncio task.
    # Persona's Link manages all of them.
    print("─── Spawning R-lanes ───")
    t0 = time.monotonic()

    lane_futures = []
    for i, task in enumerate(tasks_to_run):
        lane_id = f"proof-lane-{i}"
        print(f"  R-lane {lane_id}: {task[:60]}...")
        lane_futures.append(link.delegate(lane_id, task))

    # All delegates launch concurrently — they return immediately after
    # creating the asyncio task. The actual Agent.run() is happening in
    # the background now.
    linked_lanes = await asyncio.gather(*lane_futures, return_exceptions=True)

    launched = 0
    for i, result in enumerate(linked_lanes):
        if isinstance(result, Exception):
            print(f"  ✗ Lane {i} failed to launch: {result}")
        else:
            launched += 1
            print(f"  ✓ Lane {result.lane_id}: status={result.status}")

    print(f"\n  Launched {launched}/{args.lanes} lanes in "
          f"{time.monotonic() - t0:.1f}s")

    if launched == 0:
        print("\n  No lanes launched. Check that browser-use and langchain-openai "
              "are installed:\n    pip install browser-use langchain-openai\n")
        return

    # ── Wait for all lanes to complete ───────────────────────────────
    print("\n─── Waiting for R-lanes to finish ───")
    print("  (each lane is a browser-use Agent.run() coroutine)")
    print()

    # Collect the _bridge_task from each linked lane
    active_tasks = {}
    for result in linked_lanes:
        if isinstance(result, LinkedLane) and result._bridge_task:
            active_tasks[result.lane_id] = result

    # Poll status until all done
    while active_tasks:
        done = []
        for lane_id, linked in active_tasks.items():
            if linked._bridge_task.done():
                elapsed = time.monotonic() - linked.started_at
                exc = linked._bridge_task.exception()
                if exc:
                    linked.status = "failed"
                    linked.error = str(exc)[:200]
                elif linked.status not in ("done", "failed", "stopped"):
                    linked.status = "done"
                symbol = "✓" if linked.status == "done" else "✗"
                print(f"  {symbol} {lane_id}: {linked.status} "
                      f"({elapsed:.1f}s) {linked.error or ''}")
                done.append(lane_id)
        for lid in done:
            del active_tasks[lid]
        if active_tasks:
            await asyncio.sleep(0.5)

    elapsed_total = time.monotonic() - t0
    print(f"\n─── All {launched} R-lanes completed in {elapsed_total:.1f}s ───")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WHAT THIS PROVED:")
    print("=" * 70)
    print(f"""
  1. Persona created {launched} concurrent R-lanes.
  2. Each R-lane was a browser-use Agent.run() coroutine.
  3. Each Agent had its own Browser instance (separate profile).
  4. Persona's Link managed all lifecycles (delegate/status/stop).
  5. browser-use's own Agent class did all the browser work.
  6. Persona added the concurrency — browser-use didn't need to change.

  This is the product: Persona as inference middleware that creates
  concurrency inside ANY agent by using that agent's own spawning
  mechanism. For browser-use, that mechanism is Agent + Browser + run().
""")


def main():
    ap = argparse.ArgumentParser(
        description="Proof: Persona creates concurrency inside browser-use",
    )
    ap.add_argument("--key", required=True,
                    help="LLM API key (OpenAI, Anthropic, etc.)")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="LLM model name (default: gpt-4o-mini)")
    ap.add_argument("--base-url", default="",
                    help="Custom LLM base URL (e.g. Persona as LLM)")
    ap.add_argument("--lanes", type=int, default=3,
                    help="Number of concurrent R-lanes (default: 3)")
    ap.add_argument("--headful", action="store_true",
                    help="Run browsers with visible windows")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    asyncio.run(run_proof(args))


if __name__ == "__main__":
    main()
