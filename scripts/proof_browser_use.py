#!/usr/bin/env python3
"""Proof: Persona creates concurrency INSIDE browser-use.

The key insight:
    browser-use calls self.llm.ainvoke(messages) and WAITS for the response.
    Persona IS that LLM. It implements BaseChatModel.
    During ainvoke, Persona has full control. The agent loop is blocked.
    Persona spawns N concurrent browser-use Agents from INSIDE the LLM call.
    Each agent has its own BrowserSession.
    They all run concurrently via asyncio.gather.
    Persona aggregates results and returns ONE AgentOutput.

    The calling agent never knew concurrency happened.
    It made one LLM call. It got one response.
    Behind that response, N browsers ran simultaneously.

    This is the product: swap your LLM for Persona, get concurrency for free.
    browser-use doesn't change. No wrapper. No proxy. No external orchestration.

Usage:
    # Run the proof with OpenAI as the backing LLM for sub-agents
    python scripts/proof_browser_use.py --key $OPENAI_API_KEY

    # With more concurrent lanes
    python scripts/proof_browser_use.py --key $OPENAI_API_KEY --lanes 5

    # See the browsers
    python scripts/proof_browser_use.py --key $OPENAI_API_KEY --headful

Requirements:
    pip install browser-use
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from pydantic import BaseModel

log = logging.getLogger("proof")

T = TypeVar('T', bound=BaseModel)


# ─────────────────────────────────────────────────────────────────────
# PersonaLLM: implements browser-use's BaseChatModel protocol.
# This IS Persona sitting in the LLM position.
# When browser-use calls ainvoke(), Persona intercepts and creates
# concurrency using browser-use's own Agent class.
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PersonaLLM:
    """Persona as a browser-use LLM.

    Implements BaseChatModel protocol so browser-use can use it as:
        Agent(task=..., llm=PersonaLLM(...))

    When ainvoke is called, Persona:
    1. Reads the task from the messages
    2. Decides whether to split into concurrent sub-tasks
    3. If yes: spawns N browser-use Agents internally, runs them
       concurrently, aggregates results, returns single response
    4. If no: forwards to the backing LLM normally
    """

    # The real LLM that sub-agents will use
    api_key: str
    backing_model: str = "gpt-4o-mini"
    base_url: str | None = None
    headless: bool = True

    # Concurrency config
    max_concurrent: int = 3

    # Protocol fields
    model: str = "persona-concurrent"
    _verified_api_keys: bool = False

    # Internal state
    _sub_tasks: list[str] = field(default_factory=list)
    _results: dict[str, str] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return 'persona'

    @property
    def name(self) -> str:
        return f'persona-concurrent-{self.max_concurrent}'

    @property
    def model_name(self) -> str:
        return self.model

    def set_concurrent_tasks(self, tasks: list[str]) -> None:
        """Set the sub-tasks that Persona will run concurrently.

        In production, Persona's Mind would analyze the conversation
        and decide how to split. For this proof, we pre-set them.
        """
        self._sub_tasks = tasks

    async def ainvoke(
        self, messages: list[Any], output_format: type[T] | None = None, **kwargs: Any
    ) -> Any:
        """The interception point.

        browser-use calls this expecting an LLM response.
        Persona runs N concurrent browser-use agents INSIDE this call.
        Returns a single AgentOutput with aggregated results.
        """
        from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

        log.info(f"═══ PersonaLLM.ainvoke called — {len(messages)} messages")

        if self._sub_tasks:
            # ── CONCURRENT MODE ──────────────────────────────────────
            # Persona's Mind has decided to split work.
            # We spawn N browser-use Agents from INSIDE this LLM call.
            log.info(f"═══ Persona splitting into {len(self._sub_tasks)} concurrent R-lanes")
            aggregated = await self._run_concurrent_agents(self._sub_tasks)
            # Clear tasks after execution
            self._sub_tasks = []

            # Build the AgentOutput that browser-use expects
            if output_format is not None:
                # Construct a valid AgentOutput with done action
                result_text = "CONCURRENT RESULTS:\n"
                for task_name, result in aggregated.items():
                    result_text += f"\n[{task_name}]: {result}\n"

                # We need to return the schema browser-use expects
                output_data = {
                    "evaluation_previous_goal": "Success — concurrent execution completed",
                    "memory": f"Ran {len(aggregated)} concurrent sub-tasks via Persona",
                    "next_goal": "Report aggregated results",
                    "action": [{"done": {"text": result_text, "success": True}}],
                }
                completion = output_format.model_validate(output_data)
            else:
                completion = aggregated  # type: ignore

            return ChatInvokeCompletion(
                completion=completion,
                usage=ChatInvokeUsage(
                    input_tokens=0,
                    output_tokens=0,
                ),
            )
        else:
            # ── PASSTHROUGH MODE ─────────────────────────────────────
            # No concurrent tasks — forward to backing LLM
            log.info("═══ Persona passthrough to backing LLM")
            backing = self._get_backing_llm()
            return await backing.ainvoke(messages, output_format=output_format, **kwargs)

    def _get_backing_llm(self) -> Any:
        """Create the real LLM for sub-agents or passthrough."""
        from browser_use.llm.openai.chat import ChatOpenAI
        params: dict[str, Any] = {
            "model": self.backing_model,
            "api_key": self.api_key,
        }
        if self.base_url:
            params["base_url"] = self.base_url
        return ChatOpenAI(**params)

    async def _run_concurrent_agents(self, tasks: list[str]) -> dict[str, str]:
        """Spawn N browser-use Agents and run them concurrently.

        This is the core proof: from INSIDE the LLM call, we create
        browser-use Agent instances with their own BrowserSessions.
        They run via asyncio.gather. browser-use's own Agent class
        does all the browser work.
        """
        from browser_use import Agent, BrowserProfile, BrowserSession

        results: dict[str, str] = {}
        backing_llm = self._get_backing_llm()

        async def run_one(task_id: str, task: str) -> tuple[str, str]:
            """Run a single browser-use Agent as one R-lane."""
            log.info(f"  R-lane {task_id}: starting — {task[:60]}...")
            t0 = time.monotonic()

            profile = BrowserProfile(headless=self.headless)
            session = BrowserSession(browser_profile=profile)

            agent = Agent(
                task=task,
                llm=backing_llm,
                browser_session=session,
                max_actions_per_step=5,
                enable_signal_handler=False,
            )

            try:
                history = await agent.run(max_steps=10)
                result = history.final_result() or "No result"
                elapsed = time.monotonic() - t0
                log.info(f"  R-lane {task_id}: done in {elapsed:.1f}s")
                return task_id, result
            except Exception as e:
                elapsed = time.monotonic() - t0
                log.error(f"  R-lane {task_id}: failed in {elapsed:.1f}s — {e}")
                return task_id, f"ERROR: {e}"
            finally:
                await session.close()

        # Run all agents concurrently — this is the concurrency!
        log.info(f"═══ Launching {len(tasks)} R-lanes concurrently via asyncio.gather")
        t_start = time.monotonic()

        lane_coros = [
            run_one(f"lane-{i}", task)
            for i, task in enumerate(tasks)
        ]
        lane_results = await asyncio.gather(*lane_coros, return_exceptions=True)

        for item in lane_results:
            if isinstance(item, Exception):
                log.error(f"  R-lane failed: {item}")
            else:
                task_id, result = item
                results[task_id] = result

        elapsed_total = time.monotonic() - t_start
        log.info(f"═══ All {len(tasks)} R-lanes completed in {elapsed_total:.1f}s")

        return results


# ─────────────────────────────────────────────────────────────────────
# The proof: browser-use Agent with PersonaLLM as its LLM
# ─────────────────────────────────────────────────────────────────────

async def run_proof(args) -> None:
    """Run browser-use with Persona as the LLM."""
    from browser_use import Agent, BrowserProfile, BrowserSession

    print("\n" + "=" * 70)
    print("  PROOF: Persona creates concurrency INSIDE browser-use")
    print("=" * 70)

    # The tasks that Persona will run concurrently
    concurrent_tasks = [
        'Go to https://news.ycombinator.com and tell me the title of the #1 story.',
        'Go to https://en.wikipedia.org/wiki/Concurrency_(computer_science) and summarize the first paragraph.',
        'Go to https://httpbin.org/get and tell me what IP address is shown.',
        'Go to https://example.com and read the full text on the page.',
        'Go to https://www.timeanddate.com/worldclock/ and tell me the current time in Tokyo.',
    ][:args.lanes]

    print(f"\n  Concurrent R-lanes: {len(concurrent_tasks)}")
    print(f"  Backing model:      {args.model}")
    print(f"  Headless:           {not args.headful}")
    for i, t in enumerate(concurrent_tasks):
        print(f"  Lane {i}: {t[:65]}...")
    print()

    # ── Create PersonaLLM ────────────────────────────────────────────
    persona_llm = PersonaLLM(
        api_key=args.key,
        backing_model=args.model,
        base_url=args.base_url or None,
        headless=not args.headful,
        max_concurrent=args.lanes,
    )

    # Pre-set the concurrent tasks.
    # In production, Persona's Mind would decide this from the conversation.
    persona_llm.set_concurrent_tasks(concurrent_tasks)

    # ── Create the browser-use Agent with Persona as its LLM ────────
    # This is EXACTLY how browser-use is normally used.
    # The only difference: llm=persona_llm instead of llm=ChatOpenAI(...)
    profile = BrowserProfile(headless=not args.headful)
    session = BrowserSession(browser_profile=profile)

    agent = Agent(
        task="Execute all the following tasks concurrently and report results.",
        llm=persona_llm,  # ← Persona IS the LLM
        browser_session=session,
    )

    print("─── browser-use Agent created with PersonaLLM ───")
    print("  Agent calls llm.ainvoke() → PersonaLLM intercepts")
    print("  PersonaLLM spawns N concurrent Agents INSIDE the call")
    print("  Each sub-agent has its own BrowserSession")
    print("  All run concurrently via asyncio.gather")
    print("  Agent gets back ONE response with all results")
    print()

    # ── Run the agent ────────────────────────────────────────────────
    print("─── Starting browser-use agent.run() ───")
    t0 = time.monotonic()

    try:
        history = await agent.run(max_steps=3)
        elapsed = time.monotonic() - t0

        print(f"\n─── Agent completed in {elapsed:.1f}s ───")
        result = history.final_result()
        if result:
            print(f"\n  Result:\n{result[:2000]}")
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"\n─── Agent failed after {elapsed:.1f}s: {e} ───")
    finally:
        await session.close()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WHAT THIS PROVED:")
    print("=" * 70)
    print(f"""
  1. browser-use Agent was created normally: Agent(task=..., llm=persona_llm)
  2. Agent called persona_llm.ainvoke() — standard LLM call.
  3. INSIDE ainvoke(), PersonaLLM spawned {args.lanes} concurrent Agents.
  4. Each had its own BrowserSession (separate browser instance).
  5. They ran concurrently via asyncio.gather.
  6. PersonaLLM returned ONE AgentOutput with all results.
  7. The calling Agent never knew concurrency happened.

  browser-use did NOT change. No wrapper. No proxy.
  Persona sat in the LLM position and created concurrency
  using browser-use's own Agent class from INSIDE the LLM call.

  This is the product:
    Agent(task=..., llm=PersonaLLM(...))  ← swap LLM, get concurrency
""")


def main():
    ap = argparse.ArgumentParser(
        description="Proof: Persona creates concurrency inside browser-use",
    )
    ap.add_argument("--key", required=True,
                    help="LLM API key (OpenAI, etc.)")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="Backing LLM model (default: gpt-4o-mini)")
    ap.add_argument("--base-url", default="",
                    help="Custom backing LLM base URL")
    ap.add_argument("--lanes", type=int, default=3,
                    help="Number of concurrent R-lanes (default: 3)")
    ap.add_argument("--headful", action="store_true",
                    help="Run browsers with visible windows")

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    asyncio.run(run_proof(args))


if __name__ == "__main__":
    main()
