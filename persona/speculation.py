"""Persona's speculation: ThoughtStream, SpeculativeRouter, PossibilitySpeculator and
SpeculativeEngine - the part of Persona's mind that prepares next actions and fires them
when the real receipt matches. Moved verbatim out of the retired ``link.py``; no logic changed.
Execution belongs to Agent TARS (see tars_loop.py / tars_engine.py)."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ThoughtStream — live inference mirror for simultaneous shared consciousness
# ═══════════════════════════════════════════════════════════════════════════


class ThoughtStream:
    """Live inference mirror — streams one agent's reasoning to all others.

    Standard multi-agent frameworks are sequential talkers: Agent A runs,
    finishes, sends a message, then Agent B runs. ThoughtStream creates
    simultaneous shared consciousness: Agent 1 knows what Agent 0 is
    *intending* to do while Agent 0 is still thinking it.

    Mechanics:
        Because Persona IS the LLM (sits in the ainvoke() slot), we control
        the token stream server-side. As Agent 0 generates <thought> tokens
        token-by-token, ThoughtStream captures and broadcasts them to all
        other agents' context windows in real time.

    Two consumption modes:
        1. Streaming Mirror — tokens arrive as they're generated, other
           agents see partial thoughts mid-inference
        2. Snapshot — agent reads the latest complete thought block before
           starting its own inference step

    This is the "Inter-Agent Attention Layer": before Agent 1 begins its
    own inference, its input is dynamically appended with:
        [Agent_0_Current_Thought: "Typing login credentials..."]

    Thread-safe via asyncio. No locking needed for reads — worst case
    a reader sees a slightly stale snapshot, which is fine because the
    stream is append-only.
    """

    def __init__(self, max_tokens_per_agent: int = 2048):
        self._streams: dict[str, list[str]] = {}
        # agent_id -> list of token chunks (append-only)
        self._thought_blocks: dict[str, list[str]] = {}
        # agent_id -> list of complete thought blocks
        self._active_thought: dict[str, str] = {}
        # agent_id -> current partial thought being generated
        self._listeners: dict[str, list[asyncio.Event]] = {}
        # agent_id -> list of events to notify on new tokens
        self._max_tokens = max_tokens_per_agent
        self._lock = asyncio.Lock()
        self._t0 = time.monotonic()

    # ── Writing: agent pushes its reasoning tokens ─────────────────

    async def push_tokens(self, agent_id: str, tokens: str) -> None:
        """Agent pushes token chunk as it's generated.

        Called from inside ainvoke() as the model streams tokens.
        Other agents' listeners are notified immediately.
        """
        async with self._lock:
            if agent_id not in self._streams:
                self._streams[agent_id] = []
                self._active_thought[agent_id] = ""
            self._streams[agent_id].append(tokens)
            self._active_thought[agent_id] += tokens

            # Trim if too long (keep tail)
            if len(self._active_thought[agent_id]) > self._max_tokens:
                self._active_thought[agent_id] = \
                    self._active_thought[agent_id][-self._max_tokens:]

        # Notify all listeners for this agent (outside lock)
        for evt in self._listeners.get(agent_id, []):
            evt.set()

    async def complete_thought(self, agent_id: str,
                                thought_type: str = "reasoning") -> None:
        """Mark current thought block as complete.

        Called when the agent finishes a reasoning step (before action).
        The complete block is archived and the active buffer is cleared.
        """
        async with self._lock:
            active = self._active_thought.get(agent_id, "")
            if active.strip():
                if agent_id not in self._thought_blocks:
                    self._thought_blocks[agent_id] = []
                self._thought_blocks[agent_id].append(active.strip())
                # Keep last 5 blocks
                if len(self._thought_blocks[agent_id]) > 5:
                    self._thought_blocks[agent_id] = \
                        self._thought_blocks[agent_id][-5:]
            self._active_thought[agent_id] = ""
            self._streams[agent_id] = []

    async def push_action_intent(self, agent_id: str,
                                  action: str, target: str,
                                  reasoning: str = "") -> None:
        """Shortcut: agent declares what it's about to do and why.

        More structured than raw token streaming — used when the agent
        has already decided on an action but hasn't executed it yet.
        Other agents see this as: "[agent_0 INTENDS: click '#submit'
        because form is filled]"
        """
        intent = f"{action} '{target}'"
        if reasoning:
            intent += f" — {reasoning}"
        async with self._lock:
            if agent_id not in self._streams:
                self._streams[agent_id] = []
            self._active_thought[agent_id] = f"INTENT: {intent}"
            self._streams[agent_id].append(f"INTENT: {intent}")
        # Notify listeners
        for evt in self._listeners.get(agent_id, []):
            evt.set()

    # ── Reading: other agents consume the stream ───────────────────

    async def get_live_thoughts(self, exclude_agent: str | None = None,
                                 max_chars: int = 1000) -> str:
        """Snapshot of all agents' current thoughts for injection into prompt.

        This is the Inter-Agent Attention Layer. Returns a formatted block
        ready to inject into another agent's system prompt or context:

            LIVE AGENT THOUGHTS:
            [agent-0] THINKING: "I see a login form, will type credentials..."
            [agent-1] INTENT: click '#submit' — form is complete
            [agent-2] IDLE (last: "Extracted price from #total")

        Called by an agent before its own inference step.
        """
        async with self._lock:
            lines = []
            for aid, active in self._active_thought.items():
                if aid == exclude_agent:
                    continue
                if active.strip():
                    # Truncate to max_chars per agent
                    thought = active.strip()[:max_chars]
                    if thought.startswith("INTENT:"):
                        lines.append(f"  [{aid}] {thought}")
                    else:
                        lines.append(f"  [{aid}] THINKING: \"{thought}\"")
                else:
                    # Show last completed thought
                    blocks = self._thought_blocks.get(aid, [])
                    if blocks:
                        last = blocks[-1][:200]
                        lines.append(f"  [{aid}] IDLE (last: \"{last}\")")
                    else:
                        lines.append(f"  [{aid}] IDLE")

            if not lines:
                return ""
            return "LIVE AGENT THOUGHTS:\n" + "\n".join(lines)

    async def get_agent_intent(self, agent_id: str) -> str | None:
        """Get a specific agent's current intent/thought.

        Used for targeted awareness — e.g., hand-1 checks what hand-0
        is about to do before choosing its own action.
        """
        async with self._lock:
            return self._active_thought.get(agent_id, "").strip() or None

    async def wait_for_thought(self, agent_id: str,
                                timeout: float = 5.0) -> str | None:
        """Block until an agent pushes new tokens. Returns the new content.

        Used for the Streaming Mirror mode — hand-1 literally waits for
        hand-0 to start thinking, then reacts.
        """
        evt = asyncio.Event()
        if agent_id not in self._listeners:
            self._listeners[agent_id] = []
        self._listeners[agent_id].append(evt)

        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
            return self._active_thought.get(agent_id, "")
        except asyncio.TimeoutError:
            return None
        finally:
            self._listeners.get(agent_id, []).remove(evt) if evt in \
                self._listeners.get(agent_id, []) else None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def clear_agent(self, agent_id: str) -> None:
        """Remove an agent's stream (agent done or crashed)."""
        async with self._lock:
            self._streams.pop(agent_id, None)
            self._thought_blocks.pop(agent_id, None)
            self._active_thought.pop(agent_id, None)
            self._listeners.pop(agent_id, None)

    async def get_summary(self) -> str:
        """Full summary of all thought activity."""
        async with self._lock:
            lines = []
            elapsed = time.monotonic() - self._t0
            lines.append(f"ThoughtStream active {elapsed:.1f}s, "
                         f"{len(self._streams)} agents")
            for aid in self._streams:
                active = self._active_thought.get(aid, "")
                blocks = len(self._thought_blocks.get(aid, []))
                total_tokens = sum(len(c) for c in self._streams.get(aid, []))
                status = "THINKING" if active.strip() else "IDLE"
                lines.append(f"  [{aid}] {status} — "
                             f"{blocks} completed thoughts, "
                             f"{total_tokens} chars streamed")
            return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SpeculativeRouter — probability-based speculative batching
# ═══════════════════════════════════════════════════════════════════════════


class SpeculativeBranch:
    """One probability branch — an agent pursuing one hypothesis.

    Each branch has:
    - A probability score (0.0–1.0) that updates as evidence arrives
    - A ThoughtStream feed (live reasoning visible to all branches)
    - A halt signal (killed early when probability drops below threshold)
    - Results accumulated so far
    """

    def __init__(self, branch_id: str, hypothesis: str,
                 initial_probability: float,
                 tasks: list[dict[str, Any]] | None = None):
        self.branch_id = branch_id
        self.hypothesis = hypothesis       # What this branch is betting on
        self.probability = initial_probability
        self.tasks = tasks or []
        self.results: list[dict[str, Any]] = []
        self.halted = False
        self.halt_reason: str = ""
        self.started_at = time.monotonic()
        self.finished_at: float | None = None
        self.tokens_generated: int = 0
        self._evidence: list[dict[str, Any]] = []
        # [{type, data, delta, ts}] — evidence that shifted the probability

    def record_evidence(self, evidence_type: str, data: str,
                        probability_delta: float) -> None:
        """Record evidence that shifts this branch's probability.

        Positive delta = evidence supports this hypothesis.
        Negative delta = evidence contradicts it.
        """
        self.probability = max(0.0, min(1.0,
                                         self.probability + probability_delta))
        self._evidence.append({
            "type": evidence_type,
            "data": data[:200],
            "delta": probability_delta,
            "new_prob": self.probability,
            "ts": time.monotonic(),
        })

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        self.finished_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.monotonic()
        return end - self.started_at

    @property
    def is_alive(self) -> bool:
        return not self.halted

    def summary(self) -> str:
        status = "ALIVE" if self.is_alive else f"HALTED({self.halt_reason})"
        return (f"[{self.branch_id}] p={self.probability:.2f} {status} "
                f"— \"{self.hypothesis}\" "
                f"({len(self.results)} results, {self.elapsed:.2f}s)")


class SpeculativeRouter:
    """Probability-based speculative batching — the Persona way.

    Instead of blind speculative execution (fire N branches, discard losers),
    this router spawns N concurrent agents weighted by probability, each
    streaming its reasoning via ThoughtStream. The router watches the
    probability landscape shift in real time:

    - High-probability branches get priority task claims
    - Low-probability branches are halted early (saving compute)
    - Evidence from ANY branch updates ALL branches' probabilities
    - The winning branch's partial work is KEPT, not recomputed

    This is NOT speculative decoding (token-level prediction).
    This is speculative REASONING — multiple hypothesis-driven agents
    pursuing different strategies simultaneously, with a shared mind
    that prunes losers and amplifies winners mid-execution.

    Example: delegating a task where the right approach is unclear.
        Branch A (p=0.4): "Delegate to Hermes — it has the right tools"
        Branch B (p=0.35): "Call the API directly — faster for this"
        Branch C (p=0.25): "Run a shell command — simpler path"

    As Branch B's API call returns a 200, its probability rises.
    Branch A's delegation times out — its probability drops below
    threshold and is halted. Branch C's command finds the data —
    its results feed into Branch B's pipeline. Final output:
    Branch B's result, enriched by Branch C's discovery, with
    Branch A's wasted work cut short early.

    Works with any tool call — browser, agent delegation, API,
    file operations, shell commands.

    Integration with ThoughtStream:
        Every branch's reasoning is visible to all other branches.
        Before Branch B fills a field, it sees:
            [branch-A] INTENT: click '#next-btn' — assuming wizard flow
            [branch-C] THINKING: "Found hidden section, expanding..."
        This lets branches adjust their own probability estimates
        reactively, not just from their own evidence.
    """

    # ── Speed dial ──────────────────────────────────────────────
    # One number, 1–10. Higher = faster execution, more token burn.
    #
    #   Speed 1:  1 branch, aggressive pruning  (sequential, cheap)
    #   Speed 3:  3 branches, moderate pruning   (balanced)
    #   Speed 5:  5 branches, light pruning      (fast)
    #   Speed 7:  8 branches, minimal pruning    (aggressive)
    #   Speed 10: 16 branches, NO pruning        (all possibilities, max burn)
    #
    # The user sees a slider. Internally it controls:
    #   max_branches  — how many concurrent possibilities
    #   halt_threshold — how quickly losers get killed
    #   branch_depth  — how deep the branching tree goes
    SPEED_PROFILES: ClassVar[dict[int, dict[str, Any]]] = {
        1:  {"max_branches": 1,  "halt_threshold": 0.30, "promotion_threshold": 0.6,  "branch_depth": 1},
        2:  {"max_branches": 2,  "halt_threshold": 0.20, "promotion_threshold": 0.65, "branch_depth": 2},
        3:  {"max_branches": 3,  "halt_threshold": 0.15, "promotion_threshold": 0.7,  "branch_depth": 3},
        4:  {"max_branches": 4,  "halt_threshold": 0.10, "promotion_threshold": 0.75, "branch_depth": 3},
        5:  {"max_branches": 5,  "halt_threshold": 0.08, "promotion_threshold": 0.8,  "branch_depth": 4},
        6:  {"max_branches": 6,  "halt_threshold": 0.05, "promotion_threshold": 0.85, "branch_depth": 5},
        7:  {"max_branches": 8,  "halt_threshold": 0.03, "promotion_threshold": 0.9,  "branch_depth": 6},
        8:  {"max_branches": 10, "halt_threshold": 0.02, "promotion_threshold": 0.95, "branch_depth": 7},
        9:  {"max_branches": 12, "halt_threshold": 0.01, "promotion_threshold": 0.99, "branch_depth": 8},
        10: {"max_branches": 16, "halt_threshold": 0.00, "promotion_threshold": 1.10, "branch_depth": 10},
    }

    def __init__(self, thought_stream: ThoughtStream | None = None,
                 speed: int = 5,
                 halt_threshold: float | None = None,
                 promotion_threshold: float | None = None,
                 max_branches: int | None = None):
        self._lock = asyncio.Lock()
        self.thought_stream = thought_stream or ThoughtStream()

        # Speed dial drives everything — explicit overrides still work
        self.speed = max(1, min(10, speed))
        profile = self.SPEED_PROFILES[self.speed]
        self.halt_threshold = halt_threshold if halt_threshold is not None else profile["halt_threshold"]
        self.promotion_threshold = promotion_threshold if promotion_threshold is not None else profile["promotion_threshold"]
        self.max_branches = max_branches if max_branches is not None else profile["max_branches"]
        self.branch_depth = profile["branch_depth"]

        self._branches: dict[str, SpeculativeBranch] = {}
        self._branch_tree: dict[str, list[str]] = {}  # parent → [children]
        self._branch_generation: dict[str, int] = {}   # branch_id → depth
        self._events: list[dict[str, Any]] = []
        self._winner: str | None = None
        self._started_at = time.monotonic()

        log.info("SpecRouter: speed=%d (max=%d branches, halt=%.2f, promo=%.2f, depth=%d)",
                 self.speed, self.max_branches, self.halt_threshold,
                 self.promotion_threshold, self.branch_depth)

    # ── Branch management ──────────────────────────────────────────

    def add_branch(self, branch_id: str, hypothesis: str,
                   probability: float,
                   tasks: list[dict[str, Any]] | None = None) -> SpeculativeBranch:
        """Register a new probability branch.

        Probabilities across all branches don't need to sum to 1.0 —
        they're independent confidence scores, not a distribution.
        (Though normalizing them is valid for the pitch.)
        """
        if len(self._branches) >= self.max_branches:
            raise ValueError(f"Max {self.max_branches} branches")
        branch = SpeculativeBranch(branch_id, hypothesis, probability, tasks)
        self._branches[branch_id] = branch
        self._events.append({
            "event": "branch_added", "branch": branch_id,
            "hypothesis": hypothesis, "probability": probability,
            "ts": time.monotonic(),
        })
        log.info("SpecRouter: +%s (p=%.2f) \"%s\"",
                 branch_id, probability, hypothesis)
        return branch

    @classmethod
    def from_speed(cls, speed: int,
                   thought_stream: ThoughtStream | None = None) -> SpeculativeRouter:
        """Create a router from just a speed dial value (1–10).

        This is the user-facing API:
            router = SpeculativeRouter.from_speed(7)  # aggressive
            router = SpeculativeRouter.from_speed(3)  # balanced
        """
        return cls(thought_stream=thought_stream, speed=speed)

    def branch_from(self, parent_id: str,
                    child_hypotheses: list[dict[str, Any]]) -> list[SpeculativeBranch]:
        """Spawn child branches from a parent — recursive branching tree.

        After a branch completes an action and sees N possible next steps,
        it branches into N children. Each child inherits the parent's
        accumulated results and runs its own hypothesis.

        This is the aggressive model: after every action, the possibility
        space fans out. Speed dial controls how deep this goes.

        Args:
            parent_id: the branch that's splitting
            child_hypotheses: list of {hypothesis: str, probability: float,
                                        tasks: list[dict]}

        Returns:
            List of newly created child branches (empty if depth exceeded
            or max branches hit).
        """
        parent = self._branches.get(parent_id)
        if not parent:
            return []

        parent_gen = self._branch_generation.get(parent_id, 0)
        if parent_gen >= self.branch_depth:
            log.info("SpecRouter: %s at max depth %d, no branching",
                     parent_id, self.branch_depth)
            return []

        children = []
        for i, hyp in enumerate(child_hypotheses):
            if len(self._branches) >= self.max_branches:
                log.info("SpecRouter: max branches %d hit, stopping branching",
                         self.max_branches)
                break

            child_id = f"{parent_id}.{i}"
            child = SpeculativeBranch(
                child_id,
                hyp["hypothesis"],
                hyp.get("probability", parent.probability * 0.8),
                hyp.get("tasks", []),
            )
            # Inherit parent's results
            child.results = list(parent.results)
            self._branches[child_id] = child
            self._branch_generation[child_id] = parent_gen + 1

            # Track tree structure
            if parent_id not in self._branch_tree:
                self._branch_tree[parent_id] = []
            self._branch_tree[parent_id].append(child_id)

            self._events.append({
                "event": "branch_spawned", "parent": parent_id,
                "child": child_id, "generation": parent_gen + 1,
                "hypothesis": hyp["hypothesis"],
                "probability": child.probability,
                "ts": time.monotonic(),
            })
            children.append(child)
            log.info("SpecRouter: %s → %s (gen=%d, p=%.2f) \"%s\"",
                     parent_id, child_id, parent_gen + 1,
                     child.probability, hyp["hypothesis"])

        # Halt parent — children take over
        if children:
            parent.halt(f"branched into {len(children)} children")

        return children

    def get_tree_depth(self) -> int:
        """Current max depth of the branching tree."""
        if not self._branch_generation:
            return 0
        return max(self._branch_generation.values())

    def get_tree_visualization(self) -> str:
        """ASCII tree of all branches for logging/display."""
        lines = [f"SpecRouter Tree (speed={self.speed}, depth={self.get_tree_depth()}/{self.branch_depth}):"]

        # Find root branches (no parent)
        all_children = set()
        for kids in self._branch_tree.values():
            all_children.update(kids)
        roots = [bid for bid in self._branches if bid not in all_children]

        def render(bid: str, prefix: str = "  ", is_last: bool = True):
            branch = self._branches.get(bid)
            if not branch:
                return
            connector = "└── " if is_last else "├── "
            status = "✓" if branch.is_alive else "✗"
            lines.append(f"{prefix}{connector}[{bid}] p={branch.probability:.2f} {status} \"{branch.hypothesis}\"")
            children = self._branch_tree.get(bid, [])
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_id in enumerate(children):
                render(child_id, child_prefix, i == len(children) - 1)

        for i, root in enumerate(sorted(roots)):
            render(root, "  ", i == len(roots) - 1)

        return "\n".join(lines)

    async def update_probability(self, branch_id: str,
                                  evidence_type: str, data: str,
                                  delta: float) -> dict[str, Any]:
        """Update a branch's probability based on new evidence.

        Returns the action taken: {"action": "continue"|"halt"|"promote",
                                    "branch": ..., "new_prob": ...}

        Called by:
        - The branch's own hand (self-assessment from execution results)
        - Other branches (cross-evidence from ThoughtStream)
        - An external scorer (if wired up)
        """
        async with self._lock:
            branch = self._branches.get(branch_id)
            if not branch or branch.halted:
                return {"action": "skip", "branch": branch_id,
                        "reason": "not found or halted"}

            old_prob = branch.probability
            branch.record_evidence(evidence_type, data, delta)
            new_prob = branch.probability

            self._events.append({
                "event": "probability_update", "branch": branch_id,
                "old": old_prob, "new": new_prob, "delta": delta,
                "evidence": evidence_type, "data": data[:100],
                "ts": time.monotonic(),
            })

            # Broadcast probability shift via ThoughtStream
            await self.thought_stream.push_tokens(
                branch_id,
                f"[p={old_prob:.2f}→{new_prob:.2f}] {evidence_type}: {data[:80]}"
            )

            # ── Decision: halt, promote, or continue ───────────────
            if new_prob <= self.halt_threshold:
                branch.halt(f"probability {new_prob:.3f} ≤ {self.halt_threshold}")
                log.info("SpecRouter: HALT %s (p=%.3f ≤ %.2f)",
                         branch_id, new_prob, self.halt_threshold)
                return {"action": "halt", "branch": branch_id,
                        "new_prob": new_prob,
                        "reason": f"below threshold {self.halt_threshold}"}

            if new_prob >= self.promotion_threshold and not self._winner:
                self._winner = branch_id
                # Halt all other branches
                halted_others = []
                for bid, b in self._branches.items():
                    if bid != branch_id and b.is_alive:
                        b.halt(f"outcompeted by {branch_id} (p={new_prob:.2f})")
                        halted_others.append(bid)
                log.info("SpecRouter: PROMOTE %s (p=%.2f) — halted %s",
                         branch_id, new_prob, halted_others)
                return {"action": "promote", "branch": branch_id,
                        "new_prob": new_prob,
                        "halted_others": halted_others}

            return {"action": "continue", "branch": branch_id,
                    "new_prob": new_prob}

    async def cross_pollinate(self, source_branch: str,
                               discovery: str,
                               target_branches: list[str] | None = None
                               ) -> None:
        """Feed one branch's discovery into other branches.

        When Branch C finds hidden fields, that evidence helps Branch B
        (which is filling visible fields) but hurts Branch A (which
        assumed a wizard flow). The router adjusts both.

        If target_branches is None, broadcasts to all alive branches.
        """
        targets = target_branches or [
            bid for bid, b in self._branches.items()
            if bid != source_branch and b.is_alive
        ]
        for bid in targets:
            await self.thought_stream.push_tokens(
                bid,
                f"[cross-evidence from {source_branch}] {discovery[:200]}"
            )
        self._events.append({
            "event": "cross_pollinate", "source": source_branch,
            "targets": targets, "discovery": discovery[:200],
            "ts": time.monotonic(),
        })

    # ── Execution ──────────────────────────────────────────────────

    async def run_branches(self, executor_fn,
                            timeout: float = 30.0) -> dict[str, Any]:
        """Run all branches concurrently, with probability-based routing.

        executor_fn(branch: SpeculativeBranch, router: SpeculativeRouter)
            -> list[dict] (results)

        The executor is called for each branch. Inside it, the executor
        should periodically call router.update_probability() and check
        branch.halted before each action.

        Returns:
            {
                "winner": branch_id or None,
                "branches": {id: branch.summary(), ...},
                "results": {id: [...results...], ...},
                "elapsed": float,
                "events": [...],
            }
        """
        alive = [b for b in self._branches.values() if b.is_alive]
        if not alive:
            return {"winner": None, "branches": {}, "results": {},
                    "elapsed": 0, "events": self._events}

        log.info("SpecRouter: running %d branches concurrently", len(alive))

        async def run_one(branch: SpeculativeBranch):
            try:
                results = await asyncio.wait_for(
                    executor_fn(branch, self),
                    timeout=timeout,
                )
                branch.results = results or []
                if not branch.halted:
                    branch.finished_at = time.monotonic()
            except asyncio.TimeoutError:
                branch.halt("timeout")
            except Exception as e:
                branch.halt(f"error: {str(e)[:100]}")
                log.error("SpecRouter: %s failed: %s", branch.branch_id, e)

        t0 = time.monotonic()
        await asyncio.gather(*[run_one(b) for b in alive],
                              return_exceptions=True)
        elapsed = time.monotonic() - t0

        # If no winner was promoted during execution, pick highest prob
        if not self._winner:
            alive_after = [b for b in self._branches.values()
                           if b.results]
            if alive_after:
                self._winner = max(alive_after,
                                    key=lambda b: b.probability).branch_id

        return {
            "winner": self._winner,
            "branches": {bid: b.summary()
                         for bid, b in self._branches.items()},
            "results": {bid: b.results
                        for bid, b in self._branches.items()},
            "elapsed": elapsed,
            "events": self._events,
            "probability_history": {
                bid: [(e["delta"], e["new_prob"])
                      for e in b._evidence]
                for bid, b in self._branches.items()
            },
        }

    # ── Queries ────────────────────────────────────────────────────

    def get_branch(self, branch_id: str) -> SpeculativeBranch | None:
        return self._branches.get(branch_id)

    def alive_branches(self) -> list[SpeculativeBranch]:
        return [b for b in self._branches.values() if b.is_alive]

    def get_winner(self) -> SpeculativeBranch | None:
        if self._winner:
            return self._branches.get(self._winner)
        return None

    async def get_landscape(self) -> str:
        """Probability landscape — snapshot of all branches for logging."""
        async with self._lock:
            lines = [f"SpecRouter: {len(self._branches)} branches, "
                     f"winner={self._winner or 'none'}"]
            for bid, b in sorted(self._branches.items(),
                                  key=lambda x: -x[1].probability):
                bar = "█" * int(b.probability * 20)
                lines.append(f"  {b.summary()}")
                lines.append(f"    [{bar:<20}] {b.probability:.0%}")
            return "\n".join(lines)

    async def get_summary(self) -> str:
        """Compact summary for reporting."""
        async with self._lock:
            elapsed = time.monotonic() - self._started_at
            total_results = sum(len(b.results) for b in self._branches.values())
            alive = sum(1 for b in self._branches.values() if b.is_alive)
            halted = sum(1 for b in self._branches.values() if b.halted)
            return (f"SpecRouter: {len(self._branches)} branches "
                    f"({alive} alive, {halted} halted), "
                    f"winner={self._winner or 'pending'}, "
                    f"{total_results} total results, {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# PossibilitySpeculator — pre-compute-and-hold for zero-gap dispatch
# ═══════════════════════════════════════════════════════════════════════════

class PossibilitySpeculator:
    """Pre-compute outcomes AT DECISION TIME, before the tool even executes.

    This is how humans think. The moment you decide to do something —
    click a button, call an API, delegate a task, run a command — you
    already know the realistic outcomes:
        - It works → do the next thing
        - It fails → handle the error

    So you pre-compute ALL paths RIGHT THEN, at decision time, before
    the tool even fires. Pre-compute and action run in parallel.
    By the time the result comes back, the next action is already
    waiting — zero inference gap.

    Works with ANY tool call — browser actions, agent delegation
    (Hermes/Cursor/Claude Code/Cline), API calls, file operations,
    shell commands — anything that's an async callable.

    Timeline:
        DECIDE "execute tool X"
            ├── START action (tool call)              ← runs anywhere
            └── START pre-compute                     ← runs in LLM, same moment
                    ├── if success → prepare next step
                    └── if error   → prepare fallback
        ACTION COMPLETES → match result → INSTANT dispatch

    The dial (1–10) controls how many outcomes to pre-compute:
        Dial 1:  2 outcomes (success/fail — the minimum)
        Dial 5:  4 outcomes (covers most tool results)
        Dial 10: 10 outcomes (every conceivable result)

    Higher dial = more token burn, but more likely to hit on dispatch.
    Most tool calls have 2–3 realistic outcomes, so dial 3–5
    covers nearly everything without waste.
    """

    DIAL_PROFILES: ClassVar[dict[int, dict[str, Any]]] = {
        1:  {"max_possibilities": 2},
        2:  {"max_possibilities": 2},
        3:  {"max_possibilities": 3},
        4:  {"max_possibilities": 3},
        5:  {"max_possibilities": 4},
        6:  {"max_possibilities": 5},
        7:  {"max_possibilities": 6},
        8:  {"max_possibilities": 7},
        9:  {"max_possibilities": 8},
        10: {"max_possibilities": 10},
    }

    def __init__(self, thought_stream: ThoughtStream | None = None,
                 dial: int = 5,
                 max_possibilities: int | None = None,
                 precompute_grace: float = 0.05):
        self.thought_stream = thought_stream or ThoughtStream()
        self.dial = max(1, min(10, dial))
        profile = self.DIAL_PROFILES[self.dial]
        self.max_possibilities = max_possibilities or profile["max_possibilities"]
        self.precompute_grace = max(0.0, precompute_grace)

        # Runtime state
        self._lock = asyncio.Lock()
        self._step_count = 0
        self._instant_dispatches = 0
        self._fallback_thinks = 0
        self._total_think_time = 0.0
        self._total_action_time = 0.0
        self._total_precompute_time = 0.0
        self._events: list[dict[str, Any]] = []
        self._started_at = time.monotonic()

        log.info("PossibilitySpeculator: dial=%d (max_poss=%d)",
                 self.dial, self.max_possibilities)

    @classmethod
    def from_dial(cls, dial: int,
                  thought_stream: ThoughtStream | None = None) -> "PossibilitySpeculator":
        """Create from dial value alone."""
        return cls(thought_stream=thought_stream, dial=dial)

    async def precompute_possibilities(
        self,
        compute_fn,
        step_context: dict[str, Any] | None = None,
    ) -> dict[str, dict]:
        """Pre-compute possible outcomes and their next actions.

        compute_fn(step_context) -> dict mapping outcome_key -> action_dict.
        Called at DECISION TIME, before the action executes.

        Returns the possibilities dict, ready to hold until the
        action result arrives.
        """
        t0 = time.monotonic()
        possibilities = compute_fn(step_context)
        if inspect.isawaitable(possibilities):
            possibilities = await possibilities
        if not isinstance(possibilities, dict):
            raise TypeError("possibilities must be a dict")
        elapsed = time.monotonic() - t0
        self._total_precompute_time += elapsed

        # Trim to max
        if len(possibilities) > self.max_possibilities:
            keys = list(possibilities.keys())[:self.max_possibilities]
            possibilities = {k: possibilities[k] for k in keys}

        async with self._lock:
            self._events.append({
                "event": "precompute",
                "step": self._step_count,
                "num_possibilities": len(possibilities),
                "elapsed": elapsed,
                "ts": time.monotonic(),
            })

        return possibilities

    async def decide_and_act(
        self,
        action_fn,
        precompute_fn,
        step_context: dict[str, Any] | None = None,
        is_last_step: bool = False,
    ) -> tuple[Any, dict | None]:
        """The core loop: decide, then action + pre-compute in parallel.

        At decision time (NOW), we already know what we're about to do.
        So we SIMULTANEOUSLY:
            1. Execute the action (any tool call — browser, API, delegation)
            2. Pre-compute what happens next for each possible outcome

        Both start at the same moment — the decision point.

        Args:
            action_fn: coroutine that executes the browser action
            precompute_fn: coroutine(ctx) -> {outcome_key: next_action}
            step_context: context (page state, history, decision)
            is_last_step: skip precompute on last step

        Returns:
            (action_result, possibilities_or_None)
        """
        self._step_count += 1

        if is_last_step:
            t0 = time.monotonic()
            result = await action_fn()
            self._total_action_time += time.monotonic() - t0
            return result, None

        # Speculation is advisory: slow or failed guesses cannot hold the result.
        t0 = time.monotonic()
        precompute = asyncio.create_task(
            self.precompute_possibilities(precompute_fn, step_context)
        )
        try:
            result = await action_fn()
            self._total_action_time += time.monotonic() - t0
            try:
                if not precompute.done():
                    await asyncio.wait({precompute}, timeout=self.precompute_grace)
                possibilities = (precompute.result()
                                 if precompute.done() and not precompute.cancelled() else None)
            except Exception:
                log.debug("possibility precompute failed; use fresh evidence", exc_info=True)
                possibilities = None
            return result, possibilities
        finally:
            if not precompute.done():
                precompute.cancel()
            await asyncio.gather(precompute, return_exceptions=True)

    # Keep old name as alias for SpeculativeEngine compatibility
    execute_with_precompute = decide_and_act

    @staticmethod
    def outcome_key(result: Any) -> str:
        raw = result
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                pass
        if isinstance(raw, dict) and isinstance(raw.get("status"), str):
            return raw["status"]
        return str(result)

    def dispatch(self, result: Any, possibilities: dict[str, dict] | None
                 ) -> dict | None:
        """Match action result against pre-computed possibilities.

        Returns the matching next action if found, None if no match
        (caller falls back to a fresh LLM think).
        """
        if not possibilities:
            return None

        result_key = str(result)
        if result_key not in possibilities:
            result_key = self.outcome_key(result)
        if result_key in possibilities:
            self._instant_dispatches += 1
            self._events.append({
                "event": "instant_dispatch",
                "step": self._step_count,
                "result_key": result_key[:100],
                "ts": time.monotonic(),
            })
            log.debug("PossSpec: instant dispatch step %d (key=%s)",
                      self._step_count, result_key[:50])
            return possibilities[result_key]

        # No match — caller must think fresh
        self._fallback_thinks += 1
        self._events.append({
            "event": "fallback_think",
            "step": self._step_count,
            "result_key": result_key[:100],
            "available_keys": list(possibilities.keys())[:5],
            "ts": time.monotonic(),
        })
        log.debug("PossSpec: fallback think step %d (key=%s not in %s)",
                  self._step_count, result_key[:50],
                  list(possibilities.keys())[:3])
        return None

    async def record_think(self, duration: float) -> None:
        """Record a fallback think time (when dispatch missed)."""
        self._total_think_time += duration

    # ── Queries ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self._started_at
        return {
            "dial": self.dial,
            "steps": self._step_count,
            "instant_dispatches": self._instant_dispatches,
            "fallback_thinks": self._fallback_thinks,
            "hit_rate": (self._instant_dispatches /
                         max(1, self._instant_dispatches + self._fallback_thinks)),
            "think_time": self._total_think_time,
            "action_time": self._total_action_time,
            "precompute_time": self._total_precompute_time,
            "elapsed": elapsed,
        }

    def get_summary(self) -> str:
        s = self.stats
        return (f"PossSpec dial={s['dial']}: {s['steps']} steps, "
                f"{s['instant_dispatches']} instant / "
                f"{s['fallback_thinks']} fallback "
                f"({s['hit_rate']:.0%} hit), "
                f"think={s['think_time']:.1f}s, "
                f"elapsed={s['elapsed']:.1f}s")

    def get_timeline(self) -> str:
        """ASCII timeline of events for debugging."""
        lines = [f"PossSpec Timeline (dial={self.dial}):"]
        for e in self._events:
            t = e["ts"] - self._started_at
            if e["event"] == "instant_dispatch":
                lines.append(f"  {t:6.2f}s ⚡ INSTANT step {e['step']}")
            elif e["event"] == "fallback_think":
                lines.append(f"  {t:6.2f}s 🐌 FALLBACK step {e['step']} "
                             f"(wanted {e['result_key'][:30]})")
            elif e["event"] == "precompute":
                lines.append(f"  {t:6.2f}s 🔮 PRECOMPUTE step {e['step']} "
                             f"({e['num_possibilities']} poss, {e['elapsed']:.2f}s)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SpeculativeEngine — unified probability batching + possibility pipeline
# ═══════════════════════════════════════════════════════════════════════════

class SpeculativeEngine:
    """Combines SpeculativeRouter's probability batching with
    PossibilitySpeculator's pre-compute-and-hold pipeline.

    Core principle: ALWAYS pre-compute possibilities from the previous
    decision point. Pre-computation starts the moment you have something
    solid — the result of the last step IS that solid ground.

    On TOP of that, high-confidence batching fires immediately:

        Every step:
            → Pre-compute possibilities (ALWAYS, from previous decision)
            → These run in background during current action

        If confidence >= threshold (we KNOW what's next):
            → ALSO batch-fire the next action immediately
            → If the batch was WRONG, possibilities are still ready
            → Example: delegated task to Hermes → 95% sure it'll succeed,
              pre-compute covers the 5% where agent fails

        If confidence < threshold (uncertain):
            → Only possibilities — wait for result, instant-dispatch match
            → Example: called API → could be success/error/rate-limit

    This means high-confidence steps get BOTH speed AND safety:
        - The batch fires with zero delay (speed)
        - Pre-computed possibilities catch batch misses (safety)
        - Low-confidence steps still get zero-gap dispatch

    The confidence_threshold is driven by the dial:
        Dial 1:  threshold=0.9 (rarely batches, mostly possibilities)
        Dial 5:  threshold=0.7 (batches when fairly confident)
        Dial 10: threshold=0.5 (batches aggressively, possibilities backup)
    """

    DIAL_PROFILES: ClassVar[dict[int, dict[str, Any]]] = {
        1:  {"confidence_threshold": 0.90, "max_batch": 1,  "max_possibilities": 2},
        2:  {"confidence_threshold": 0.85, "max_batch": 2,  "max_possibilities": 3},
        3:  {"confidence_threshold": 0.80, "max_batch": 2,  "max_possibilities": 3},
        4:  {"confidence_threshold": 0.75, "max_batch": 3,  "max_possibilities": 4},
        5:  {"confidence_threshold": 0.70, "max_batch": 3,  "max_possibilities": 4},
        6:  {"confidence_threshold": 0.65, "max_batch": 4,  "max_possibilities": 5},
        7:  {"confidence_threshold": 0.60, "max_batch": 5,  "max_possibilities": 6},
        8:  {"confidence_threshold": 0.55, "max_batch": 6,  "max_possibilities": 8},
        9:  {"confidence_threshold": 0.52, "max_batch": 8,  "max_possibilities": 10},
        10: {"confidence_threshold": 0.50, "max_batch": 10, "max_possibilities": 12},
    }

    def __init__(self, thought_stream: ThoughtStream | None = None,
                 dial: int = 5,
                 confidence_threshold: float | None = None):
        self.thought_stream = thought_stream or ThoughtStream()
        self.dial = max(1, min(10, dial))
        profile = self.DIAL_PROFILES[self.dial]

        self.confidence_threshold = (confidence_threshold
                                     if confidence_threshold is not None
                                     else profile["confidence_threshold"])
        self.max_batch = profile["max_batch"]
        self.max_possibilities = profile["max_possibilities"]

        # Sub-engines
        self.router = SpeculativeRouter(
            thought_stream=self.thought_stream,
            speed=self.dial,
        )
        self.speculator = PossibilitySpeculator(
            thought_stream=self.thought_stream,
            dial=self.dial,
            max_possibilities=self.max_possibilities,
        )

        # Stats
        self._lock = asyncio.Lock()
        self._step_count = 0
        self._high_path_count = 0    # batched (confident)
        self._low_path_count = 0     # pre-computed (uncertain)
        self._events: list[dict[str, Any]] = []
        self._started_at = time.monotonic()

        log.info("SpecEngine: dial=%d (threshold=%.0f%%, max_batch=%d, max_poss=%d)",
                 self.dial, self.confidence_threshold * 100,
                 self.max_batch, self.max_possibilities)

    @classmethod
    def from_dial(cls, dial: int,
                  thought_stream: ThoughtStream | None = None) -> "SpeculativeEngine":
        """Create from dial value alone."""
        return cls(thought_stream=thought_stream, dial=dial)

    async def step(
        self,
        confidence: float,
        action_fn,
        precompute_fn=None,
        batch_actions: list[dict[str, Any]] | None = None,
        step_context: dict[str, Any] | None = None,
        is_last_step: bool = False,
    ) -> dict[str, Any]:
        """Execute one step — ALWAYS pre-compute, batch on top if confident.

        Pre-computation runs every step (from the previous decision).
        High confidence adds batch-firing ON TOP, with possibilities as
        a safety net if the batch was wrong.

        Args:
            confidence: 0.0–1.0, how sure we are about the next action.
            action_fn: async callable that executes the current action.
            precompute_fn: async callable(ctx) → {outcome_key: action_dict}.
                           ALWAYS runs (unless is_last_step).
            batch_actions: list of actions to batch-fire immediately when
                          confidence is high and _expected_outcome matches.
                          These are suggestions; the caller executes them.
            step_context: context dict passed to precompute_fn.
            is_last_step: if True, skip pre-computation.

        Returns:
            {
                "result": action result,
                "path": "high+precompute" | "precompute",
                "next_action": dict | None,
                "batch_fired": bool,
                "possibilities": dict | None,
                "confidence": float,
            }
        """
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite probability between 0 and 1")
        async with self._lock:
            self._step_count += 1
            step_num = self._step_count

        is_high = confidence >= self.confidence_threshold
        if is_high:
            self._high_path_count += 1
        else:
            self._low_path_count += 1

        # ── ALWAYS pre-compute possibilities during action ─────────
        # This is the foundation — every step gets possibilities ready
        result, possibilities = await self.speculator.execute_with_precompute(
            action_fn,
            precompute_fn or (lambda ctx: {}),
            step_context=step_context,
            is_last_step=is_last_step,
        )

        next_action = None
        batch_fired = False

        if is_high and batch_actions and not is_last_step:
            # ── HIGH CONFIDENCE: batch fires ON TOP ────────────────
            # We already pre-computed possibilities (safety net).
            # Now also use the batch action — it's what we're confident about.
            batch_size = min(len(batch_actions), self.max_batch)
            batch_to_fire = batch_actions[:batch_size]
            predicted = batch_to_fire[0] if batch_to_fire else None
            expected = predicted.get("_expected_outcome") if predicted else None
            batch_fired = bool(expected and (
                expected == str(result) or expected == self.speculator.outcome_key(result)))
            next_action = predicted if batch_fired else self.speculator.dispatch(result, possibilities)

            self._events.append({
                "event": "high_plus_precompute",
                "step": step_num,
                "confidence": confidence,
                "batch_fired": batch_fired,
                "batch_size": batch_size,
                "num_possibilities": len(possibilities) if possibilities else 0,
                "ts": time.monotonic(),
            })
            log.debug("SpecEngine: HIGH+PRECOMPUTE step %d (conf=%.0f%%, "
                      "batch=%s, %d poss ready)",
                      step_num, confidence * 100, batch_fired,
                      len(possibilities) if possibilities else 0)

            return {
                "result": result,
                "path": "high+precompute",
                "next_action": next_action,
                "batch_remaining": batch_to_fire[1:] if batch_fired and len(batch_to_fire) > 1 else [],
                "batch_fired": batch_fired,
                "possibilities": possibilities,
                "confidence": confidence,
            }

        else:
            # ── PRECOMPUTE ONLY (low confidence or no batch) ───────
            # Possibilities are our only play — instant-dispatch on match
            if possibilities and not is_last_step:
                next_action = self.speculator.dispatch(result, possibilities)

            self._events.append({
                "event": "precompute_only",
                "step": step_num,
                "confidence": confidence,
                "dispatched": next_action is not None,
                "num_possibilities": len(possibilities) if possibilities else 0,
                "ts": time.monotonic(),
            })
            log.debug("SpecEngine: PRECOMPUTE step %d (conf=%.0f%%, dispatched=%s)",
                      step_num, confidence * 100, next_action is not None)

            return {
                "result": result,
                "path": "precompute",
                "next_action": next_action,
                "batch_fired": False,
                "possibilities": possibilities,
                "confidence": confidence,
            }

    # ── Queries ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        spec_stats = self.speculator.stats
        elapsed = time.monotonic() - self._started_at
        return {
            "dial": self.dial,
            "confidence_threshold": self.confidence_threshold,
            "steps": self._step_count,
            "high_path": self._high_path_count,
            "low_path": self._low_path_count,
            "instant_dispatches": spec_stats["instant_dispatches"],
            "fallback_thinks": spec_stats["fallback_thinks"],
            "hit_rate": spec_stats["hit_rate"],
            "elapsed": elapsed,
        }

    def get_summary(self) -> str:
        s = self.stats
        return (f"SpecEngine dial={s['dial']}: {s['steps']} steps, "
                f"{s['high_path']} high-conf / {s['low_path']} low-conf, "
                f"{s['instant_dispatches']} instant dispatches "
                f"({s['hit_rate']:.0%} hit), "
                f"elapsed={s['elapsed']:.1f}s")

    def get_timeline(self) -> str:
        """ASCII timeline of engine decisions."""
        lines = [f"SpecEngine Timeline (dial={self.dial}, "
                 f"threshold={self.confidence_threshold:.0%}):"]
        for e in self._events:
            t = e["ts"] - self._started_at
            conf = e["confidence"]
            if e["event"] == "high_plus_precompute":
                safety = "" if e["batch_fired"] else " → SAFETY NET"
                lines.append(f"  {t:6.2f}s 🚀 HIGH+PRE step {e['step']} "
                             f"(conf={conf:.0%}, batch={e['batch_size']}, "
                             f"{e['num_possibilities']} poss{safety})")
            elif e["event"] == "precompute_only":
                marker = "⚡" if e["dispatched"] else "🔮"
                lines.append(f"  {t:6.2f}s {marker} PRE  step {e['step']} "
                             f"(conf={conf:.0%}, "
                             f"{e['num_possibilities']} poss, "
                             f"{'instant' if e['dispatched'] else 'fallback'})")
        return "\n".join(lines)
