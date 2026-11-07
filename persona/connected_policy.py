"""Goal-conditioned, jointly-rewarded L/R coordination policy.

Machine-learning component (contextual policy-gradient bandit), NOT a
deterministic 'L always executes / R always talks' scheduler. Actual model
weight adaptation is a separate offline SFT/preference training stage and
requires reviewed trajectories.

Candidate actions are proposed by the current L/R models. This learner only
ranks safe, explicitly eligible actions and learns from independent whole-team
success. Guardrails and authorization remain outside the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class TeamChoice:
    point: str
    option: str
    context: tuple[str, ...]
    probability: float


class GoalConditionedPolicy:
    """Small, online contextual softmax policy; both lobes share team reward.

    This is genuine online parameter updating but NOT Tinker neural SGD.
    No optimizer updates from unverified task receipts or model self-judgment.
    Caller owns persistence, review gating, and safe-action filtering.
    """

    def __init__(self, *, learning_rate: float = 0.12, temperature: float = 0.8,
                 seed: int = 0, weights: dict | None = None):
        if not 0 < learning_rate <= 0.5 or not 0.1 <= temperature <= 2.0:
            raise ValueError("Unsafe policy hyperparameters")
        self.learning_rate = learning_rate
        self.temperature = temperature
        self._random = random.Random(seed)
        self.weights: dict[str, float] = dict(weights or {})
        self.baseline = 0.5
        self.updates = 0

    @staticmethod
    def features(goal_type: str, *, user_engaged: bool, resources: Iterable[str],
                 role: str, has_parallel_work: bool, verify_pending: bool) -> tuple[str, ...]:
        """Bounded goal/state features, not raw user speech or tool contents."""
        if role not in ("L", "R", "team"):
            raise ValueError("Bad role")
        allowed = {"research", "coding", "browser", "computer", "conversation",
                   "general", "other"}
        category = goal_type if goal_type in allowed else "other"
        resources = sorted({r for r in resources if r in {"browser", "computer", "cloud", "other"}})
        return (f"goal:{category}", f"role:{role}",
                "engaged" if user_engaged else "disengaged",
                "parallel" if has_parallel_work else "dependent",
                "verify_pending" if verify_pending else "verified_or_no_claim",
                *(f"resource:{r}" for r in resources))

    def _key(self, point: str, option: str, feature: str) -> str:
        return f"{point}|{option}|{feature}"

    def _score(self, point: str, option: str, context: tuple[str, ...]) -> float:
        return sum(self.weights.get(self._key(point, option, feature), 0.0)
                   for feature in context) / math.sqrt(max(len(context), 1))

    def probabilities(self, point: str, options: Iterable[str], context: tuple[str, ...]
                      ) -> dict[str, float]:
        available = tuple(dict.fromkeys(options))
        if not available:
            raise ValueError("No authorized alternatives")
        scores = [self._score(point, o, context) / self.temperature for o in available]
        peak = max(scores)
        exps = [math.exp(s - peak) for s in scores]
        total = sum(exps)
        return {option: value / total for option, value in zip(available, exps)}

    def choose(self, point: str, options: Iterable[str], context: tuple[str, ...],
               *, explore: bool = False) -> TeamChoice:
        distribution = self.probabilities(point, options, context)
        # In production, choose the highest-probability authorized option; the
        # only random exploration is in isolated/approved training environments.
        if explore:
            threshold = self._random.random()
            cum = 0.0
            selected = next(reversed(distribution))
            for option, probability in distribution.items():
                cum += probability
                if threshold <= cum:
                    selected = option
                    break
        else:
            selected = max(distribution, key=distribution.get)
        return TeamChoice(point, selected, context, distribution[selected])

    def learn(self, choice: TeamChoice, authorized_options: Iterable[str],
              *, reward: float, independently_verified: bool,
              evidence_ref: str, sandbox: bool) -> None:
        """REINFORCE advantage update from independently verified team reward."""
        from .connected_training import _EVIDENCE
        if (not sandbox or not independently_verified or
                not isinstance(evidence_ref, str) or not _EVIDENCE.fullmatch(evidence_ref)):
            raise PermissionError("Learning requires independent evidence in authorized sandbox")
        if not 0.0 <= reward <= 1.0 or not math.isfinite(reward):
            raise ValueError("Bad team reward")
        options = tuple(dict.fromkeys(authorized_options))
        if choice.option not in options:
            raise ValueError("Chosen action is no longer authorized")
        probabilities = self.probabilities(choice.point, options, choice.context)
        advantage = reward - self.baseline
        scale = self.learning_rate / math.sqrt(max(len(choice.context), 1))
        for option in options:
            direction = (1.0 if option == choice.option else 0.0) - probabilities[option]
            for feature in choice.context:
                key = self._key(choice.point, option, feature)
                previous = self.weights.get(key, 0.0)
                self.weights[key] = min(5.0, max(-5.0, previous + scale * advantage * direction))
        self.baseline = 0.95 * self.baseline + 0.05 * reward
        self.updates += 1

    def snapshot(self) -> dict:
        """Weights only, no raw person or tool data."""
        return {"version": 1, "updates": self.updates,
                "baseline": round(self.baseline, 6),
                "weights": dict(sorted(self.weights.items()))}


class LearnedCoordinator:
    """Sandbox-only AdaptiveCoordinator that tries different L/R task roles.

    Resource ownership remains in the existing durable task runner. The
    learner decides only among safe allocation choices and never grants a
    tool permission or an irreversible action.
    """

    def __init__(self, policy: GoalConditionedPolicy, *,
                 goal_category: str = "general", explore: bool = True):
        from .kernel.adaptive import AdaptiveCoordinator
        self.base = AdaptiveCoordinator()
        self.policy = policy
        self.goal_category = goal_category
        self.explore = explore
        self.choices: list[tuple[TeamChoice, tuple[str, ...]]] = []

    def _choose(self, point: str, options: tuple[str, ...], *,
                engaged: bool, independent: int, role: str = "team") -> str:
        features = self.policy.features(
            self.goal_category, user_engaged=engaged, resources=[],
            role=role, has_parallel_work=independent >= 2,
            verify_pending=True)
        choice = self.policy.choose(point, options, features,
                                    explore=self.explore)
        self.choices.append((choice, options))
        return choice.option

    def allocate(self, *, user_engaged: bool, independent_work: int,
                 needs_verification: bool = False,
                 current_speaker: str = "L"):
        from .kernel.adaptive import Assignment
        if user_engaged or independent_work < 1:
            # The user's speaking lane is never wagered against task speed.
            return self.base.allocate(
                user_engaged=user_engaged, independent_work=independent_work,
                needs_verification=needs_verification,
                current_speaker=current_speaker)
        if independent_work >= 2:
            mode = self._choose("lobe_allocation",
                                ("dual_execution", "execute_and_verify"),
                                engaged=False, independent=independent_work)
            if mode == "dual_execution":
                return Assignment("L", "execute"), Assignment("R", "execute")
        executor = self._choose("execution_owner", ("L", "R"),
                                engaged=False, independent=independent_work)
        partner = "R" if executor == "L" else "L"
        return (Assignment(executor, "execute"),
                Assignment(partner, "verify" if needs_verification else "plan"))

    def resource(self, key: str, owner: str):
        return self.base.resource(key, owner)


async def train_verified_goal_policy(
        mind, *, persona_id: int, goal: str, policy: GoalConditionedPolicy,
        external_verify, approve_goal, user_permission: bool = False,
        goal_category: str = "general") -> dict:
    """One actual sandbox goal, running the real durable L/R tool scheduler.

    No synthetic teacher acts as executor. The independent harness and C
    approve the *whole task outcome* before any shared-reward update.
    This updates only the small coordination policy, NOT neural model weights.
    """
    import asyncio
    import json
    from hashlib import sha256
    from time import monotonic
    from .connected_training import _EVIDENCE
    from .brains import verifier

    if not user_permission or not getattr(mind, "connected_training_sandbox", False):
        raise PermissionError("Sandbox and explicit user authorization required")
    if not callable(approve_goal) or not approve_goal(goal):
        raise PermissionError("Goal not authorized by sandbox side-effect policy")
    if not callable(external_verify):
        raise ValueError("Independent result verifier required")
    if not goal.strip() or len(goal) > 2500:
        raise ValueError("Need a bounded goal")
    goal_agent = mind.goal_agent
    previous = goal_agent.coordinator, goal_agent.runner.coordinator
    coordinator = LearnedCoordinator(policy, goal_category=goal_category,
                                      explore=True)
    goal_agent.coordinator = coordinator
    goal_agent.runner.coordinator = coordinator
    started = monotonic()
    try:
        task = await goal_agent.create(persona_id, goal)
        await goal_agent.runner.wait(task.id)
        finished = await goal_agent.ledger.aget(persona_id, task.id)
        if finished is None:
            raise RuntimeError("Task lost its durable checkpoint")
        checkpoint = finished.checkpoint or {}
        items = checkpoint.get("items", [])
        summary = json.dumps({
            "status": finished.status,
            "subtask_states": [r.get("state", "unknown") for r in items],
            "batches": checkpoint.get("batches", 0),
            "verified_items": sum(r.get("state") == "verified" for r in items),
            "total_items": len(items),
        }, sort_keys=True)
        c = getattr(mind, "c", None)
        if c is None:
            c_state, c_level = "UNRATED", ""
        else:
            verdict = await verifier.verify(
                c, context=goal, response_text=summary,
                latest_request=goal, events=summary)
            c_state = str(getattr(verdict, "state", "UNRATED"))
            c_level = str(getattr(verdict, "level", ""))
        # Caller MUST independently inspect the real external end-state.
        # An executor receipt and a C verdict are not sufficient by themselves.
        proof = await external_verify(goal, finished)
        evidence_ref = proof.get("evidence_ref", "")
        independently_verified = (
            proof.get("independently_verified") is True
            and isinstance(evidence_ref, str)
            and _EVIDENCE.fullmatch(evidence_ref) is not None)
        external_success = (finished.status == "done"
                            and proof.get("success") is True)
        allowed = c_state == "READY" and c_level == "GREEN"
        if independently_verified and allowed and coordinator.choices:
            duration_ms = (monotonic() - started) * 1000
            # Correctness dominates speed; success on a late task is still
            # rewarded more than a fast but wrong/incomplete result.
            reward = (max(0.5, 1.0 - min(duration_ms / 120000, 0.3))
                      if external_success else 0.0)
            for choice, options in coordinator.choices:
                policy.learn(choice, options, reward=reward,
                             independently_verified=True,
                             evidence_ref=evidence_ref, sandbox=True)
        return {
            "goal_sha256": sha256(goal.encode("utf-8")).hexdigest(),
            "task_status": finished.status, "C_state": c_state,
            "C_level": c_level, "independently_verified": bool(independently_verified),
            "success": bool(external_success and independently_verified and allowed),
            "policy_decisions": len(coordinator.choices),
            "policy_updates": policy.updates,
            "neural_weights_updated": False,
            "raw_tool_data_saved": False,
        }
    finally:
        # Never leave exploratory policy attached to production work.
        goal_agent.coordinator, goal_agent.runner.coordinator = previous
