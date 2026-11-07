"""Fast-weight delta: per-turn learning via L/R divergence.

Model:
  - L (fast): base + fast_delta, runs with per-turn updates
  - R (slow): base only, anchored, conservative
  - Delta = L − R: the divergence IS the learning signal

Every turn:
  1. Measure how L diverged from R
  2. Apply that divergence as delta
  3. Accumulate in fast_weight
  4. A serves with base + fast_weight

No training jobs, no replay, no consolidation — continuous learning.
"""
from __future__ import annotations

import json
import logging

from ..store import jdump, jload, now

log = logging.getLogger(__name__)


class FastWeightDelta:
    """Accumulator for per-turn weight shifts from L/R divergence.

    Stores the running delta: how much has the persona shifted from its base?
    Updated per-turn based on (L output − R output).
    Applied when serving A.
    """

    def __init__(self):
        # Simple accumulation: {turn_id: {delta_vector}}
        # In practice: {turn_id: {"warmth": 0.3, "formality": -0.2, ...}}
        # Or just track raw divergences: [{"at": ts, "divergence": "..."}]
        self.state: dict = {"deltas": [], "turn_count": 0}

    def to_json(self) -> str:
        return jdump(self.state)

    @classmethod
    def from_json(cls, data: str) -> FastWeightDelta:
        obj = cls()
        try:
            obj.state = jload(data) or {"deltas": [], "turn_count": 0}
        except Exception:  # noqa: BLE001
            log.warning("fast_weight json parse failed; starting fresh")
        return obj

    def apply_divergence(self, divergence: dict) -> None:
        """Record L−R divergence as a weight update.

        divergence: {
          "at": timestamp,
          "turn_id": int,
          "l_output": "...",
          "r_output": "...",
          "measurement": "persona warmer by 0.3, less formal by 0.2"
        }
        """
        self.state.setdefault("deltas", []).append(divergence)
        self.state["turn_count"] = self.state.get("turn_count", 0) + 1

    def merge_ref(self, base_ref: str) -> str:
        """Return a reference for 'base + accumulated_delta'.

        In serving: apply all accumulated deltas to the base weights.
        """
        return jdump({
            "kind": "with_delta",
            "base": base_ref,
            "delta_turns": self.state.get("turn_count", 0),
            "deltas": self.state.get("deltas", [])[-10:]  # last 10 for compactness
        })


class LRDivergenceMeasurer:
    """Compare L (fast) and R (slow) outputs to extract the learning signal."""

    async def measure(self, l_output: str, r_output: str, l_reply: str, r_reply: str,
                     persona: dict) -> dict:
        """Measure how L diverged from R.

        In a real system, C would run a comparison:
        - "L was warmer, R was more measured"
        - "L took more risks, R was cautious"
        - Quantify: warmth delta = +0.3, caution delta = -0.2, etc.

        For MVP: just note that they're different.
        """
        # TODO: actual LLM-based measurement by C
        # For now: simple heuristic or placeholder
        differs = l_reply != r_reply
        return {
            "at": now(),
            "differs": differs,
            "l_len": len(l_reply),
            "r_len": len(r_reply),
            "l_snippet": l_reply[:100],
            "r_snippet": r_reply[:100],
            "measurement": "divergence detected" if differs else "synchronized"
        }


class EMASynchronizer:
    """2026-10-21 rehearsal. Not the mouth. See README.md

    EMA loop: R absorbs L, L decays toward base. Continuous learning without explosion.

    Every turn:
      1. R steps forward: R = 0.95*R + 0.05*L (5% maturity)
      2. L decays back: L_delta *= 0.95 (emotional cooling)

    Net effect:
      - R slowly consolidates L's lasting changes
      - L naturally cools off (fading affect)
      - Prevents L from accumulating forever
      - Persona grows stably without saturation
    """

    def __init__(self, alpha: float = 0.05):
        """alpha: fraction of L that R absorbs per turn. Default 0.05 = 5%."""
        self.alpha = alpha

    def sync_step(self, r_delta: FastWeightDelta, l_delta: FastWeightDelta) -> tuple[FastWeightDelta, FastWeightDelta]:
        """Execute one EMA sync: R absorbs from L, L decays.

        Returns: (updated_r, updated_l)
        """
        # R absorbs alpha% of L's accumulated changes (maturity)
        r_new = FastWeightDelta.from_json(r_delta.to_json())
        r_new.state["deltas"] = r_delta.state.get("deltas", []).copy()

        # Blend: take alpha% from L's state into R's state
        # For now: just add a record that R absorbed from L
        r_new.state.setdefault("absorptions", []).append({
            "at": now(),
            "alpha": self.alpha,
            "from_l_turns": l_delta.state.get("turn_count", 0)
        })
        r_new.state["turn_count"] = r_delta.state.get("turn_count", 0) + 1

        # L decays: multiply accumulated delta by (1 - alpha)
        l_new = FastWeightDelta.from_json(l_delta.to_json())
        l_new.state["deltas"] = l_delta.state.get("deltas", []).copy()
        l_new.state.setdefault("decay_events", []).append({
            "at": now(),
            "decay_rate": 1 - self.alpha,
            "emotional_cooling": "fading affect: L bleeds into R baseline"
        })
        l_new.state["turn_count"] = l_delta.state.get("turn_count", 0) + 1

        return r_new, l_new
