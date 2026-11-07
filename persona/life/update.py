"""Aging: the per-turn weight update.

This is the piece that makes the weights move on the life of the model rather
than on a calendar. C has just graded a turn. That grade is not only a record -
it is a *signal*, and here it is handed to whatever is running A's weights so
they can take one small step toward what C approved and away from what C
rejected.

Design notes, because the honest ones matter:

* **A must be running on weights we can touch.** A hosted, frozen endpoint
  cannot learn. So the update goes to a configurable endpoint
  (``SAWII_UPDATE_URL``) that a self-hosted server (vLLM + LoRA, or the
  transformers/PEFT wrapper) exposes. When that URL is unset the module is a
  no-op and says so - it never pretends a hosted model learned anything.

* **The signal is C's, not ours.** GREEN reinforces, YELLOW is a nudge away,
  RED is a clear penalty. We do not invent a reward the verifier didn't give.

* **The anchor gates it, exactly like it gates the night.** If C's rehearsal
  says the mind has drifted below the floor, we stop writing to the weights -
  a drifting verifier teaching every turn is how you get a fast, confident
  stranger.

* **One step per turn, and it is slow on purpose.** Aging is not fine-tuning.
  The server owns the learning rate; we only ever ask for one step, and we
  never batch a day's worth into the middle of a conversation.

* **Failures are loud.** A failed update is recorded and reported to the
  subsystems dashboard. The mind keeps talking - but you know the body is
  hurt.
"""
from __future__ import annotations

import logging
from typing import Any

from ..store import now

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised through the fake in tests
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import httpx2 as httpx  # type: ignore
    except ImportError:
        httpx = None  # type: ignore

# C's level -> the direction we push the weights. Magnitudes are deliberately
# small and symmetric: the server's learning rate does the real scaling.
REWARD = {"GREEN": 1.0, "YELLOW": -0.4, "RED": -1.0}


def reward_for(level: str | None) -> float | None:
    """Map a verdict level to a learning signal. None means 'don't teach'."""
    if not level:
        return None
    return REWARD.get(str(level).strip().upper())


async def strategy_feedback(ledger, persona_id: int, strategy: str, *,
                            verified: bool, success: bool, cost: float = 0) -> dict:
    """Learn coordination preference, never modify the served model weights.

    This signal is intentionally separate from Updater.update and its C-verdict,
    anchor, and trainable-backend gates. Unverified outcomes are not rewards.
    """
    if not verified:
        return {"status": "skipped", "reason": "unverified task evidence"}
    await ledger.arecord_outcome(persona_id, strategy, success=success, cost=cost)
    return {"status": "recorded", "strategy": strategy, "success": success}


class Updater:
    """Hands C's verdict to the thing that owns A's weights, once per turn."""

    def __init__(self, settings, store, health=None):
        self.s = settings
        self.store = store
        self.health = health

    @property
    def enabled(self) -> bool:
        return bool(self.s.update_url) and httpx is not None

    def _report(self, ok: bool, extra: dict | None = None, reason: str = "") -> None:
        if self.health is None:
            return
        try:
            h = self.health.get("aging")
            if ok:
                h.ok(extra)
            else:
                h.fail(reason or "aging step failed", extra)
        except Exception:  # noqa: BLE001 - health reporting must never break a turn
            pass

    async def update(self, persona: dict, *, turn_id: int | None, level: str | None,
                     prompt: str = "", response: str = "") -> dict[str, Any]:
        """One aging step, driven by C's verdict. Always returns a report."""
        pid = persona["id"]
        if not self.s.update_url:
            # Honest no-op: A is on weights we cannot touch.
            self._report(True, {"status": "no endpoint - weights frozen"})
            return {"status": "frozen", "reason": "SAWII_UPDATE_URL not set"}
        reward = reward_for(level)
        if reward is None:
            return {"status": "skipped", "reason": "no verdict this turn"}
        # The same gate the night uses. A drifting verifier must not be allowed
        # to rewrite the weights forty times a day. A persona that has never been
        # anchored (anchor_at == 0) has no evidence of drift either way, so it is
        # not frozen out of its own aging on day one.
        anchor = persona.get("anchor_score")
        anchored = bool(persona.get("anchor_at"))
        if anchored and anchor is not None and anchor < self.s.juice_floor:
            log.info("aging refused: anchor %.3f below floor %.2f", anchor, self.s.juice_floor)
            self._report(False, {"anchor": anchor}, reason="anchor below floor - refusing to write")
            return {"status": "refused", "reason": f"anchor {anchor} < floor {self.s.juice_floor}"}
        body = {
            "persona_id": pid,
            "turn_id": turn_id,
            "level": level,
            "reward": reward,
            "prompt": prompt[: self.s.update_max_chars],
            "response": response[: self.s.update_max_chars],
        }
        try:
            async with httpx.AsyncClient(timeout=self.s.update_timeout) as client:
                r = await client.post(self.s.update_url, json=body)
                r.raise_for_status()
                try:
                    detail = r.json()
                except Exception:  # noqa: BLE001
                    detail = {"raw": r.text[:200]}
        except Exception as exc:  # noqa: BLE001
            log.warning("aging step failed: %s", exc)
            self._report(False, None, reason=f"{type(exc).__name__}: {exc}"[:180])
            await self._log(pid, turn_id, level, reward, "failed", str(exc)[:400])
            return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:200]}
        await self._log(pid, turn_id, level, reward, "applied", str(detail)[:400])
        self._report(True, {"status": "applied", "level": level})
        return {"status": "applied", "level": level, "reward": reward, "detail": detail}

    async def _log(self, pid: int, turn_id, level, reward: float, status: str, detail: str) -> None:
        try:
            await self.store.insert("updates", persona_id=pid, turn_id=turn_id, level=level or "",
                                    reward=reward, status=status, detail=detail, created_at=now())
        except Exception as exc:  # noqa: BLE001 - the record is not worth losing a turn
            log.warning("could not record update: %s", exc)