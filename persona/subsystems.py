"""Subsystem health registry.

Every lobe/subsystem that can quietly fail - embedder, verifier, memory,
psyche, lane - should write to this registry on each success and failure.
``/health/subsystems`` reads it to tell you at a glance which parts of
Maya are alive, which are degrading, which are dead.

Keep it opinionated and small: no histograms, no deep metrics. Just the
last-success timestamp, last-error details, counts in the current and the
previous rolling-hour bucket. If you can't tell "is this subsystem dead"
from these fields, add a field.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SubsystemHealth:
    name: str
    # When the subsystem last did its job cleanly.
    last_success_at: float | None = None
    last_success_wall: float | None = None   # wall-clock seconds (time.time)
    # When it last failed or degraded; last_error_text keeps the reason.
    last_error_at: float | None = None
    last_error_wall: float | None = None
    last_error_text: str = ""
    # Running totals; cheap to maintain, enough for a glance.
    success_count: int = 0
    error_count: int = 0
    # Subsystem-specific state (e.g. embedder's model name, lane's active jobs).
    extra: dict = field(default_factory=dict)

    def ok(self, extra: dict | None = None) -> None:
        self.last_success_at = time.monotonic()
        self.last_success_wall = time.time()
        self.success_count += 1
        if extra:
            self.extra.update(extra)

    def fail(self, reason: str, extra: dict | None = None) -> None:
        self.last_error_at = time.monotonic()
        self.last_error_wall = time.time()
        self.last_error_text = (reason or "")[:300]
        self.error_count += 1
        if extra:
            self.extra.update(extra)

    def snapshot(self) -> dict:
        now_mono = time.monotonic()
        total = self.success_count + self.error_count
        rate = self.success_count / total if total else None
        # A subsystem that has never been exercised is *idle*, not dead. Reporting
        # idle subsystems as dead made /health/subsystems cry wolf on every fresh
        # boot (all six "dead" with zero calls) and hid real failures in the noise.
        idle = total == 0
        alive = self.last_success_at is not None and (
            now_mono - self.last_success_at < 24 * 3600)
        degraded = (self.last_error_at is not None and self.last_success_at is not None
                    and self.last_error_at > self.last_success_at)
        status = "idle" if idle else ("degraded" if degraded else ("alive" if alive else "dead"))
        return {
            "name": self.name,
            "alive": alive,
            "degraded": degraded,
            "idle": idle,
            "status": status,
            "last_success_at": self.last_success_wall,
            "last_error_at": self.last_error_wall,
            "age_since_success_s": (now_mono - self.last_success_at) if self.last_success_at else None,
            "age_since_error_s": (now_mono - self.last_error_at) if self.last_error_at else None,
            "last_error_text": self.last_error_text,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": rate,
            "extra": self.extra,
        }


class Registry:
    """A per-Mind registry. One SubsystemHealth per subsystem name."""

    def __init__(self):
        self._subs: dict[str, SubsystemHealth] = {}

    def get(self, name: str) -> SubsystemHealth:
        if name not in self._subs:
            self._subs[name] = SubsystemHealth(name=name)
        return self._subs[name]

    def all(self) -> dict[str, dict]:
        return {name: sub.snapshot() for name, sub in self._subs.items()}

    def summary(self) -> dict:
        """Overall health: ok if every known subsystem is alive and not degraded."""
        snaps = [sub.snapshot() for sub in self._subs.values()]
        dead = [s["name"] for s in snaps if s["status"] == "dead"]
        degraded = [s["name"] for s in snaps if s["status"] == "degraded"]
        idle = [s["name"] for s in snaps if s["status"] == "idle"]
        ok = not dead and not degraded
        return {"ok": ok, "dead": dead, "degraded": degraded, "idle": idle, "count": len(snaps)}
