# Kept for reference; the harness (runner.py) now measures real turns via Mind metrics.
"""Latency measurement utilities for testing interface.

Provides context managers and timers to measure inference, EMA sync,
and database persistence times.
"""
from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Latency measurements for a single turn."""

    l_inference_ms: float = 0.0
    r_inference_ms: float = 0.0
    divergence_measurement_ms: float = 0.0
    ema_sync_ms: float = 0.0
    persist_ms: float = 0.0
    context_load_ms: float = 0.0
    character_anchor_check_ms: float = 0.0
    total_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "l_inference_ms": self.l_inference_ms,
            "r_inference_ms": self.r_inference_ms,
            "divergence_measurement_ms": self.divergence_measurement_ms,
            "ema_sync_ms": self.ema_sync_ms,
            "persist_ms": self.persist_ms,
            "context_load_ms": self.context_load_ms,
            "character_anchor_check_ms": self.character_anchor_check_ms,
            "total_ms": self.total_ms,
            "details": self.details,
        }


class Timer:
    """Simple wall-clock timer in milliseconds."""

    def __init__(self):
        self.start_time: float | None = None
        self.elapsed_ms: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop and return elapsed ms."""
        if self.start_time is None:
            log.warning("Timer.stop() called without start()")
            return 0.0
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return self.elapsed_ms

    def __enter__(self) -> Timer:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()


@contextlib.contextmanager
def measure(name: str, breakdown: LatencyBreakdown | None = None, field_name: str | None = None):
    """Context manager to measure a named operation.

    Args:
        name: Human-readable operation name
        breakdown: Optional LatencyBreakdown to accumulate into
        field_name: Optional field name in breakdown to store result (e.g., 'l_inference_ms')

    Usage:
        breakdown = LatencyBreakdown()
        with measure("L inference", breakdown, "l_inference_ms"):
            # ... do work
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        log.debug(f"{name}: {elapsed:.1f}ms")
        if breakdown and field_name:
            setattr(breakdown, field_name, elapsed)
