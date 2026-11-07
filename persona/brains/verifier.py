"""Thin async bridge around the meter module.

The meter module is pure-sync; our turn loop is async. ``verify`` builds the
cycle prompt, calls B's LLM, retries once on an unusable reply, and runs the
gateway-side ``rate`` to produce a ``Meter``. Any failure returns UNRATED and
never a fake GREEN.
"""
from __future__ import annotations

import logging
from typing import Any

from ..llm import LLM
from .meter import B_SYSTEM, Basis, Meter, _REPAIR, build_cycle_prompt, parse_review, rate, unrated

log = logging.getLogger(__name__)


async def verify(llm: LLM, *, context: str, response_text: str, events: str = "", prior_state: str = "",
                 latest_request: str = "", sensed: str = "", auditor_text: str = "",
                 max_chars: int = 18000, basis: Basis | None = None, retries: int = 1,
                 on_health=None, events_share: float | None = None) -> Meter:
    """Run the meter. If ``on_health`` is a SubsystemHealth, report ok/fail
    for the subsystems dashboard. ``events_share`` widens the tool-result
    section when the reply is about what the tools brought back."""
    user = build_cycle_prompt(context, response_text, events, prior_state, max_chars,
                              latest_request, sensed, auditor_text, events_share=events_share)
    record = "\n".join([context, events, latest_request, sensed, auditor_text])
    basis = basis or Basis(tool_events=1 if events.strip() else 0,
                           sensed=bool(sensed.strip()),
                           auditor=bool(auditor_text.strip()))
    msg_user = user
    last_meter: Meter | None = None
    for attempt in range(retries + 1):
        try:
            res = await llm.chat(
                [{"role": "system", "content": B_SYSTEM},
                 {"role": "user", "content": msg_user}],
                temperature=0.0, max_tokens=2500, json_mode=True)
            raw = res.text or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("meter call failed: %s", exc)
            if on_health is not None:
                on_health.fail(f"call failed: {type(exc).__name__}: {exc}")
            return unrated(f"reviewer call failed: {type(exc).__name__}")
        if parse_review(raw) is not None:
            try:
                last_meter = rate(raw, response_text, record, basis)
                if on_health is not None:
                    on_health.ok(extra={"last_level": getattr(last_meter, "level", None),
                                         "last_score": getattr(last_meter, "score", None)})
                return last_meter
            except Exception as exc:  # noqa: BLE001
                log.warning("meter rate error: %s", exc)
                if on_health is not None:
                    on_health.fail(f"rate error: {type(exc).__name__}: {exc}")
                return unrated(f"meter error: {type(exc).__name__}")
        msg_user = user + _REPAIR
    if on_health is not None:
        on_health.fail("reviewer returned unusable output after retries")
    return unrated("reviewer returned unusable output")


def meter_dict(m: Meter) -> dict[str, Any]:
    """A JSON-safe projection of a Meter for the chat API."""
    d = m.to_dict()
    # Trim long reasons and keep the ergonomics friendly.
    return d
