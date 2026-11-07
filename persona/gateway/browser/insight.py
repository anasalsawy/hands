"""BrowserPaneInsight — an InsightCap(mode='L') that reads the page.

While L is waiting for a browser tool (click, type, navigate) to return, R
has idle time. R's job in that window is NOT to speak — L owns the mic — but
to look at what's on the page and prepare the next action.

This is the computer-use payoff from README.md §2: tool-wait as
think-time. Without R's insight pass, every tool call is a hard serial step.
With it, each wait window becomes productive prep.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import BrowserSessionProto

log = logging.getLogger(__name__)


class BrowserPaneInsight:
    """InsightCap implementation. Satisfies the Protocol in `kernel/caps.py`.

    Only responds in L mode — R is observing, not planning actions. For R
    mode we return "" so the kernel's R-insight channel defers to whatever
    the host's own InsightCap would have provided (psyche in companion app's
    case).
    """

    def __init__(self, session: "BrowserSessionProto"):
        self.session = session

    async def insight(self, *, mode: str, said: str, question: str) -> str:
        if mode != "L":
            return ""
        try:
            snap = await self.session.snapshot()
        except Exception as exc:  # noqa: BLE001
            log.warning("browser insight snapshot failed: %s", exc)
            return ""
        if not snap.url or snap.url == "about:blank":
            return ""
        # The silent writing is told what L just caused and what to pay
        # attention to next. We keep it short — this is a note, not a brief.
        parts = [
            f"page is {snap.title!r} at {snap.url}",
            f"visible elements: {len(snap.elements)}",
        ]
        # Highlight interactive elements likely to be the next action.
        buttons = [e for e in snap.elements if (e.get("tag") or "").lower() in ("button", "a")][:5]
        if buttons:
            parts.append("notable: " + ", ".join(
                f"[{e.get('index')}]{(e.get('text') or '').strip()[:30]}"
                for e in buttons
            ))
        return " · ".join(parts)
