"""BrowserTools — a `ToolsCap` wrapping the browser session.

Shape B: the kernel calls `BrowserTools.run(name, arguments)` as part of its
tool-calling loop; the browser executes; the result is text the next beat can
fold into speech. The silent writing (R) uses the wait window to prepare the
next action — see insight.py for the InsightCap side.

Schema asymmetry (per README.md, `schemas(mode="L"|"R")`):
    - L (the speaking/driving writing) gets the action tools
    - R (the silent/observing writing) gets observation tools only

A host that doesn't separate the writings (`mode=""`) gets the union, since
that's the honest default.
"""
from __future__ import annotations

import logging
import json
from typing import TYPE_CHECKING

from .session import read_page

if TYPE_CHECKING:
    from .session import BrowserSessionProto

log = logging.getLogger(__name__)


# OpenAI function-tool schemas. These are the operations L is allowed to
# emit as function_call entries; the executor dispatches them to the session.
_ACTION_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Open a URL in the active tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute https URL to open."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": (
                "Click a visible interactive element by its index and optional text label. "
                "Providing text ensures correct re-grounding if elements shift."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "text": {"type": "string", "description": "Visible text on the element to re-ground against."},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text into an input element by its index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_back",
            "description": "Navigate back to the previous page.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_wait_for_selector",
            "description": "Wait for a CSS selector to appear (useful after a navigation or click).",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "timeout_ms": {"type": "integer", "default": 5000},
                },
                "required": ["selector"],
            },
        },
    },
]

_OBSERVATION_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "description": (
                "Read the current page: URL, title, visible text, and an indexed list of "
                "interactive elements. Returns JSON. Use this before any click or type."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class BrowserTools:
    """ToolsCap implementation. Satisfies the Protocol in `kernel/caps.py`."""

    def __init__(self, session: "BrowserSessionProto"):
        self.session = session

    def schemas(self, mode: str = "") -> list[dict]:
        """L gets action tools; R gets observation tools; "" gets both."""
        if mode == "L":
            return list(_ACTION_SCHEMAS) + list(_OBSERVATION_SCHEMAS)
        if mode == "R":
            return list(_OBSERVATION_SCHEMAS)
        return list(_ACTION_SCHEMAS) + list(_OBSERVATION_SCHEMAS)

    async def run(self, name: str, arguments: dict) -> str:
        """Dispatch a function call to the session. Returns text, not JSON
        for most tools — the kernel folds it directly into context. snapshot
        is the exception: it returns a compact JSON string so L can read the
        element indices.
        """
        try:
            match name:
                case "browser_navigate":
                    return await self.session.navigate(str(arguments.get("url", "")))
                case "browser_click":
                    idx = int(arguments.get("index", 0))
                    txt = str(arguments.get("text", "")).strip()
                    if txt and hasattr(self.session, "snapshot"):
                        try:
                            snap = await read_page(self.session, screenshot=False)
                            idx = ground_index(snap.elements, txt, idx)
                        except Exception:
                            pass
                    return await self.session.click(idx)
                case "browser_type":
                    return await self.session.type_text(
                        int(arguments.get("index", 0)),
                        str(arguments.get("text", "")),
                    )
                case "browser_scroll":
                    return await self.session.scroll(str(arguments.get("direction", "down")))
                case "browser_back":
                    return await self.session.go_back()
                case "browser_wait_for_selector":
                    return await self.session.wait_for_selector(
                        str(arguments.get("selector", "")),
                        int(arguments.get("timeout_ms", 5000)),
                    )
                case "browser_snapshot":
                    return _render_snapshot(await read_page(self.session, screenshot=False))
                case _:
                    return json.dumps({"error": f"unknown browser tool: {name}"})
        except Exception as exc:  # noqa: BLE001 — the hand reports a structured failure
            log.warning("browser tool %s failed: %s", name, exc)
            # DualLoop._failed and the independent goal verifier inspect
            # JSON error receipts. Plain 'browser tool failed' strings were
            # being interpreted as successful navigation/click results.
            return json.dumps({"error": f"browser tool {name} failed: {exc}"})


def _render_snapshot(snap) -> str:
    """Compact string representation L can read.

    JSON-like but hand-formatted so it stays small (we send it back as a
    tool_result and it counts against the model's context budget).
    """
    import json
    return json.dumps({
        "url": snap.url,
        "title": snap.title,
        "visible_text": snap.visible_text[:1500],
        "elements": [
            # Keep it tight: index + tag + short label + relevant attrs
            {k: v for k, v in e.items() if k in ("index", "tag", "text", "attrs")}
            for e in snap.elements[:60]
        ],
    }, ensure_ascii=False)
