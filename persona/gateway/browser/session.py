"""Browser session contract and explicit test doubles.

Production sessions are Agent TARS: ``TarsSession`` runs the real TARS browser tools
(DOM, markdown, screenshot, vision control) inside a persistent TARS engine. The dry-run
session is an explicit offline test double only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """What R observes between L's tool calls — the browser's visible state."""

    url: str
    title: str
    visible_text: str
    elements: list[dict]  # [{index: 0, tag: "button", text: "Submit", ...}, ...]
    screenshot_png: bytes | None = None


class BrowserSessionProto(Protocol):
    """What a browser session must offer. Both real and dry-run satisfy this."""

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def navigate(self, url: str) -> str: ...
    async def click(self, index: int) -> str: ...
    async def type_text(self, index: int, text: str) -> str: ...
    async def scroll(self, direction: str) -> str: ...
    async def go_back(self) -> str: ...
    async def wait_for_selector(self, selector: str, timeout_ms: int = 5000) -> str: ...
    # ``screenshot=False`` skips the image (a tool read never uses it).
    # ``ground=False`` reads without re-binding the indices the mind last saw
    # (the pane's own refreshes).
    async def snapshot(self, *, screenshot: bool = True, ground: bool = True) -> Snapshot: ...


async def read_page(session, *, screenshot: bool = True, ground: bool = True) -> Snapshot:
    """``session.snapshot`` with the options, for sessions that predate them."""
    import inspect
    try:
        takes = "ground" in inspect.signature(session.snapshot).parameters
    except (TypeError, ValueError):
        takes = False
    if takes:
        return await session.snapshot(screenshot=screenshot, ground=ground)
    return await session.snapshot()


# --------------------------------------------------------------------------
# Dry-run session — a Python-only stub.
#
# Used in tests and in dev when the browser isn't available. Records intent
# and returns stable fake responses so the Shape B pattern (L drives, R
# observes during the wait) can be exercised end-to-end without a browser.
# --------------------------------------------------------------------------
@dataclass
class DryRunSession:
    """A browser session that pretends. Useful for tests and dev builds."""

    actions: list[dict] = field(default_factory=list)
    current_url: str = "about:blank"
    current_title: str = "(nothing loaded)"
    _started: bool = False

    async def start(self) -> None:
        self._started = True

    async def close(self) -> None:
        self._started = False

    async def navigate(self, url: str) -> str:
        self.actions.append({"name": "navigate", "url": url})
        self.current_url = url
        self.current_title = f"[dry-run] {url}"
        return f"navigated to {url}"

    async def click(self, index: int) -> str:
        self.actions.append({"name": "click", "index": index})
        return f"clicked element {index}"

    async def type_text(self, index: int, text: str) -> str:
        self.actions.append({"name": "type", "index": index, "text": text})
        return f"typed into element {index}: {text!r}"

    async def scroll(self, direction: str) -> str:
        self.actions.append({"name": "scroll", "direction": direction})
        return f"scrolled {direction}"

    async def go_back(self) -> str:
        self.actions.append({"name": "go_back"})
        return "went back"

    async def wait_for_selector(self, selector: str, timeout_ms: int = 5000) -> str:
        self.actions.append({"name": "wait_for_selector", "selector": selector, "timeout_ms": timeout_ms})
        return f"waited for {selector}"

    async def snapshot(self, *, screenshot: bool = True, ground: bool = True) -> Snapshot:
        return Snapshot(
            url=self.current_url,
            title=self.current_title,
            visible_text="[dry-run] no page content",
            elements=[
                {"index": 0, "tag": "button", "text": "[dry-run button]"},
                {"index": 1, "tag": "a", "text": "[dry-run link]", "href": "about:blank"},
            ],
            screenshot_png=None,
        )


# --------------------------------------------------------------------------
# Agent TARS session: Persona's browser is TARS's browser.
# --------------------------------------------------------------------------
import asyncio
import base64
import json
import os
import re

_ELEMENT = re.compile(r"^\[(\d+)\]<(\w+)([^>]*)>(.*?)(?:</\w+>)?$")


def _text_of(raw) -> str:
    """TARS MCP tools return either JSON or a list of {type:text,text} parts."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return str(raw)
    if isinstance(data, list):
        return "\n".join(str(p.get("text", "")) for p in data if isinstance(p, dict) and p.get("type") == "text")
    if isinstance(data, dict):
        return json.dumps(data)
    return str(data)


def _failed(raw) -> bool:
    text = _text_of(raw).lstrip().lower()
    return text.startswith(("failed", "error", "browser not", "unknown tool", "denied")) or '"status":"error"' in text


class TarsSession:
    """One browser, owned by one TARS engine. Every method is a native TARS tool call."""

    def __init__(self, *, headless: bool = True, tars: dict | None = None, workspace: str | None = None,
                 engine=None):
        from ...tars_engine import TarsEngine
        cfg = {"headless": headless, "browser_control": "hybrid", **(tars or {})}
        chrome = os.getenv("SAWII_CHROME") or next((c for c in (
            "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser") if os.path.exists(c)), None)
        if chrome:
            cfg.setdefault("chrome_path", chrome)
        self.engine = engine or TarsEngine(workspace=workspace, tars=cfg)
        self.action_lock = asyncio.Lock()
        self._mapped = False  # TARS element indices exist only after browser_get_clickable_elements

    async def start(self) -> None:
        if self.engine.process is None:
            await self.engine.start()

    async def close(self) -> None:
        await self.engine.close()

    async def _tool(self, name: str, args: dict | None = None) -> str:
        raw = await self.engine.call(name, args or {})
        if _failed(raw):
            raise RuntimeError(_text_of(raw)[:300])
        return raw

    async def navigate(self, url: str) -> str:
        self._mapped = False
        await self._tool("browser_navigate", {"url": url})
        for _ in range(40):  # TARS maps the DOM only on a settled page
            if "complete" in _text_of(await self.engine.call("browser_evaluate", {"script": "document.readyState"})):
                break
            await asyncio.sleep(0.25)
        return f"navigated to {url}"

    async def _map(self) -> list[dict]:
        text = _text_of(await self.engine.call("browser_get_clickable_elements", {}))
        self._mapped = True
        elements: list[dict] = []
        for line in text.splitlines():
            m = _ELEMENT.match(line.strip())
            if m:
                elements.append({"index": int(m.group(1)), "tag": m.group(2),
                                 "text": re.sub(r"<[^>]+>", "", m.group(4)).strip()[:120],
                                 "attrs": m.group(3).strip()[:200]})
        return elements

    async def click(self, index: int) -> str:
        if not self._mapped:
            await self._map()
        await self._tool("browser_click", {"index": index})
        self._mapped = False
        return f"clicked element {index}"

    async def type_text(self, index: int, text: str) -> str:
        if not self._mapped:
            await self._map()
        await self._tool("browser_form_input_fill", {"index": index, "value": text})
        return f"typed into element {index}: {text!r}"

    async def scroll(self, direction: str) -> str:
        amount = {"down": 600, "up": -600, "top": -100000, "bottom": None}.get(direction, 600)
        await self._tool("browser_scroll", {} if amount is None else {"amount": amount})
        return f"scrolled {direction}"

    async def go_back(self) -> str:
        self._mapped = False
        await self._tool("browser_go_back")
        return "went back"

    async def wait_for_selector(self, selector: str, timeout_ms: int = 5000) -> str:
        probe = f"!!document.querySelector({json.dumps(selector)})"
        deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
        while True:
            if "true" in _text_of(await self._tool("browser_evaluate", {"script": probe})):
                return f"found {selector}"
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"timed out waiting for {selector}")
            await asyncio.sleep(0.25)

    async def frame(self) -> bytes | None:
        """The current viewport as PNG, taken by TARS's own screenshot tool."""
        try:
            info = json.loads(await self.engine.call("browser_screenshot", {}))
            path = info.get("filepath")
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    data = fh.read()
                os.unlink(path)
                return data
        except (ValueError, OSError, RuntimeError):
            pass
        return None

    async def snapshot(self, *, screenshot: bool = True, ground: bool = True) -> Snapshot:
        tabs = _text_of(await self.engine.call("browser_tab_list", {}))
        log.debug("tab_list %r", tabs)
        title = (re.search(r"Current Tab: \[\d+\] (.*)", tabs) or [None, ""])[1].strip()
        url = (re.search(r"\(([^)\n]*)\)?\s*$", tabs.splitlines()[-1]) or [None, ""])[1] if tabs else ""
        md = json.loads(await self.engine.call("browser_get_markdown", {}) or "{}")
        # Only a grounding read may rebuild TARS's element map: the pane must never move the
        # indices the mind is about to click.
        elements = await self._map() if ground else []
        png = await self.frame() if screenshot else None
        return Snapshot(url=url, title=title, visible_text=str(md.get("content", "")) if isinstance(md, dict) else "",
                        elements=elements, screenshot_png=png)


def create_browser_session(
    *,
    dry_run: bool = False,
    headless: bool = True,
    cdp_url: str | None = None,
    workspace: str | None = None,
) -> BrowserSessionProto:
    """Agent TARS's browser, unless a test asked for the dry-run stub."""
    if dry_run:
        return DryRunSession()
    return TarsSession(headless=headless, workspace=workspace,
                       tars={"cdp_endpoint": cdp_url} if cdp_url else None)
