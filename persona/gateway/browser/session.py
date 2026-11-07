"""Browser session contract and explicit test doubles.

Production sessions use the pinned edited Browser-Use SDK through edited.py.
The legacy session classes remain importable for compatibility with old callers;
create_browser_session never selects them for a real Companion browser.
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
# Real session — wraps browser_use.BrowserSession (Playwright-backed).
#
# Imported lazily. If browser-use isn't installed, create_browser_session()
# falls back to DryRunSession transparently.
# --------------------------------------------------------------------------
class RealSession:
    """Shape B wrap of browser_use's browser primitives.

    We intentionally do not use browser_use.Agent (that's their agent loop —
    Shape A). We use their BrowserSession / Browser primitives directly, so
    Sawii's kernel owns the outer loop and gets the kernel-wins (tool-wait
    prefetch, narrate-while-driving, silent error recovery).
    """

    def __init__(self, headless: bool = True, cdp_url: str | None = None):
        self.headless = headless
        self.cdp_url = cdp_url
        self._bu_session: Any = None  # browser_use.BrowserSession
        self._page: Any = None        # Playwright Page

    async def start(self) -> None:
        # Lazy import — don't require browser-use at module load time.
        try:
            from browser_use import BrowserSession  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "browser-use is not installed; "
                "install it with `pip install browser-use` or use DryRunSession for dev"
            ) from exc
        self._bu_session = BrowserSession(
            headless=self.headless,
            cdp_url=self.cdp_url,
        )
        await self._bu_session.start()
        self._page = await self._bu_session.get_current_page()

    async def close(self) -> None:
        if self._bu_session:
            try:
                await self._bu_session.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("browser session close failed: %s", exc)
            self._bu_session = None
            self._page = None

    async def navigate(self, url: str) -> str:
        if not self._page:
            raise RuntimeError("session not started")
        await self._page.goto(url)
        return f"navigated to {url}"

    async def click(self, index: int) -> str:
        if not self._bu_session:
            raise RuntimeError("session not started")
        # browser-use indexes visible interactive elements; it exposes a
        # click_element_by_index helper on the session controller.
        await self._bu_session.click_element_by_index(index)
        return f"clicked element {index}"

    async def type_text(self, index: int, text: str) -> str:
        if not self._bu_session:
            raise RuntimeError("session not started")
        await self._bu_session.input_text(index, text)
        return f"typed into element {index}"

    async def scroll(self, direction: str) -> str:
        if not self._page:
            raise RuntimeError("session not started")
        delta = -500 if direction == "up" else 500
        await self._page.evaluate(f"window.scrollBy(0, {delta})")
        return f"scrolled {direction}"

    async def go_back(self) -> str:
        if not self._page:
            raise RuntimeError("session not started")
        await self._page.go_back()
        return "went back"

    async def wait_for_selector(self, selector: str, timeout_ms: int = 5000) -> str:
        if not self._page:
            raise RuntimeError("session not started")
        try:
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            return f"found {selector}"
        except Exception as exc:  # noqa: BLE001
            # Tool capability turns this into a structured {error}; returning
            # a prose "timed out" string falsely looks like a successful wait.
            raise TimeoutError(f"timed out waiting for {selector}: {exc}") from exc

    async def snapshot(self, *, screenshot: bool = True, ground: bool = True) -> Snapshot:
        if not self._bu_session or not self._page:
            raise RuntimeError("session not started")
        if not ground:
            # Observation must not refresh Browser Use's action selector cache.
            data = await self._page.evaluate("""() => ({title: document.title,
                text: (document.body?.innerText || '').slice(0, 14000),
                elements: Array.from(document.querySelectorAll('a,button,input,textarea,select,[role=dialog],[role=alert]'))
                  .filter(e => e.getClientRects().length).slice(0,80).map((e,i) => ({
                    index:i,tag:e.tagName,text:(e.getAttribute('aria-label') || e.innerText || e.placeholder || '').slice(0,120)
                  }))})""")
            png = await self._page.screenshot(type="png") if screenshot else None
            return Snapshot(url=self._page.url, title=data.get("title", ""),
                            visible_text=data.get("text", ""), elements=data.get("elements", []),
                            screenshot_png=png)
        state = await self._bu_session.get_browser_state()
        shot = screenshot
        screenshot = None
        try:
            if shot:
                screenshot = await self._page.screenshot(type="png")
        except Exception as exc:  # noqa: BLE001
            log.warning("screenshot failed: %s", exc)
        elements = []
        try:
            # browser-use's state exposes an interactive-elements list
            for i, el in enumerate(getattr(state, "interactive_elements", []) or []):
                elements.append({
                    "index": i,
                    "tag": getattr(el, "tag_name", "?"),
                    "text": (getattr(el, "text", "") or "")[:120],
                    "attrs": {k: v for k, v in (getattr(el, "attributes", {}) or {}).items()
                              if k in ("href", "aria-label", "name", "placeholder")},
                })
        except Exception as exc:  # noqa: BLE001
            log.warning("element enumeration failed: %s", exc)
        return Snapshot(
            url=getattr(state, "url", "") or self._page.url,
            title=getattr(state, "title", "") or await self._page.title(),
            visible_text=(getattr(state, "visible_text", "") or "")[:4000],
            elements=elements,
            screenshot_png=screenshot,
        )


def _chrome_path() -> str | None:
    import os
    for path in ("/usr/local/bin/google-chrome", "/usr/bin/google-chrome",
                 "/usr/bin/chromium", "/usr/bin/chromium-browser"):
        if os.path.exists(path):
            return path
    return None


class PlaywrightSession:
    """The app's own Chrome with Skyvern-style live element identities."""

    def __init__(self):
        self._pw = None
        self._browser = None
        self._page = None
        self._element_ids: dict[int, str] = {}
        self._id_serial = 0

    async def start(self) -> None:
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        kwargs = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        # System Chrome when the machine has it. Otherwise Playwright's own
        # Chromium (what the image installs with WITH_CHROMIUM=1).
        chrome = _chrome_path()
        if chrome:
            kwargs["executable_path"] = chrome
        self._browser = await self._pw.chromium.launch(**kwargs)
        self._page = await self._browser.new_page(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._browser = self._page = self._pw = None
        self._element_ids = {}

    def _interactive(self):
        if self._page is None:
            raise RuntimeError("session not started")
        return self._page.locator("a, button, input, textarea, select, [role='button'], [role='link']")

    async def navigate(self, url: str) -> str:
        if self._page is None:
            raise RuntimeError("session not started")
        await self._page.goto(url, wait_until="domcontentloaded", timeout=20000)
        self._element_ids = {}
        try:
            # A short settle for script-drawn pages; never a wait for every ad.
            await self._page.wait_for_load_state("load", timeout=1500)
        except Exception:  # noqa: BLE001
            pass
        return f"navigated to {self._page.url}"

    async def _live_element(self, index: int):
        """Resolve a snapshot index through its stamped live DOM identity."""
        if self._page is None:
            raise RuntimeError("session not started")
        element_id = self._element_ids.get(int(index))
        if not element_id:
            raise RuntimeError("no live element for that index; take a fresh browser snapshot")
        return self._page.locator(f'[data-sawii-element-id="{element_id}"]')

    async def click(self, index: int) -> str:
        await (await self._live_element(index)).click(timeout=8000)
        return f"clicked element {index}"

    async def type_text(self, index: int, text: str) -> str:
        await (await self._live_element(index)).fill(text, timeout=8000)
        return f"typed into element {index}"

    async def scroll(self, direction: str) -> str:
        if self._page is None:
            raise RuntimeError("session not started")
        await self._page.mouse.wheel(0, -500 if direction == "up" else 500)
        return f"scrolled {direction}"

    async def go_back(self) -> str:
        if self._page is None:
            raise RuntimeError("session not started")
        await self._page.go_back(timeout=8000)
        return "went back"

    async def wait_for_selector(self, selector: str, timeout_ms: int = 5000) -> str:
        if self._page is None:
            raise RuntimeError("session not started")
        try:
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            return f"found {selector}"
        except Exception as exc:  # noqa: BLE001
            # A timeout is failure; BrowserTools converts exceptions to
            # structured JSON error receipts that the dual loop recognizes.
            raise TimeoutError(f"timed out waiting for {selector}: {exc}") from exc

    async def snapshot(self, *, screenshot: bool = True, ground: bool = True) -> Snapshot:
        if self._page is None:
            raise RuntimeError("session not started")
        # One round trip for the whole read. Per-element locator calls cost a
        # selector pass over the page each (160+ of them): 10-38 s live.
        data = await self._page.evaluate(_READ_PAGE, [_INTERACTIVE, 40, self._id_serial])
        self._id_serial = max(self._id_serial, int(data.get("serial") or 0))
        elements = []
        ids: dict[int, str] = {}
        for i, el in enumerate(data.get("elements") or []):
            ids[i] = str(el.get("id") or "")
            item = {"index": i, "id": ids[i], "tag": el.get("tag") or "?", "text": el.get("text") or ""}
            if el.get("href"):
                item["attrs"] = {"href": el["href"]}
            elements.append(item)
        if ground:
            self._element_ids = ids
        png = None
        if screenshot:
            try:
                png = await self._page.screenshot(type="png")
            except Exception as exc:  # noqa: BLE001
                log.warning("screenshot failed: %s", exc)
        return Snapshot(url=self._page.url, title=str(data.get("title") or ""),
                        visible_text=str(data.get("text") or ""), elements=elements, screenshot_png=png)


_INTERACTIVE = "a, button, input, textarea, select, [role='button'], [role='link']"
_READ_PAGE = """([selector, limit, serial]) => {
  const elements = [];
  for (const n of document.querySelectorAll(selector)) {
    if (elements.length >= limit) break;
    let id = n.getAttribute('data-sawii-element-id');
    if (!id) { serial += 1; id = 'sawii-' + serial; n.setAttribute('data-sawii-element-id', id); }
    const text = ((n.innerText || n.value || n.getAttribute('aria-label') || '') + '').trim().slice(0, 120);
    const href = n.tagName === 'A' && n.href && !n.href.startsWith('javascript:') ? n.href.slice(0, 200) : '';
    elements.push({id, tag: n.tagName.toLowerCase(), text, href});
  }
  const body = document.body ? document.body.innerText : '';
  return {elements, serial, title: document.title, text: (body || '').slice(0, 4000)};
}"""


def create_browser_session(
    *,
    dry_run: bool = False,
    headless: bool = True,
    cdp_url: str | None = None,
) -> BrowserSessionProto:
    """The app's Chrome, unless a test asked for the dry-run stub.

    ``headless`` is accepted for callers that still pass it. The app's Chrome
    is headless and the pane shows the screenshots.
    """
    if dry_run:
        return DryRunSession()
    from .edited import EditedBrowserSession
    return EditedBrowserSession(headless=headless, cdp_url=cdp_url)
