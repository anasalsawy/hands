"""Screenshot broadcasting — one producer, many WebSocket subscribers.

The browser pane in ui/webview subscribes to /v1/browser/stream and renders
PNG screenshots as they arrive. Multiple viewers (owner's phone + owner's
laptop, say) can watch the same session.

Rate: ~2 fps default. Snapshots are taken ONLY when the session is active
(i.e., L has an outstanding tool call or the broadcaster asks for one).
"""
from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .session import read_page

if TYPE_CHECKING:
    from .session import BrowserSessionProto

log = logging.getLogger(__name__)


@dataclass
class BrowserBroadcast:
    """Shared state between the browser session and its WebSocket subscribers."""

    session: "BrowserSessionProto"
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _last_frame_json: dict | None = None
    _pending: "asyncio.Task | None" = None
    _again: bool = False

    def schedule(self) -> None:
        """Refresh the pane in the background. The hand that just acted does
        not wait for a screenshot; refreshes asked for meanwhile become one."""
        if self._pending is not None and not self._pending.done():
            self._again = True
            return
        self._pending = asyncio.ensure_future(self._refresh())

    async def _refresh(self) -> None:
        while True:
            self._again = False
            await self.push_snapshot()
            if not self._again:
                return

    def attach(self) -> asyncio.Queue:
        """Register a new subscriber. Returns a queue that receives frames."""
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.subscribers.add(q)
        if self._last_frame_json is not None:
            try:
                q.put_nowait(self._last_frame_json)
            except asyncio.QueueFull:
                pass
        return q

    def detach(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    async def push_snapshot(self) -> None:
        """Snapshot the session once and broadcast to all subscribers."""
        async with self._lock:
            try:
                snap = await read_page(self.session, ground=False)
            except Exception as exc:  # noqa: BLE001
                log.warning("broadcast snapshot failed: %s", exc)
                return
            png_b64 = (
                base64.b64encode(snap.screenshot_png).decode("ascii")
                if snap.screenshot_png else ""
            )
            frame = {
                "type": "frame",
                "url": snap.url,
                "title": snap.title,
                "elements": snap.elements,
                "screenshot_png_b64": png_b64,
                "visible_text": snap.visible_text[:800],
            }
            self._last_frame_json = frame
            for q in list(self.subscribers):
                try:
                    q.put_nowait(frame)
                except asyncio.QueueFull:
                    # drop a frame for a slow viewer rather than block
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(frame)
                    except asyncio.QueueFull:
                        pass

    async def push_action(self, kind: str, detail: dict) -> None:
        """Broadcast a one-line "action" frame so the UI can show what L just did,
        even when the resulting screenshot hasn't been captured yet."""
        frame = {"type": "action", "kind": kind, **detail}
        for q in list(self.subscribers):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass
