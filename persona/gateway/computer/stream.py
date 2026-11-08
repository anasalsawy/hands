"""Continuous capture producer for one WebSocket computer card."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .capture import grab_jpeg

log = logging.getLogger(__name__)


@dataclass
class StreamSpec:
    kind: str = "desktop"  # desktop | window | region
    hwnd: int | None = None
    pid: int | None = None
    title: str | None = None
    x: int | None = None
    y: int | None = None
    w: int | None = None
    h: int | None = None
    fps: float = 8.0
    exclude_self: bool = True
    quality: int = 55

    def clamp(self) -> "StreamSpec":
        self.kind = (self.kind or "desktop").strip().lower()
        if self.kind not in {"desktop", "window", "region"}:
            self.kind = "desktop"
        self.fps = max(2.0, min(15.0, float(self.fps or 8.0)))
        self.quality = max(35, min(80, int(self.quality or 55)))
        return self


@dataclass
class ComputerStreamSession:
    """One live card: capture loop → callback with JPEG bytes."""

    spec: StreamSpec
    on_frame: object  # async callable(bytes, meta)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    frames: int = 0
    last_error: str | None = None

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="computer-stream")

    async def stop(self) -> None:
        self._stop.set()
        t = self._task
        if t is not None:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        spec = self.spec.clamp()
        interval = 1.0 / spec.fps
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                jpeg, meta = await asyncio.to_thread(
                    grab_jpeg,
                    kind=spec.kind,
                    hwnd=spec.hwnd,
                    pid=spec.pid,
                    title=spec.title,
                    x=spec.x,
                    y=spec.y,
                    w=spec.w,
                    h=spec.h,
                    exclude_self=spec.exclude_self,
                    quality=spec.quality,
                )
                self.frames += 1
                self.last_error = None
                meta["frame"] = self.frames
                await self.on_frame(jpeg, meta)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("computer stream frame failed: %s", self.last_error)
                # brief pause so a missing window doesn't spin the CPU
                await asyncio.sleep(0.75)
            elapsed = time.perf_counter() - t0
            delay = interval - elapsed
            if delay > 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
