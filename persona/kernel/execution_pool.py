"""Persona-wide R execution lanes.

Default mode is elastic: an idle lane is reused, otherwise R(n+1) is
allocated immediately. Explicit capacity is retained for compatibility with
legacy bounded callers/tests. Each pool belongs to one persona; all lanes
share the owner's durable timeline, task ledger, and conflict coordinator.
Resource conflicts are enforced separately, not by slot exhaustion.
"""
from __future__ import annotations
import asyncio
from collections import deque
from contextlib import asynccontextmanager


class ExecutionPool:
    def __init__(self, capacity: int | None = None):
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._free = deque(f"R{i}" for i in range(1, capacity + 1)) if capacity else deque()
        self._waiters = deque()
        self._guard = asyncio.Lock()
        self._next = (capacity or 0) + 1
        self._active: set[str] = set()

    @asynccontextmanager
    async def reserve(self):
        loop = asyncio.get_running_loop()
        waiter = None
        async with self._guard:
            if self.capacity is None:
                slot = self._free.popleft() if self._free else f"R{self._next}"
                if slot == f"R{self._next}":
                    self._next += 1
                self._active.add(slot)
            else:
                waiter = loop.create_future()
                self._waiters.append(waiter)
                self._wake()
        if waiter is not None:
            try:
                slot = await waiter
            except BaseException:
                async with self._guard:
                    try:
                        self._waiters.remove(waiter)
                    except ValueError:
                        if waiter.done() and not waiter.cancelled():
                            self._free.append(waiter.result())
                    self._wake()
                raise
            async with self._guard:
                self._active.add(slot)
        try:
            yield slot
        finally:
            async with self._guard:
                self._active.discard(slot)
                self._free.append(slot)
                self._wake()

    def _wake(self):
        if self.capacity is None:
            return
        while self._free and self._waiters:
            future = self._waiters.popleft()
            if not future.done():
                future.set_result(self._free.popleft())

    @property
    def available(self):
        # Elastic mode can always create another execution lane.
        return len(self._free) if self.capacity is not None else None

    @property
    def active(self):
        return len(self._active)

    @property
    def created(self):
        return self._next - 1
