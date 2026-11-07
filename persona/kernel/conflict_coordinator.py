"""Conflict-aware action admission for independent R workers.

A shared resource can have concurrent readers or one writer. A superseded
proposal is never admitted. Already-running external tools are NOT
automatically cancelled; callers must reconcile their results before further
side effects. This is an in-process coordinator, not a distributed lock.
"""
from __future__ import annotations
import asyncio
from collections import defaultdict, Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass

@dataclass(frozen=True)
class ActionIntent:
    task_id: str
    resource: str
    operation: str  # read | write
    version: int

class SupersededAction(RuntimeError):
    pass

class ResourceConflictCoordinator:
    def __init__(self):
        self._condition=asyncio.Condition()
        self._readers=defaultdict(Counter)
        self._writers={}
        self._versions=defaultdict(int)

    async def supersede(self, task_id: str) -> int:
        async with self._condition:
            self._versions[task_id]+=1
            self._condition.notify_all()
            return self._versions[task_id]

    async def current_version(self, task_id: str) -> int:
        async with self._condition:
            return self._versions[task_id]

    @asynccontextmanager
    async def admit(self, intent: ActionIntent):
        if intent.operation not in ('read','write'):
            raise ValueError('operation must be read or write')
        if not intent.resource or not intent.task_id:
            raise ValueError('resource and task_id required')
        key=intent.resource
        async with self._condition:
            while True:
                if intent.version!=self._versions[intent.task_id]:
                    raise SupersededAction('proposal no longer matches user instructions')
                writer=self._writers.get(key)
                readers=self._readers[key]
                can_run=(not writer and (intent.operation=='read' or not readers))
                if can_run:
                    if intent.operation=='write':
                        self._writers[key]=intent.task_id
                    else:
                        readers[intent.task_id]+=1
                    break
                await self._condition.wait()
        try:
            # Check again at the side-effect boundary, not just at scheduling.
            if intent.version!=await self.current_version(intent.task_id):
                raise SupersededAction('instruction changed before execution')
            yield
        finally:
            async with self._condition:
                if intent.operation=='write':
                    if self._writers.get(key)==intent.task_id:
                        del self._writers[key]
                else:
                    self._readers[key][intent.task_id]-=1
                    if self._readers[key][intent.task_id]<=0:
                        del self._readers[key][intent.task_id]
                self._condition.notify_all()

    @asynccontextmanager
    async def try_admit(self, intent: ActionIntent):
        """Atomic, fail-fast claim. Never wait for another action to finish.

        This protects workers in this process. Distributed deployments require
        a shared transactional claim store before enabling cross-process writes.
        """
        if intent.operation not in ("read", "write"):
            raise ValueError("operation must be read or write")
        if not intent.resource or not intent.task_id:
            raise ValueError("resource and task_id required")
        async with self._condition:
            if intent.version != self._versions[intent.task_id]:
                raise SupersededAction("proposal no longer matches user instructions")
            readers = self._readers[intent.resource]
            writer = self._writers.get(intent.resource)
            granted = not writer and (intent.operation == "read" or not readers)
            if granted:
                if intent.operation == "read":
                    readers[intent.task_id] += 1
                else:
                    self._writers[intent.resource] = intent.task_id
        try:
            yield granted
        finally:
            if granted:
                async with self._condition:
                    if intent.operation == "read":
                        readers = self._readers[intent.resource]
                        readers[intent.task_id] -= 1
                        if readers[intent.task_id] <= 0:
                            del readers[intent.task_id]
                    elif self._writers.get(intent.resource) == intent.task_id:
                        del self._writers[intent.resource]
                    self._condition.notify_all()
