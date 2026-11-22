"""Agent TARS as a persistent tool server for Persona's mind.

Persona decides; TARS executes. ``TarsEngine`` boots the real Agent TARS core
(``engine/persona-loop.ts`` in ``mode: tools``) and runs its native tools on request:
DOM browser tools, ``browser_vision_control`` (visual / hybrid), filesystem, commands,
MCP servers, search. There is no second browser stack and no loop here - callers
(Persona's mind, the Court, the pane) pick tools, TARS runs them.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

from .tars_loop import ENGINE, SETUP, STARTUP_TIMEOUT, TARS_DIST

_ENV_KEEP = {"PATH", "HOME", "SYSTEMROOT", "WINDIR", "USERPROFILE", "APPDATA",
             "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR"}


class TarsEngine:
    def __init__(self, *, workspace: str | None = None, tars: dict | None = None,
                 instructions: str = "Persona tool server"):
        self.workspace = workspace
        self.tars = dict(tars or {})
        if workspace:
            self.tars.setdefault("workspace", workspace)
        self.instructions = instructions
        self.specs: list[dict] = []
        self.process: asyncio.subprocess.Process | None = None
        self._calls: dict[str, asyncio.Future] = {}
        self._n = 0
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.events: list[dict] = []
        self.inbox: asyncio.Queue = asyncio.Queue()  # loop traffic (think/approve/receipt/done...)
        self.looping = False

    async def start(self) -> "TarsEngine":
        node = os.getenv("SAWII_NODE") or shutil.which("node")
        if not node or not (ENGINE / "node_modules/tsx").exists() or not TARS_DIST.exists():
            raise RuntimeError(SETUP)
        env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEEP}
        self.process = await asyncio.create_subprocess_exec(
            node, "--import", "tsx", str(ENGINE / "persona-loop.ts"),
            cwd=ENGINE, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            limit=64 * 1024 * 1024)
        ready = await asyncio.wait_for(self.process.stdout.readline(), timeout=15)
        if not ready or json.loads(ready).get("type") != "ready":
            raise RuntimeError(SETUP)
        await self._send({"type": "start", "mode": "tools", "goal": "", "instructions": self.instructions,
                          "tools": [], "max_steps": 100, "tars": self.tars})
        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=STARTUP_TIMEOUT)
            if not line:
                raise RuntimeError("Agent TARS engine exited while booting")
            msg = json.loads(line)
            if msg.get("type") == "catalog":
                self.specs = msg.get("specs") or []
                break
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("error", "Agent TARS failed to boot"))
        self._reader = asyncio.create_task(self._read())
        return self

    async def _send(self, message: dict) -> None:
        self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
        await self.process.stdin.drain()

    async def _read(self) -> None:
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                msg = json.loads(line)
                if msg.get("type") == "result":
                    fut = self._calls.get(msg.get("id"))
                    if fut and not fut.done():
                        fut.set_result(msg.get("result", ""))
                else:
                    if msg.get("type") == "event":
                        self.events.append(msg)
                        del self.events[:-200]
                    if self.looping:
                        self.inbox.put_nowait(line)
        finally:
            for fut in self._calls.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("Agent TARS engine stopped"))
            self.inbox.put_nowait(b"")  # wakes a running loop: the engine is gone

    @property
    def tools(self) -> list[str]:
        return [s["name"] for s in self.specs]

    def schemas(self) -> list[dict]:
        """OpenAI-format tool schemas for TARS's native tools."""
        return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                   "parameters": s["parameters"]}} for s in self.specs]

    async def call(self, name: str, args: dict | None = None, timeout_s: float = 60.0) -> str:
        if self.process is None or self.process.returncode is not None:
            raise RuntimeError("Agent TARS engine is not running")
        self._n += 1
        cid = f"t-{self._n}"
        fut = asyncio.get_running_loop().create_future()
        self._calls[cid] = fut
        await self._send({"type": "call", "id": cid, "name": name, "args": args or {}})
        try:
            return await asyncio.wait_for(fut, timeout_s)
        finally:
            self._calls.pop(cid, None)

    async def close(self) -> None:
        proc, self.process = self.process, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=8)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        if self._reader:
            self._reader.cancel()
