"""Her computer, inside the app.

The companion does not ask the person to start a bridge script. A home turn
gets a workspace on this machine (shell and files, confined to that folder)
and, when the browser is up, the screen the pane is already showing.

A visit does not get this. The caller brings its own tools.
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import subprocess
from pathlib import Path


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


_STR = {"type": "string"}
_INT = {"type": "integer"}

SCHEMAS = [
    _schema("computer_shell",
            "Run a shell command in her workspace inside the app. The working directory is that workspace.",
            {"command": _STR, "timeout_s": _INT}, ["command"]),
    _schema("computer_read_file", "Read a file in her workspace.",
            {"path": _STR, "offset": _INT, "limit": _INT}, ["path"]),
    _schema("computer_write_file", "Write a file in her workspace.",
            {"path": _STR, "content": _STR}, ["path", "content"]),
    _schema("computer_edit_file", "Replace one exact string in a workspace file.",
            {"path": _STR, "old": _STR, "new": _STR}, ["path", "old", "new"]),
    _schema("computer_list_files", "List files in her workspace.",
            {"path": _STR}, []),
    _schema("computer_search_files", "Search file contents in her workspace.",
            {"pattern": _STR, "path": _STR}, ["pattern"]),

]


class Computer:
    def __init__(self, data_dir: str):
        self.root = Path(data_dir) / "computers"

    def schemas(self) -> list[dict]:
        return list(SCHEMAS)

    def names(self) -> set[str]:
        return {t["function"]["name"] for t in SCHEMAS}

    def workspace(self, pid: int) -> str:
        path = self.root / str(int(pid))
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def status(self, pid: int) -> dict:
        return {"ready": True, "place": "in the app", "workspace": self.workspace(pid),
                "tools": sorted(self.names())}

    def resolve(self, pid: int, path: str | None) -> str:
        root = os.path.realpath(self.workspace(pid))
        raw = path or root
        if not os.path.isabs(raw):
            raw = os.path.join(root, raw)
        real = os.path.realpath(raw)
        if real != root and not real.startswith(root + os.sep):
            raise PermissionError(f"{path} is outside her workspace")
        return real

    async def call(self, pid: int, name: str, arguments: dict | None, *, browser_hub=None) -> str:
        """Her workspace tools are Agent TARS's own filesystem and command tools, run in the
        persona's TARS engine and confined by TARS to her workspace folder."""
        args = arguments or {}
        bare = name.removeprefix("computer_")
        if bare not in {"shell", "read_file", "write_file", "edit_file", "list_files", "search_files"}:
            return json.dumps({"error": f"unknown tool {name}"})
        if browser_hub is None or not hasattr(browser_hub, "engine"):
            return json.dumps({"error": "Agent TARS engine unavailable"})
        work = self.workspace(pid)
        try:
            engine = await browser_hub.engine(pid, workspace=work)
            if bare == "shell":
                return await engine.call("run_command", {"command": str(args.get("command") or ""), "cwd": work},
                                         timeout_s=max(5, min(int(args.get("timeout_s") or 30), 120)) + 10)
            path = str(args.get("path") or work)
            if bare == "read_file":
                return await engine.call("read_file", {"path": path})
            if bare == "write_file":
                parent = os.path.dirname(path if os.path.isabs(path) else os.path.join(work, path))
                if parent:
                    await engine.call("create_directory", {"path": parent})
                return await engine.call("write_file", {"path": path, "content": str(args.get("content") or "")})
            if bare == "edit_file":
                return await engine.call("edit_file", {"path": path, "edits": [
                    {"oldText": str(args.get("old") or ""), "newText": str(args.get("new") or "")}]})
            if bare == "list_files":
                return await engine.call("list_directory", {"path": path})
            return await engine.call("search_files", {"path": path, "pattern": str(args.get("pattern") or "")})
        except Exception as exc:  # noqa: BLE001 - a hand reports a structured failure
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    # The helpers below remain for status/resolve; execution is Agent TARS.
