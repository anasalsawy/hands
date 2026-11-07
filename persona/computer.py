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
        args = arguments or {}
        bare = name.removeprefix("computer_")
        if bare in {"shell", "read_file", "write_file", "edit_file", "list_files", "search_files"}:
            return await asyncio.to_thread(self._file_tool, pid, bare, args)
        return json.dumps({"error": "GUI tools were replaced by Open Computer Use; connect the desktop app"})

    def _file_tool(self, pid: int, name: str, args: dict) -> str:
        try:
            if name == "shell":
                return self._shell(pid, str(args.get("command") or ""), int(args.get("timeout_s") or 30))
            if name == "read_file":
                return self._read(pid, args.get("path"), int(args.get("offset") or 0), int(args.get("limit") or 2000))
            if name == "write_file":
                return self._write(pid, str(args.get("path") or ""), str(args.get("content") or ""))
            if name == "edit_file":
                return self._edit(pid, str(args.get("path") or ""), str(args.get("old") or ""), str(args.get("new") or ""))
            if name == "list_files":
                return self._list(pid, args.get("path"))
            if name == "search_files":
                return self._search(pid, str(args.get("pattern") or ""), args.get("path"))
        except PermissionError as exc:
            return json.dumps({"error": str(exc)})
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "the command took too long"})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        return json.dumps({"error": f"unknown tool computer_{name}"})

    def _shell(self, pid: int, command: str, timeout_s: int) -> str:
        command = command.strip()
        if not command:
            return json.dumps({"error": "empty command"})
        timeout_s = max(1, min(timeout_s, 120))
        work = self.workspace(pid)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": work,
               "LANG": os.environ.get("LANG", "C.UTF-8"), "TMPDIR": work}
        proc = subprocess.run(command, shell=True, cwd=work, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
        return json.dumps({"exit_code": proc.returncode, "stdout": proc.stdout[-8000:],
                           "stderr": proc.stderr[-4000:]})

    def _read(self, pid: int, path, offset: int, limit: int) -> str:
        real = self.resolve(pid, str(path or ""))
        with open(real, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        chunk = lines[max(0, offset): max(0, offset) + max(1, min(limit, 4000))]
        return "".join(chunk)

    def _write(self, pid: int, path: str, content: str) -> str:
        real = self.resolve(pid, path)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"ok": True, "path": os.path.relpath(real, self.workspace(pid))})

    def _edit(self, pid: int, path: str, old: str, new: str) -> str:
        real = self.resolve(pid, path)
        with open(real, encoding="utf-8") as f:
            text = f.read()
        if old not in text:
            return json.dumps({"error": "that text is not in the file"})
        with open(real, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
        return json.dumps({"ok": True})

    def _list(self, pid: int, path) -> str:
        real = self.resolve(pid, None if not path else str(path))
        root = self.workspace(pid)
        names = []
        for dirpath, dirnames, filenames in os.walk(real):
            dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules"}][:40]
            for fn in filenames[:200]:
                names.append(os.path.relpath(os.path.join(dirpath, fn), root))
            if len(names) > 200:
                break
        return "\n".join(names)

    def _search(self, pid: int, pattern: str, path) -> str:
        if not pattern:
            return json.dumps({"error": "empty pattern"})
        real = self.resolve(pid, None if not path else str(path))
        root = self.workspace(pid)
        hits = []
        for dirpath, dirnames, filenames in os.walk(real):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                try:
                    with open(full, encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                hits.append(f"{os.path.relpath(full, root)}:{i}: {line.strip()[:200]}")
                                if len(hits) >= 40:
                                    return "\n".join(hits)
                except OSError:
                    continue
        return "\n".join(hits)

