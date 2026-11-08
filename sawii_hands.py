"""Sawii's hands on your computer - shell and files, like any agent app.

Run this on YOUR machine. It connects to your Sawii with your persona key and
lets her (only her, only in the companion app) run commands and read, write,
edit and search files inside the folders you allow. Nothing runs on the shared
server.

    pip install websockets
    python sawii_hands.py --url https://<your-sawii> --key sp_... --root ~/Projects --approve ask

--root      a folder she may work in (repeat for more). Nothing outside them.
--computer-use  also let her see the screen and use the mouse and keyboard
                (npm install -g open-computer-use; run its doctor command).
--approve   ask   : every shell command and every write asks you here first (default)
            writes: reads are automatic, writes and shell ask
            auto  : nothing asks (only for folders you fully trust)

There is no delete tool. A shell command could still delete things, which is
why shell asks by default.
"""
from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

MAX_OUT = 16000

TOOLS = [
    {"type": "function", "function": {
        "name": "shell", "description": "Run a shell command on the person's computer, inside an allowed folder. "
        "Returns exit code, stdout and stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string", "description": "folder to run in (absolute, or relative to the first allowed folder)"},
            "timeout_s": {"type": "integer", "description": "default 120, max 600"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file (optionally a line range).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a text file (or append).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["overwrite", "append"]}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace one exact piece of text in a file (must occur exactly once).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "List files and folders (depth 1-4).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "depth": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "search_files", "description": "Search file contents with a regular expression.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "glob": {"type": "string", "description": "e.g. *.py"}}, "required": ["pattern"]}}},
]

# GUI schemas are discovered from the installed Open Computer Use MCP server.
READS = {"read_file", "list_files", "search_files", "ocu_list_apps", "ocu_get_app_state"}

OCU_FILE = Path(__file__).with_name("ocu_mcp.py")


def load_ocu(path: str | None = None):
    """Load the Open Computer Use client that sits beside this script.

    Returns ``(instance_or_None, error_string)`` so a missing or broken
    ``ocu_mcp.py`` degrades to "computer use disabled" instead of crashing the
    engine at start-up. The engine still serves shell/file tools without it.
    """
    import importlib.util

    target = Path(path) if path else OCU_FILE
    if not target.is_file():
        return None, (f"{target.name} not found beside {Path(__file__).name}; "
                      f"no ocu_* screen/mouse/keyboard tools (restore it from "
                      f"the sawii-persona repo at scripts/ocu_mcp.py)")
    try:
        spec = importlib.util.spec_from_file_location("sawii_ocu_mcp", target)
        if spec is None or spec.loader is None:
            return None, f"cannot import {target} (no import spec for this file)"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.OcuMCP(), ""
    except Exception as exc:  # noqa: BLE001 - a broken OCU must not kill the engine
        return None, f"{target.name} failed to start ({type(exc).__name__}: {exc})"


class Hands:
    def __init__(self, roots: list[str], approve: str = "ask", ask=None, computer_use: bool = False):
        self.computer_use = computer_use
        self.ocu = None
        self.ocu_error = ""
        if computer_use:
            # Import beside this script, also when loaded by the test runner.
            self.ocu, self.ocu_error = load_ocu()
            if self.ocu is None:
                print(f"computer use OFF - {self.ocu_error}", file=sys.stderr)
        self.roots = [os.path.realpath(os.path.expanduser(r)) for r in roots]
        if not self.roots:
            raise SystemExit("give at least one --root folder")
        self.approve = approve
        self.ask = ask or self._ask_terminal

    # ---- safety -----------------------------------------------------------
    def resolve(self, path: str | None) -> str:
        p = os.path.expanduser(path or self.roots[0])
        if not os.path.isabs(p):
            p = os.path.join(self.roots[0], p)
        real = os.path.realpath(p)
        if not any(real == r or real.startswith(r + os.sep) for r in self.roots):
            raise PermissionError(f"{path} is outside the folders you allowed")
        return real

    def allowed(self, name: str, summary: str) -> bool:
        if self.approve == "auto" or (self.approve == "writes" and name in READS):
            return True
        return self.ask(f"Sawii wants to {summary}")

    @staticmethod
    def _ask_terminal(question: str) -> bool:
        try:
            return input(f"\n{question}\nAllow? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    # ---- tools ------------------------------------------------------------
    def run(self, name: str, a: dict, stop=None) -> str:
        if name.startswith('ocu_'):
            if not self.computer_use:
                return json.dumps({"error": "computer use is disabled (start with --computer-use)"})
            if self.ocu is None:
                return json.dumps({"error": f"computer use is unavailable: {self.ocu_error or 'ocu_mcp.py not loaded'}"})
            if not self.allowed(name, f'use Open Computer Use: {name}'):
                return json.dumps({'error':'the person declined'})
            try:
                return self.ocu.call(name[4:], a, stop=stop)
            except Exception as exc:
                return json.dumps({'error':str(exc)})
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return fn(**a)
        except PermissionError as exc:
            return json.dumps({"error": str(exc)})
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments: {exc}"})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def tools(self) -> list[dict]:
        # A degraded computer-use start (ocu_mcp.py missing/broken, or the
        # open-computer-use server refusing to boot) advertises only the file
        # and shell tools rather than crashing the connection.
        if not self.computer_use or self.ocu is None:
            return TOOLS
        try:
            return TOOLS + self.ocu.tools()
        except Exception as exc:  # noqa: BLE001 - no GUI tools is not a reason to die
            self.ocu_error = f"Open Computer Use unavailable ({type(exc).__name__}: {exc})"
            self.ocu = None
            print(f"computer use OFF - {self.ocu_error}", file=sys.stderr)
            return TOOLS

    def t_shell(self, command: str, cwd: str | None = None, timeout_s: int = 120) -> str:
        where = self.resolve(cwd)
        if not self.allowed("shell", f"run in {where}:\n  {command}"):
            return json.dumps({"error": "the person declined this command"})
        shell = ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["bash", "-lc", command]
        try:
            p = subprocess.run(shell, cwd=where, capture_output=True, text=True,
                               timeout=max(1, min(int(timeout_s or 120), 600)))
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"timed out after {timeout_s} s"})
        return json.dumps({"exit_code": p.returncode, "stdout": p.stdout[-MAX_OUT:], "stderr": p.stderr[-4000:]})

    def t_read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        real = self.resolve(path)
        if not self.allowed("read_file", f"read {real}"):
            return json.dumps({"error": "the person declined"})
        with open(real, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        part = lines[max(0, int(offset)):max(0, int(offset)) + max(1, int(limit))]
        return "".join(part)[:MAX_OUT * 4]

    def t_write_file(self, path: str, content: str, mode: str = "overwrite") -> str:
        real = self.resolve(path)
        if not self.allowed("write_file", f"{'append to' if mode == 'append' else 'write'} {real} ({len(content)} chars)"):
            return json.dumps({"error": "the person declined"})
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "a" if mode == "append" else "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"ok": True, "path": real, "chars": len(content)})

    def t_edit_file(self, path: str, old: str, new: str) -> str:
        real = self.resolve(path)
        with open(real, encoding="utf-8") as f:
            text = f.read()
        n = text.count(old)
        if n != 1:
            return json.dumps({"error": f"the text to replace occurs {n} times (must be exactly once)"})
        if not self.allowed("edit_file", f"edit {real}"):
            return json.dumps({"error": "the person declined"})
        with open(real, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
        return json.dumps({"ok": True, "path": real})

    def t_list_files(self, path: str | None = None, depth: int = 2) -> str:
        base = self.resolve(path)
        if not self.allowed("list_files", f"list {base}"):
            return json.dumps({"error": "the person declined"})
        depth = max(1, min(int(depth or 2), 4))
        out = []
        for here, dirs, files in os.walk(base):
            lvl = here[len(base):].count(os.sep)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__"))
            if lvl >= depth:
                dirs[:] = []
            rel = os.path.relpath(here, base)
            for f in sorted(files):
                out.append(os.path.join(rel, f) if rel != "." else f)
            if len(out) > 2000:
                break
        return "\n".join(out[:2000])

    def t_search_files(self, pattern: str, path: str | None = None, glob: str | None = None) -> str:
        base = self.resolve(path)
        if not self.allowed("search_files", f"search {base} for /{pattern}/"):
            return json.dumps({"error": "the person declined"})
        rx = re.compile(pattern)
        hits = []
        for here, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
            for f in files:
                if glob and not fnmatch.fnmatch(f, glob):
                    continue
                fp = os.path.join(here, f)
                try:
                    with open(fp, encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if rx.search(line):
                                hits.append(f"{os.path.relpath(fp, base)}:{i}: {line.rstrip()[:200]}")
                                if len(hits) >= 300:
                                    return "\n".join(hits)
                except OSError:
                    continue
        return "\n".join(hits) or "(no matches)"


def _host_label() -> str:
    """This machine's name for the ``hello`` frame, without ever blocking.

    ``platform.system()`` on Windows answers from a WMI query
    (``platform._wmi_query``), and a process spawned by a service does not
    always get one back: it can sit there for minutes. That is the real reason
    an engine Persona started never showed up on ``/v1/hands`` while the very
    same command worked when a person typed it in a terminal — the terminal's
    session has WMI warm, the spawned child's does not. The label is cosmetic,
    so it is taken from ``os.name``/the hostname, which cannot hang.
    """
    try:
        name = socket.gethostname() or "unknown"
    except OSError:
        name = "unknown"
    return f"{name} ({'Windows' if os.name == 'nt' else 'posix'})"


async def serve(url: str, key: str, hands: Hands, place: str = "computer", lane: str = "") -> None:
    import websockets  # pip install websockets
    ws_url = url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/v1/hands"
    backoff = 1
    while True:
        try:
            async with websockets.connect(ws_url, max_size=2 ** 24) as ws:
                await ws.send(json.dumps({"type": "hello", "key": key, "tools": hands.tools(), "place": place,
                                          "lane": lane[:32],
                                          "host": _host_label()}))
                ready = json.loads(await ws.recv())
                if ready.get("type") != "ready":
                    raise SystemExit(f"refused: {ready}")
                print(f"connected - {ready.get('persona')} can now use: {', '.join(ready.get('tools', []))}")
                lane_note = f"   lane: {lane}" if lane else ""
                print(f"folders: {', '.join(hands.roots)}   approve: {hands.approve}{lane_note}")
                backoff = 1
                pending = set()
                cancelled = {}
                import threading
                async def invoke(frame):
                    token = cancelled.setdefault(frame.get('id'), threading.Event())
                    out = await asyncio.to_thread(hands.run, frame.get('name',''), frame.get('arguments') or {}, token)
                    await ws.send(json.dumps({'type':'result','id':frame.get('id'),'ok':True,'output':out}))
                try:
                  async for raw in ws:
                    frame = json.loads(raw)
                    if frame.get('type') == 'cancel':
                        cancelled.setdefault(frame.get('id'), threading.Event()).set()
                        if hands.ocu:
                            await asyncio.to_thread(hands.ocu.close)
                    elif frame.get('type') == 'call':
                        task = asyncio.create_task(invoke(frame))
                        pending.add(task)
                        task.add_done_callback(pending.discard)
                finally:
                    for token in cancelled.values():
                        token.set()
                    if hands.ocu:
                        await asyncio.to_thread(hands.ocu.close)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"disconnected ({type(exc).__name__}); retrying in {backoff} s", file=sys.stderr)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every flag can also come from the environment, so the same program runs
    # headless as her cloud sandbox (a separate container: no app secrets, no DB).
    env = os.environ.get
    ap.add_argument("--url", default=env("SAWII_HANDS_URL"))
    ap.add_argument("--key", default=env("SAWII_HANDS_KEY"))
    ap.add_argument("--root", action="append", default=[r for r in (env("SAWII_HANDS_ROOT") or "").split(os.pathsep) if r])
    ap.add_argument("--approve", choices=["ask", "writes", "auto"], default=env("SAWII_HANDS_APPROVE") or "ask")
    ap.add_argument("--place", choices=["computer", "cloud"], default=env("SAWII_HANDS_PLACE") or "computer")
    ap.add_argument("--computer-use", action="store_true", default=env("SAWII_HANDS_COMPUTER_USE") == "1",
                    help="let her see the screen and use the mouse and keyboard (install open-computer-use)")
    ap.add_argument("--lane", default=env("SAWII_HANDS_LANE") or "",
                    help="optional execution lane id (for lane-bound workers)")
    a = ap.parse_args()
    if not a.url or not a.key:
        raise SystemExit("--url and --key are required (or SAWII_HANDS_URL / SAWII_HANDS_KEY)")
    if a.place == "cloud" and a.approve == "ask":
        a.approve = "auto"   # nobody is at a terminal in the sandbox; it is hers to work in
    asyncio.run(serve(a.url, a.key, Hands(a.root, a.approve, computer_use=a.computer_use), a.place,
                      lane=str(a.lane or "")))


if __name__ == "__main__":
    main()

