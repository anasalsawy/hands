"""Her tool belt, and the tools she makes.

The cloud workspace and the browser are already hers. This adds the rest of
a self-sufficient belt (fetch a page, calculate, make a document the person
can download, see documents.py) and plugins. A plugin is a
small file. She can write one when asked, and a person can drop one in the
same folder. It runs in the cloud workspace, not inside the server process.

A visit does not get this belt. The caller brings its own tools.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

import httpx

from .documents import SCHEMA as DOCUMENT_SCHEMA, Documents

NAME = re.compile(r"^[a-z][a-z0-9_]{0,32}$")
RESERVED = {
    "make_plugin", "list_plugins", "remove_plugin", "web_fetch", "calculate",
    "check_clock", "search_memory", "delegate_task", "fetch_consults",
    "stop_background_work", "reply_to_question", "make_document",
}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


_STR = {"type": "string"}

BELT = [
    _schema("web_fetch", "Fetch a web page (HTTP or HTTPS) and return the text. Her cloud workspace.",
            {"url": _STR}, ["url"]),
    _schema("calculate", "Arithmetic. Numbers and + - * / abs round min max.",
            {"expression": _STR}, ["expression"]),
    _schema("make_plugin",
            "Create a tool she can call from now on. kind=http fetches a URL with {arg} placeholders. "
            "kind=shell is a Python script that reads a JSON object of the arguments from stdin and "
            "prints the result. It runs in the cloud workspace.",
            {"name": _STR, "description": _STR, "parameters": {"type": "object"},
             "kind": {"type": "string", "enum": ["http", "shell"]}, "body": _STR},
            ["name", "description", "kind", "body"]),
    _schema("list_plugins", "List the tools she has made.", {}, []),
    _schema("remove_plugin", "Remove a tool she made.", {"name": _STR}, ["name"]),
    DOCUMENT_SCHEMA,
]


def _calc(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression or len(expression) > 200:
        return json.dumps({"error": "empty or too long"})
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return json.dumps({"error": f"not arithmetic: {exc}"})
    allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    funcs = {"abs": abs, "round": round, "min": min, "max": max}

    def ok(node) -> bool:
        if isinstance(node, ast.Expression):
            return ok(node.body)
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_ops):
            return ok(node.left) and ok(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return ok(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            return all(ok(a) for a in node.args) and not node.keywords
        return False

    if not ok(tree):
        return json.dumps({"error": "only numbers and + - * / abs round min max"})
    try:
        value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, funcs)  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return str(value)


async def fetch_url(url: str, *, timeout: float = 15.0) -> str:
    url = (url or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return json.dumps({"error": "only http and https urls"})
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            })
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    text = response.text[:20000]
    return json.dumps({"status": response.status_code, "url": str(response.url), "text": text})


class Plugins:
    def __init__(self, data_dir: str, computer):
        self.root = Path(data_dir) / "plugins"
        self.computer = computer
        self.documents = Documents(data_dir)

    def _dir(self, pid: int) -> Path:
        path = self.root / str(int(pid))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, pid: int) -> list[dict]:
        found = []
        for path in sorted(self._dir(pid).glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and NAME.match(str(data.get("name") or "")):
                found.append(data)
        return found

    def schemas(self, pid: int) -> list[dict]:
        out = list(BELT)
        for plugin in self._load(pid):
            params = plugin.get("parameters") if isinstance(plugin.get("parameters"), dict) else {
                "type": "object", "properties": {}}
            out.append({"type": "function", "function": {
                "name": f"plugin_{plugin['name']}",
                "description": f"[plugin, cloud workspace] {plugin.get('description') or plugin['name']}",
                "parameters": params}})
        return out

    def names(self, pid: int) -> set[str]:
        return {t["function"]["name"] for t in self.schemas(pid)}

    async def call(self, pid: int, name: str, arguments: dict | None) -> str:
        args = arguments or {}
        if name == "web_fetch":
            return await fetch_url(str(args.get("url") or ""))
        if name == "calculate":
            return _calc(str(args.get("expression") or ""))
        if name == "make_document":
            return await asyncio.to_thread(self.documents.make, pid, args)
        if name == "list_plugins":
            return json.dumps([{"name": p["name"], "kind": p.get("kind"), "description": p.get("description", "")}
                               for p in self._load(pid)])
        if name == "remove_plugin":
            return self._remove(pid, str(args.get("name") or ""))
        if name == "make_plugin":
            return self._make(pid, args)
        if name.startswith("plugin_"):
            return await self._run(pid, name.removeprefix("plugin_"), args)
        return json.dumps({"error": f"unknown tool {name}"})

    def _make(self, pid: int, args: dict) -> str:
        name = str(args.get("name") or "").strip()
        kind = str(args.get("kind") or "").strip()
        body = str(args.get("body") or "")
        description = str(args.get("description") or "").strip()[:400]
        if not NAME.match(name) or name in RESERVED or name.startswith(("computer", "browser", "host", "plugin")):
            return json.dumps({"error": "name must be a short lowercase word and not one of her built-in tools"})
        if kind not in ("http", "shell"):
            return json.dumps({"error": "kind is http or shell"})
        if not description or not body.strip():
            return json.dumps({"error": "a plugin needs a description and a body"})
        if len(body) > 20000:
            return json.dumps({"error": "body is too long"})
        parameters = args.get("parameters") if isinstance(args.get("parameters"), dict) else {
            "type": "object", "properties": {}}
        if parameters.get("type") != "object":
            parameters = {"type": "object", "properties": parameters.get("properties") or {}}
        if kind == "http":
            parsed = urllib.parse.urlparse(body.strip())
            if parsed.scheme not in ("http", "https"):
                return json.dumps({"error": "an http plugin body is an http(s) url, with {arg} placeholders"})
        record = {"name": name, "description": description, "kind": kind, "body": body,
                  "parameters": parameters}
        path = self._dir(pid) / f"{name}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if kind == "shell":
            script = self.computer.resolve(pid, f"plugins/{name}.py")
            os.makedirs(os.path.dirname(script), exist_ok=True)
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(body if body.endswith("\n") else body + "\n")
        return json.dumps({"ok": True, "name": name, "tool": f"plugin_{name}",
                           "note": "the tool is available now"})

    def _remove(self, pid: int, name: str) -> str:
        if not NAME.match(name):
            return json.dumps({"error": "unknown plugin"})
        path = self._dir(pid) / f"{name}.json"
        if not path.exists():
            return json.dumps({"error": "no such plugin"})
        path.unlink()
        script = self.computer.workspace(pid) + f"/plugins/{name}.py"
        if os.path.isfile(script):
            os.remove(script)
        return json.dumps({"ok": True, "removed": name})

    async def _run(self, pid: int, name: str, args: dict) -> str:
        plugin = next((p for p in self._load(pid) if p["name"] == name), None)
        if plugin is None:
            return json.dumps({"error": f"no plugin {name}"})
        if plugin.get("kind") == "http":
            url = str(plugin.get("body") or "")
            for key, value in args.items():
                url = url.replace("{" + str(key) + "}", urllib.parse.quote(str(value), safe=""))
            if "{" in url or "}" in url:
                return json.dumps({"error": "the url still has an empty placeholder"})
            return await fetch_url(url)
        return await asyncio.to_thread(self._run_shell, pid, name, args)

    def _run_shell(self, pid: int, name: str, args: dict) -> str:
        script = self.computer.resolve(pid, f"plugins/{name}.py")
        if not os.path.isfile(script):
            return json.dumps({"error": "the plugin script is missing"})
        work = self.computer.workspace(pid)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": work,
               "LANG": os.environ.get("LANG", "C.UTF-8"), "TMPDIR": work}
        try:
            proc = __import__("subprocess").run(
                [sys.executable, script], input=json.dumps(args), cwd=work, env=env,
                capture_output=True, text=True, timeout=20)
        except __import__("subprocess").TimeoutExpired:
            return json.dumps({"error": "the plugin took too long"})
        out = (proc.stdout or "")[-8000:]
        err = (proc.stderr or "")[-2000:]
        if proc.returncode != 0:
            return json.dumps({"exit_code": proc.returncode, "stdout": out, "stderr": err})
        return out.strip() or json.dumps({"ok": True})
