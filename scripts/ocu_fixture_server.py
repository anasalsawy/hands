"""A stub Open Computer Use MCP server, for tests only.

Speaks the same JSON-RPC-over-stdio MCP dialect that ``ocu_mcp.py`` speaks to the
real ``open-computer-use`` npm server: ``initialize`` -> ``notifications/initialized``
-> ``tools/list`` -> ``tools/call``. Point ``SAWII_OCU_EXECUTABLE`` at a wrapper that
runs this file and the engine believes it has a real GUI.

Derived from the parent sawii-persona repo's contract fixture at
``tests/fixtures/ocu_server.py`` (same three tools, same "read state before acting"
rule), extended with ``list_apps``/``click``/``type_text``/``scroll``/``keypress``
so the ocu_* tool surface can be exercised end to end offline.

Run:  python scripts/ocu_fixture_server.py
"""
import json
import sys

TOOLS = {
    "list_apps": {"app": {"type": "string"}},
    "get_app_state": {"app": {"type": "string"}},
    "click": {"app": {"type": "string"}, "element_index": {"type": "string"}},
    "type_text": {"app": {"type": "string"}, "text": {"type": "string"}},
    "scroll": {"app": {"type": "string"}, "direction": {"type": "string"}},
    "keypress": {"app": {"type": "string"}, "key": {"type": "string"}},
}

seen_state: set[tuple[str, str]] = set()


def handle(method: str, params: dict):
    if method == "initialize":
        return {"protocolVersion": "2025-03-26",
                "serverInfo": {"name": "ocu-fixture", "version": "1"},
                "capabilities": {}}
    if method == "tools/list":
        return {"tools": [{"name": n, "description": f"fixture {n}",
                           "inputSchema": {"type": "object", "properties": props}}
                          for n, props in TOOLS.items()]}
    if method == "tools/call":
        name, args = params.get("name", ""), params.get("arguments") or {}
        app = str(args.get("app", ""))
        if name == "list_apps":
            return {"content": [{"type": "text", "text": "notepad.exe, calc.exe"}]}
        if name == "get_app_state":
            seen_state.add((name, app))
            return {"content": [{"type": "text", "text": "[7] Button Save"}]}
        # Every action must follow a fresh state read for that app - the same rule
        # the real server enforces, so the test proves R must look before it clicks.
        if ("get_app_state", app) not in seen_state:
            return {"isError": True,
                    "content": [{"type": "text", "text": "Read fresh app state before acting"}]}
        if name == "click":
            return {"content": [{"type": "text", "text": f"clicked {args.get('element_index')}"}]}
        if name == "type_text":
            return {"content": [{"type": "text", "text": f"typed {len(str(args.get('text', '')))} chars"}]}
        if name == "scroll":
            return {"content": [{"type": "text", "text": f"scrolled {args.get('direction')}"}]}
        if name == "keypress":
            return {"content": [{"type": "text", "text": f"pressed {args.get('key')}"}]}
        return {"isError": True, "content": [{"type": "text", "text": f"unknown tool {name}"}]}
    return {}


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except ValueError:
        continue
    if "id" not in req:
        continue  # notification
    result = handle(req.get("method", ""), req.get("params") or {})
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)