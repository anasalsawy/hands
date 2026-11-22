"""A minimal stdio MCP server (JSON-RPC) for tests: one tool, `shout`. No SDK needed."""
import json
import sys

TOOL = {"name": "shout", "description": "Upper-case the given text.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}

for line in sys.stdin:
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        result = {"protocolVersion": msg["params"].get("protocolVersion", "2024-11-05"),
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        text = str((msg["params"].get("arguments") or {}).get("text", "")).upper()
        result = {"content": [{"type": "text", "text": text}]}
    elif mid is None:
        continue
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\n")
    sys.stdout.flush()
