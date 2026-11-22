"""Benchmark every Agent TARS execution mode through the real engine (no mocks).

  python3 scripts/tars_bench.py [runs]      writes docs/BENCHMARKS.md and prints a table

Each mode runs its own task `runs` times on a local fixture site; the same task (put 'tea' in
a todo list) is also done through every web surface (WebMCP, accessibility tree, DOM, visual,
REST, GraphQL) so the API-first claim is measured, not assumed.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from persona.tars_engine import TarsEngine  # noqa: E402
from web_fixture_server import serve  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BIG = ("data:text/html,<title>v0</title><button style='position:absolute;left:100px;top:100px;width:200px;height:100px' "
       "onclick=\"document.title='clicked'\">Big</button><input id=t style='position:absolute;left:100px;top:300px'>")


def ok_json(out, key="status", val="success"):
    try:
        return json.loads(out).get(key) == val
    except (ValueError, AttributeError):
        return False


async def main():
    server, site = serve()
    host = site.split("//")[1]
    work = tempfile.mkdtemp()
    e = await TarsEngine(workspace=work, tars={
        "api_hosts": {host: {"headers": {"authorization": "Bearer sekrit"}}},
        "ucp_agent_profile": "https://agent.example/p.json",
        "mcp_servers": {"fixture": {"command": sys.executable, "args": [str(Path(__file__).parent / "mcp_fixture_server.py")]}},
    }).start()
    call = e.call
    results: dict[str, dict] = {}

    async def bench(name, fn):
        times, good = [], 0
        for _ in range(RUNS):
            t0 = time.perf_counter()
            try:
                good += bool(await fn())
            except Exception:  # noqa: BLE001
                pass
            times.append((time.perf_counter() - t0) * 1000)
        results[name] = {"runs": RUNS, "success": good, "median_ms": round(statistics.median(times)),
                         "p95_ms": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)]),
                         "min_ms": round(min(times))}
        print(f"{name:34s} {good}/{RUNS}  median {results[name]['median_ms']:>5} ms  p95 {results[name]['p95_ms']:>5} ms", flush=True)

    async def title():
        t = json.loads(await call("browser_tab_list"))[0]["text"]
        return t.split("Current Tab: [0] ")[1].split("\n")[0]

    # ---- single modes
    async def dom():
        await call("browser_navigate", {"url": BIG})
        await call("browser_get_clickable_elements")
        await call("browser_click", {"index": 0})
        return await title() == "clicked"

    async def visual():
        await call("browser_navigate", {"url": BIG})
        out = await call("browser_vision_control", {"thought": "t", "step": "s", "action": "click(point='<point>200 150</point>')"})
        return '"success":true' in out and await title() == "clicked"

    async def hybrid():
        await call("browser_navigate", {"url": BIG})
        await call("browser_vision_control", {"thought": "t", "step": "s", "action": "click(point='<point>200 150</point>')"})
        await call("browser_get_clickable_elements")
        await call("browser_form_input_fill", {"index": 1, "value": "x"})
        return await title() == "clicked" and "x" in await call("browser_evaluate", {"script": "document.getElementById('t').value"})

    async def terminal():
        return "bench" in await call("run_command", {"command": "echo bench", "cwd": work})

    async def filesystem():
        path = os.path.join(work, "b.txt")
        await call("write_file", {"path": path, "content": "fs"})
        return "fs" in await call("read_file", {"path": path})

    async def mcp():
        return "HI" in await call("shout", {"text": "hi"})

    await bench("dom (navigate+map+click)", dom)
    await bench("visual (vision click)", visual)
    await bench("hybrid (visual then DOM)", hybrid)
    await bench("terminal (run_command)", terminal)
    await bench("filesystem (write+read)", filesystem)
    await bench("mcp (stdio server tool)", mcp)

    # ---- the same task through every web surface: add 'tea' to the todo list
    async def webmcp():
        await call("browser_navigate", {"url": site + "/webmcp"})
        await call("webmcp_list_tools")
        r = await call("webmcp_call_tool", {"name": "add-todo", "input": {"text": "tea"}})
        return ok_json(r) and "tea" in await call("browser_evaluate", {"script": "document.getElementById('list').innerText"})

    async def a11y():
        await call("browser_navigate", {"url": site + "/a11y"})
        await call("browser_accessibility_fill", {"role": "textbox", "name": "Full name", "value": "tea"})
        await call("browser_accessibility_click", {"role": "button", "name": "Save profile"})
        return "Saved tea" in await call("browser_evaluate", {"script": "document.title"})

    async def rest():
        r = json.loads(await call("api_request", {"method": "POST", "url": site + "/api/items", "body": {"name": "tea"}}))
        return r["status"] == 201

    async def graphql():
        r = json.loads(await call("graphql_query", {"endpoint": site + "/graphql", "query": "mutation($name:String){addItem(name:$name){id}}", "variables": {"name": "tea"}}))
        return r["status"] == "success"

    async def openapi():
        r = json.loads(await call("openapi_call", {"spec_url": site + "/openapi.json", "operation_id": "createItem", "body": {"name": "tea"}}))
        return r["status"] == 201

    async def ucp():
        r = json.loads(await call("ucp_call", {"origin": site, "method": "POST", "path": "/checkout-sessions", "body": {"line_items": [{"sku": "tea"}]}}))
        return r["status"] == 201

    async def plan():
        r = json.loads(await call("web_plan", {"url": site + "/webmcp", "goal": "add tea"}))
        return r["recommended"] == "ucp"

    for name, fn in [("surface: WebMCP tool", webmcp), ("surface: accessibility tree", a11y), ("surface: REST", rest),
                     ("surface: GraphQL", graphql), ("surface: OpenAPI operation", openapi), ("surface: UCP checkout", ucp),
                     ("planner (web_plan)", plan)]:
        await bench(name, fn)

    # ---- desktop (needs a display)
    if os.environ.get("DISPLAY") and "desktop_vision_control" in e.tools:
        async def desktop():
            out = await call("desktop_vision_control", {"action": "click(start_box='(900,900)')"})
            return ok_json(out)
        await bench("desktop (nut-js click)", desktop)

    await e.close()
    server.shutdown()
    lines = ["# Agent TARS execution benchmarks", "",
             f"Measured through the real engine (`persona/tars_engine.py` → Agent TARS core), {RUNS} runs per row, on a local fixture site "
             "(`scripts/web_fixture_server.py`). Headless Chrome, no model in the loop: this is tool-execution time, not reasoning time. "
             "Reproduce with `python3 scripts/tars_bench.py`.", "",
             "| mode | success | median ms | p95 ms | min ms |", "|---|---|---|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['success']}/{v['runs']} | {v['median_ms']} | {v['p95_ms']} | {v['min_ms']} |")
    (ROOT / "docs" / "BENCHMARKS.md").write_text("\n".join(lines) + "\n")
    (ROOT / "docs" / "benchmarks.json").write_text(json.dumps(results, indent=1))


asyncio.run(main())
