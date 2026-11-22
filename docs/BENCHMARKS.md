# Agent TARS execution benchmarks

Measured through the real engine (`persona/tars_engine.py` → Agent TARS core), 10 runs per row, on a local fixture site (`scripts/web_fixture_server.py`). Headless Chrome, no model in the loop: this is tool-execution time, not reasoning time. Reproduce with `python3 scripts/tars_bench.py`.

| mode | success | median ms | p95 ms | min ms |
|---|---|---|---|---|
| dom (navigate+map+click) | 10/10 | 134 | 149 | 115 |
| visual (vision click) | 10/10 | 1833 | 1845 | 1812 |
| hybrid (visual then DOM) | 10/10 | 1870 | 1886 | 1843 |
| terminal (run_command) | 10/10 | 4 | 4 | 3 |
| filesystem (write+read) | 10/10 | 2 | 19 | 1 |
| mcp (stdio server tool) | 10/10 | 0 | 1 | 0 |
| surface: WebMCP tool | 10/10 | 121 | 151 | 112 |
| surface: accessibility tree | 10/10 | 238 | 283 | 217 |
| surface: REST | 10/10 | 2 | 5 | 2 |
| surface: GraphQL | 10/10 | 2 | 3 | 1 |
| surface: OpenAPI operation | 10/10 | 2 | 2 | 2 |
| surface: UCP checkout | 10/10 | 2 | 3 | 2 |
| planner (web_plan) | 10/10 | 4 | 5 | 3 |
| desktop (nut-js click) | 10/10 | 287 | 324 | 277 |
