# Hands × TARS architectural audit (stage 0, read-only)

Repo `anasalsawy/hands` at `f288917`. Static audit only: no runtime traces yet, so nothing below is claimed as "verified running". Stage 1 produces the traces.

## 1. What is real TARS today

- `engine/ui-tars-desktop/` is a verbatim copy of ByteDance UI-TARS-desktop (`engine/PROVENANCE.md`): `multimodal/{agent-tars,omni-tars,gui-agent,tarko}` and `packages/ui-tars/{sdk,action-parser,operators/{nut-js,browser-operator,browserbase,adb}}`.
- `engine/persona-loop.ts` imports the real Omni-TARS `ComposableAgent` / `AgentPlugin` and `@tarko/agent` (`NativeToolCallEngine`). Model calls are replaced by an injected client that goes to Persona's R lobe over stdio (`persona/tars_loop.py`). This is the reasoning loop only.
- Everything TARS uses to ACT is not used: the only plugin composed is `PersonaHands`, which forwards every tool call to Python.

## 2. Execution path as it is (task → action)

Task → `persona/court.py` (Court/HandsDeck; splitter, lanes) → `persona/tars_loop.py::run_tars_loop` → `engine/persona-loop.ts` (real Omni-TARS loop) → `PersonaHands` tool proxy → Python `HandsHub` (`persona/hands.py`, `persona/link.py`) → runner `sawii_hands.py` or the lane browser → tool implementation. Verification: C/V and `SpeculativeEngine` in Python (Persona side).

## 3. Non-TARS execution implementations (the substitution to remove)

| Capability required by directive | Implemented today by | TARS component that should own it |
|---|---|---|
| GUI / native visual grounding, mouse+keyboard | `ocu_mcp.py` (Open Computer Use MCP, external npm), `winhand.py` (814 lines, custom Windows window hand) | `packages/ui-tars/operators/nut-js` + `action-parser` + `omni-tars/gui-agent` |
| DOM browser | `persona/gateway/browser/*` + vendored **browser-use** (patched, `vendor/browser-use`, `scripts/install_browser_sdk.py`, `BrowserUseDriver`/`BrowserUseNativeDriver` in `link.py`) | `operators/browser-operator` / agent-tars browser MCP tools |
| Hybrid DOM+visual | none (two separate stacks) | `omni-tars` composable plugins (gui + mcp) |
| Terminal | `sawii_hands.py` `shell` | agent-tars/omni-tars code-agent and commands MCP |
| Filesystem | `sawii_hands.py` `read_file/write_file/edit_file/list_files/search_files` | filesystem MCP in `omni-tars/mcp-agent` |
| Web search/fetch | `webtools.py` (Bing/DDG scrape) | agent-tars search tool |
| MCP discovery/execution | none in Hands | `omni-tars/mcp-agent` |
| Event stream / context engineering | custom (`ThoughtStream`, Court SSE) | `@tarko/agent` event stream (partly present through the loop) |

## 4. Other competing executors / loops

- `persona/link.py` (4051 lines): drivers for Hermes, Cursor, Claude Code, Cline, a generic hijack driver, `OpenAIBridgeDriver`, `BrowserUseDriver`, `SwarmMind`/`SwarmHand`. Each is an alternative execution path for R lanes, outside TARS.
- `README`: Link "Spawn" strategy launches a separate `sawii_hands.py` process per lane (non-TARS tool loop).
- `persona/` (about 19k lines in top-level modules, 242 files): a full copy of the Persona mind inside Hands. Not an executor; the directive says do not touch Persona mind logic. Left alone.

## 5. Flags and fallbacks that can route around TARS

`HANDS_COMPUTER_USE`, `HANDS_WINDOW_HAND`, `HANDS_LANE`, `SAWII_OCU_COMMAND`, `SAWII_NODE`, `SAWII_HANDS_AUTOFIRE`; `Dockerfile` swallows failure of the browser-use install (`|| echo ...`); `load_ocu()` degrades silently to "computer use disabled"; `stitch.py` spawns `sawii_hands.py`. `presets/ui-tars-deepinfra.yaml` is for the standalone UI-TARS app and is not read by Hands (documented in README).

## 6. Retained (not touched)

Splitter, collaborative Hands, shared reasoning, Verifier, Persona mind, Court UI (`desktop/`), speculation engine, voice, phone, sketchboards.

## 7. Proposed replacement plan (needs your approval at the marked points)

1. Runtime traces of the current path (evidence that `ComposableAgent` runs; evidence that tools do NOT go through TARS).
2. Compose the real omni-tars plugins in `persona-loop.ts`: `mcp-agent` (browser, filesystem, commands, search via MCP), `gui-agent` (nut-js operator + visual grounding), `code-agent`. Keep `PersonaHands` only for Persona-specific tools (plan tool, lane/approval/surface hooks). **Approval point A:** which Persona-specific tools stay.
3. Remove browser-use, `ocu_mcp.py`, `webtools.py`, and the shell/file tools in `sawii_hands.py` once their TARS replacements pass tests. **Approval point B:** whether the Link drivers for Hermes/Cursor/Claude Code/Cline/Swarm stay (they are other agents' loops, not Persona R lanes).
4. `winhand.py` (Windows window targeting) has no TARS equivalent: keep it as an operator under TARS or replace it. **Approval point C.**
5. Then: end-to-end tests for DOM, visual, hybrid, MCP, terminal, filesystem; multi-Hand test on the TARS foundation; mapping table and deviations log.

Not done and not claimed: any runtime evidence, any test, the research extensions (OpenAPI/GraphQL, WebMCP, UCP).

## 8. Architectural boundary (owner ruling, final)

Persona is the mind; TARS is the execution engine. Two layers, not competing architectures.
- **Persona owns:** reasoning, planning, task decomposition, collaborative Hands, verification, memory, decisions, plus the planning tool, approvals, lanes, shared state and live cards (all kept).
- **TARS owns:** computer interaction, browser (DOM, visual, hybrid), visual grounding, terminal, filesystem, MCP execution.
- **Removed from the Hands execution path:** Open Computer Use, browser-use, and any alternative agent loop. Hermes, Cursor, Claude Code and Cline drivers must not substitute for Persona's intelligence or TARS's engine.
- Missing platform capabilities are added as native TARS operators, never as a separate engine. No third architecture between Persona and TARS. No reduced TARS: the complete relevant execution stack, active in runtime.

## 9. Capability table: upstream vs new (static, from `engine/ui-tars-desktop`)

Columns still to fill per capability: where in Hands, Persona connection, runtime evidence, benchmark. All "Hands today" cells below are NOT yet TARS-backed unless stated.

| Capability | Upstream in TARS? (location) | Hands today | Status |
|---|---|---|---|
| Visual perception and grounding | yes: `packages/ui-tars/action-parser`, `multimodal/gui-agent/action-parser`, `omni-tars/gui-agent` | `ocu_mcp.py`/`winhand.py` | to integrate |
| Unified cross-platform action space | yes: `multimodal/gui-agent/*`, `packages/ui-tars/operators` (nut-js, adb, browserbase) | none | to integrate |
| DOM browser execution | yes: `packages/agent-infra/browser-use`, `mcp-servers/browser`, `operators/browser-operator` | patched browser-use (Python) | replace |
| Visual browser / hybrid DOM+visual | yes: `gui-agent/operator-aio` (`AIOHybridOperator`, `AIOBrowser`, `AIOComputer`) | two separate stacks | to integrate |
| Observation/feedback, reflective recovery | yes: `GUIAgent` loop, `ComposableAgent`, action-parser reflection | partial (loop only) | to integrate |
| MCP-native tools | yes: `omni-tars/mcp-agent`, agent-infra `mcp-servers/*` | none | to integrate |
| Terminal and filesystem | yes: agent-infra `mcp-servers/{commands,filesystem}`, `omni-tars/code-agent`, agent-tars `environments/local` | `sawii_hands.py` | replace |
| Local and remote operators | yes: nut-js (local), `operator-aio` (remote sandbox), adb | none/winhand | to integrate |
| Event stream and context engineering | yes: `@tarko/agent` (in use via `persona-loop.ts`) | active, but tools bypass it | extend |
| UI-TARS-2 hybrid GUI+terminal+FS | yes: `omni-tars/omni-agent` composition (architecture only; model weights not public) | only reasoning loop | to integrate |
| Accessibility-tree interaction | partial upstream: DOM tree in `agent-infra/browser-use`, browserbase | none | extend |
| API-first web execution (Beyond Browsing) | NOT upstream | none | NEW research extension |
| WebMCP discovery and execution | NOT upstream (no matches) | none | NEW |
| Authorized REST/OpenAPI/GraphQL | NOT upstream as an executor (mentions only) | none | NEW |
| Universal Commerce Protocol | NOT upstream (one stray mention) | none | NEW |
| Dynamic API/DOM/visual selection | NOT upstream | none | NEW (as a TARS operator/plugin, not a separate engine) |

Benchmarks: none exist yet for Hands. Upstream ships `multimodal/benchmark`; Stage 4 will run real task suites and report actual results, not simulated ones.

## 10. Outcome (implemented, tested, measured)

Persona is the mind; Agent TARS is the execution engine. Every execution path below runs through `engine/persona-loop.ts`, which boots the real Agent TARS core (`AgentTARS`, vendored in `engine/ui-tars-desktop`). Python talks to it through `persona/tars_engine.py` (persistent tool server + loop) and `persona/tars_loop.py` (Persona as the model of the TARS loop).

| Capability | Upstream implementation | Where in Hands | Connected to Persona by | Runtime evidence (test) | Benchmark (median) |
|---|---|---|---|---|---|
| DOM browser | agent-tars core browser MCP tools | `TarsSession` (`persona/gateway/browser/session.py`), Court lanes | hub `run()` / TARS loop with Persona `think` | `ModeTests.test_dom`, `HubTests` | 134 ms |
| Visual | `browser_vision_control` | same engine | same | `ModeTests.test_visual` | 1833 ms |
| Hybrid | visual + DOM on one page | same engine | same | `ModeTests.test_hybrid_*` | 1870 ms |
| Terminal / filesystem | agent-tars commands + filesystem MCP | `Computer.call` now forwards to TARS | `persona/computer.py` | `ModeTests.test_terminal_and_filesystem` | 4 ms / 2 ms |
| MCP | agent-tars `mcpServers` | `tars.mcp_servers` config | engine catalog | `ModeTests.test_mcp_server_tool` | <1 ms |
| Desktop GUI | `@ui-tars/operator-nut-js` + action-parser | `engine/desktop-tools.ts` (native TARS tools) | engine catalog | `DesktopTests` on real X11 (click + type change the app's pixels; empty-desktop click is a control) | 287 ms |
| Persona loop on TARS | `AgentTARS.run` with injected LLM client | `persona-loop.ts` `drive`, `run_tars_loop` | `think/approve/receipt/observe` messages | `LoopTests`, `PlanningTests`, `CourtTests`, `HubTests.test_tars_loop_runs_on_the_hubs_engine` | n/a |
| Multi-Hand collaboration | n/a (Persona) executed by TARS | `TabHands` + `tars_page_tools`, per-lane engines | shared mind | `CollaborationTests` (3 hands on one tab; 2 lanes concurrently) | n/a |
| API-first / REST | **new** (not upstream) | `engine/research-tools.ts` `api_request` | approval gate treats non-GET as mutating | `ResearchTests` | 2 ms |
| OpenAPI, GraphQL | **new** | `openapi_discover/call`, `graphql_query` | same | `ResearchTests` | 2 ms |
| UCP | **new** (spec: ucp.dev, `/.well-known/ucp`, REST transport, `UCP-Agent`) | `ucp_discover/call` | same | `ResearchTests.test_ucp_profile_and_checkout` | 2 ms |
| WebMCP | **new** (W3C WebML CG draft; `document.modelContext`, legacy `navigator.modelContext`) | shim in `research-tools.ts`; wraps Chrome's native API when present | same | `ResearchTests.test_webmcp_tools_run_in_the_page` | 121 ms |
| Accessibility tree | partial upstream; role/name interaction is **new** | `browser_accessibility_*` | same | `ResearchTests` | 238 ms |
| Surface selection | **new** | `web_plan` (ucp > webmcp > openapi > graphql > dom > visual) | model reads the ranking | `ResearchTests.test_web_plan_*` | 4 ms |

Full measurements: `docs/BENCHMARKS.md` (`python3 scripts/tars_bench.py`). Full test suite: `python3 -m pytest -q scripts/tars_loop_selftest.py`.

### Removed (competing executors)
`sawii_hands.py`, `ocu_mcp.py` (Open Computer Use), `webtools.py`, `winhand.py`, the Link drivers (Hermes / Cursor / Claude Code / Cline / generic / OpenAI bridge / browser-use) and `persona/link.py` (Persona's speculation classes moved verbatim to `persona/speculation.py`), the browser-use session and its patched SDK (`vendor/`), `/v1/hands` runner path, Python file/shell tools in `computer.py`.

### Deviations (documented, with reasons)
- **D1** provider label `volcengine`: TARS only enables visual/hybrid browser control for that provider name. It is a routing label; all inference goes to Persona's model.
- **D2** `@ui-tars/operator-nut-js` and `action-parser` are installed from the npm registry at the same upstream version (1.2.3) as the vendored tree, because the monorepo install needs a git tarball (electron/node-gyp) that this build environment cannot fetch. Same code, upstream's own published artifact.
- **D3** Desktop tools take UI-TARS actions on a 0-1000 grid, the same convention `browser_vision_control` uses.
- **D4** The browser tasks of a persona run on the persona's own hub engine (one browser shared by pane, tools and task), serialized by a per-engine browser lock, because TARS's DOM element map is global state and a concurrent pane screenshot broke it (found by test).
- **D5** Windows window targeting (winhand's PrintWindow capture/hwnd targeting) is not reproduced: desktop control is whole-screen via the nut-js operator. A per-window operator is the natural native extension; not built.
- **D6** Visual-mode benchmark is dominated by TARS's own settle wait, not model time (no model in these benchmarks).
- **D7** Not verified here: the Docker image build, the Windows bundle build, and remote (AIO sandbox) operators. The Dockerfile and packaging were changed to build the real TARS core; run them before release.
