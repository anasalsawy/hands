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
