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
