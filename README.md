# Hands

Desktop app that gives Persona hands on your computer. Persona is the mind; this repo is the body.

## Architecture

```
                    ┌─────────────────────────┐
                    │      Persona (mind)      │
                    │  shared context, history │
                    │    L (voice) + R (work)  │
                    └────┬──────────┬──────────┘
                         │          │
              ┌──────────┘          └──────────┐
              ▼                                ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  Link adapter    │            │  Link adapter    │
    │ (hijack or spawn)│            │ (hijack or spawn)│
    └────────┬─────────┘            └────────┬─────────┘
             ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │ sawii_hands R1   │            │ sawii_hands R2   │
    │ (engine instance)│            │ (engine instance)│
    └──────────────────┘            └──────────────────┘
```

Persona stays the single mind: one shared context, one conversation history, one line of thought. It coordinates the lanes.

Each delegated R-lane gets its own engine instance. The **Link** adapter connects Persona to the engine (or any external agent) using one of two strategies:

1. **Hijack** — the agent supports delegation/sub-agents natively (Hermes `delegate_task`, Cursor `Task`, Claude Code `Agent`). Link uses that pathway to inject R-lane work.
2. **Spawn** — the agent has no delegation. Link launches a new `sawii_hands.py` process per R-lane, connected via WebSocket.

The UI, the voice orb, the phone dialer, and the sketchboards attach around that. Do not rebuild any component. Do not edit Persona mind logic.

## The courts (the element layer)

The product is built around **cards**, and each card hosts a **live desktop instance**. Many run at once — *the courts*. That layer is UI-agnostic: `persona/court.py` (`HandsDeck`) is the headless controller, and any front end binds to it over the same typed wire.

```
persona/court.py  ──  HandsDeck / Court       (no pixels; pure contract)
        │
        ├── REST   GET/POST /v1/courts
        │            GET/DELETE /v1/courts/{id}
        │            POST /v1/courts/{id}/spawn | goal | cancel | voice
        ├── SSE    GET /v1/courts/{id}/events       (this persona's deck)
        └── WS     /v1/court-events                 (one socket, every card)

ui/webview/src/lib/courts.ts      the client binding (REST + SSE + WS)
ui/webview/src/components/DesktopHands.tsx   the deck view (motor panel + cards)
```

A **Court** fuses the three pieces that already exist: the **Link** (`persona/link.py`), a **lane** (`persona/mind.py`), and a **surface** (`persona/gateway/computer/`, the live JPEG stream each card shows). `deck.spawn(...)` is the concurrency mechanism — every call starts its own R-lane and `sawii_hands.py`, so several courts run at once.

The deck is mounted by the real app (`persona/api/app.py`, `SAWII_COURTS=1`, default on) and needs no keys to list or create cards. In the webview, open the **courts** toggle in the header to add cards, Start an instance, send a goal, and watch each card's live desktop stream.

## Quick start

**Local mode** (Persona + engine in one process):

```bash
pip install -r persona/requirements.txt  # if present, or install starlette uvicorn httpx
python stitch.py local --root ~/Projects --deepinfra-key YOUR_KEY --port 8800
```

`--deepinfra-key` is **Persona's** key (the mind). It seeds `DEEPINFRA_API_KEY` for the
L/R lobes in `persona/llm.py`. The engine never sees it and needs no key of its own.

**Remote mode** (connect to an existing Persona server):

```bash
python stitch.py remote --persona-url https://your-persona --persona-key sp_... --root ~/Projects
```

## Live voice (reuse, do not rebuild)

Hands does not invent a new voice stack. It wires two existing pieces:

1. **Persona realtime voice** — `persona/voice/realtime.py` + WebSocket `/v1/voice` + mic UI in `persona/web/index.html` (`/app`). This is the primary live conversation path (L voice lane).
2. **voice-module sidecar** — the sandbox at `../voice-module` (DeepInfra Whisper STT, MiMo TTS, barge-in always on). `stitch.py` starts it on `:18766` when found. See `voice_sidecar.py`.

Test locally after `stitch.py local`:

- Open `http://127.0.0.1:8800/app` and use the mic button.
- `GET http://127.0.0.1:8800/v1/voice-status` — wiring + sidecar health.
- `GET http://127.0.0.1:18766/health` — sidecar (if started).

Also present inside Persona (not separate servers): `persona/brains/voice_a.py` (lobe-A speak), `persona/voice/duplex.py`, `calls.py`, `clone.py`.

## Verify (run these)

```bash
# 1. the court deck runs two instances at once
python scripts/court_selftest.py

# 2. the deck's HTTP/SSE/WS contract the front end binds to
python scripts/court_http_selftest.py

# 3. the real harness: run_test_turn + run_benchmark (offline, scripted lobes)
python scripts/run_checks.py

# 4. the front end builds (from ui/webview/)
npm run typecheck && npm run build
```

## stitch.py

The entry point. Wires Persona, the engine, and the Link adapter together. See `python stitch.py --help`.

## persona/link.py

The Link adapter. Connects Persona to any external agent. Two strategies, chosen automatically from what the agent advertises. See the module docstring for the full API.

## sawii_hands.py

The execution engine. One process instance per R-lane.

**It has no LLM and no provider key.** It is a pure tool executor: it connects to
Persona's `/v1/hands` WebSocket with the persona key, advertises its tools, and
then waits. R emits a tool call, the engine executes it, the result goes back to
R, and R decides the next action. For screen work R asks for `ocu_get_app_state`
(screenshot + state) and chooses the next click itself — **the VLM is R**.
`persona/link.py` spawns one engine per delegated lane with only `--url`, `--key`
(the persona key), `--root`, `--approve`, `--place` and `--lane`.

Tool families:

- `shell`, `read_file`, `write_file`, `edit_file`, `list_files`, `search_files` — always on.
- `ocu_*` (screen, mouse, keyboard) — with `--computer-use`, via `ocu_mcp.py`, a thin
  stdio MCP client for the `open-computer-use` npm server (`npm install -g open-computer-use`).
  The schemas are discovered from the server at connect time, so `ocu_list_apps` and
  `ocu_get_app_state` are whatever that build actually offers. If `ocu_mcp.py` or the
  server is missing the engine logs `computer use OFF - <reason>` and still serves the
  file/shell tools; it does not crash.

Nothing runs outside the `--root` allow-lists, `--approve ask` is the default,
shell always asks, and there is no delete tool.

`presets/ui-tars-deepinfra.yaml` is **not** engine configuration. It is an import file
for the standalone UI-TARS Desktop app, which embeds its own operator VLM. Nothing in
this repo reads it, and there is no `--engine-preset` flag.

## persona/

A copy of the sawii-persona source. Persona is the one shared mind. It holds shared context, shared conversation history, and the shared line of thought. It coordinates lanes.

When it delegates to an R-lane and the engine has no native sub-agent feature, a new instance of `sawii_hands.py` is spawned and connected to that lane.

Key files:

### mind.py

The mind. This is the piece that thinks and coordinates. Leave its logic as it is.

### inference_lanes.py

The R-lane book and delegation. This is where lines of work are tracked and handed off.

### timesense.py

Timestamps and durations. It measures time passed and time spent. It does not invent clocks.

### sketch_graph.py

Sketchboard data: boxes and arrows as data. A global project view and per-lane private boards are intended to be built on this. The boards themselves are not built yet.

### shared_timeline.py

The shared event log. Every lane reads the same ordered stream of conversation and tool events.

## presets/ui-tars-deepinfra.yaml

An import file for the **standalone UI-TARS Desktop** app (Settings → Import Preset Config). That app embeds its own operator VLM, so it names an operator, a base URL, a model and a placeholder API key (`YOUR_DEEPINFRA_API_KEY`). The original lives in the sawii-persona repo at `presets/ui-tars-deepinfra.yaml`; this copy is here so a UI-TARS Desktop operator can be configured without hunting for it.

It configures **nothing** in this repo. `sawii_hands.py` reads no preset and no API key (see the engine section above); there is no `--engine-preset` flag. Replace the placeholder in your own copy. Never commit a real key.

The key that *does* matter here is Persona's: `stitch.py local --deepinfra-key` seeds `DEEPINFRA_API_KEY` for the mind (the L/R lobes in `persona/llm.py`).

## Not in this folder

These pieces are part of the product and must be stitched in. They are not copied here. Do not rebuild them. Attach them around Persona and the per-lane engine instances.

### Voice orb

Vocode Web UI: https://github.com/vocodedev/vocode-web

Pulsating orb, waveform, transcript, and mic.

### Phone dialer

CoreUI React dialer: https://github.com/coreui/coreui-free-react-admin-template

### Desktop shell

Tauri wrapping a React UI: https://tauri.app

The shell is the window the other UI pieces sit in. It is not a new app to design from scratch.
