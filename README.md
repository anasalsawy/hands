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

## Quick start

**Local mode** (Persona + engine in one process):

```bash
pip install -r persona/requirements.txt  # if present, or install starlette uvicorn httpx
python stitch.py local --root ~/Projects --deepinfra-key YOUR_KEY --port 8800
```

**Remote mode** (connect to an existing Persona server):

```bash
python stitch.py remote --persona-url https://your-persona --persona-key sp_... --root ~/Projects
```

## stitch.py

The entry point. Wires Persona, the engine, and the Link adapter together. See `python stitch.py --help`.

## persona/link.py

The Link adapter. Connects Persona to any external agent. Two strategies, chosen automatically from what the agent advertises. See the module docstring for the full API.

## sawii_hands.py

The UI-TARS execution engine. One process instance per R-lane.

It is configured with a UI-TARS / DeepInfra style preset: an operator URL and an API key. See `presets/ui-tars-deepinfra.yaml`. The key in that file is a placeholder. Put a real key in your own copy when you run it. Do not commit one.

The engine does computer and browser tool work. It is the hands, not the mind.

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

The UI-TARS preset for a DeepInfra-style OpenAI-compatible endpoint. It names the operator, the base URL, the model, and a placeholder API key (`YOUR_DEEPINFRA_API_KEY`). The original file lives in the sawii-persona repo at `presets/ui-tars-deepinfra.yaml`. This copy is here so the engine can be configured without hunting for it. Replace the placeholder locally when you run the engine. Do not put a real key in this repo.

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
