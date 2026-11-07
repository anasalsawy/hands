# Hands Application Core Elements

This repository contains the main components you need to stitch together the new Hands application, capable of multi-lane, live computer-use, and agent/voice functionality.

## What's included

- **persona/**: The complete Persona mind source code (from sawii-persona). This is the core brain: shared context, lane management, reasoning, history, and all agent mind logic.
- **sawii_hands.py**: The UI-TARS engine/agent script. Each R-lane runs a copy to execute delegated tasks, browser/computer actions, and interface with tools/APIs.

## How to use

- **Persona** is run as a server/backend or imported as a module. It manages context, turns, and task routing.
- Each time Persona delegates to an R-lane, you spawn an instance of **sawii_hands.py** and supply it the needed lane context, endpoint, and API key.
- All context/interactions are tracked in Persona, and events/results are surfaced there.

## Stitching Notes

These elements are separated intentionally—no stitched product is provided yet. Stitching requires:
- Wiring Persona's lane delegation to spawn `sawii_hands.py` with appropriate arguments.
- Surfacing UI (voice, dialer, sketchboard, overlays) as components connected to Persona and engine states.
- Ensuring all context and turns are routed through Persona for shared memory.

## Element Descriptions

- **persona/**: Core agent mind, memory, logic.
- **sawii_hands.py**: Engine/worker script for execution, external tooling, computer/browser use.

Add any other services (Voice Orb UI, Dialer, Global/Per-lane Sketchboard) as needed to your shell (Tauri/Electron/React/etc) and wire their state to Persona and the appropriate lane or engine worker.

---

**These components are NOT a complete product until stitched together as described above.**