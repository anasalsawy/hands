# Hands

Hands is the execution side of Sawii Persona. **Persona is the mind** (reasoning, planning, splitting a goal into lanes, verification, memory, approvals, the live cards). **Agent TARS is the execution engine** (browser DOM/visual/hybrid, desktop GUI, terminal, filesystem, MCP, plus the API-first extensions in `engine/research-tools.ts`).

```
Persona (persona/)  --think/approve/receipt-->  engine/persona-loop.ts  -->  Agent TARS core  -->  Chrome / screen / shell / files / MCP / APIs
```

- `persona/tars_engine.py` persistent TARS tool server; `persona/tars_loop.py` Persona as the model of the TARS loop.
- `persona/court.py` the deck of live cards; each card is a lane running on its own engine.
- `persona/gateway/browser/` Persona's browser hub on TARS (`TarsSession`).
- `persona/speculation.py` Persona's speculative next-action planning.
- Architecture, upstream-vs-new table, deviations: `docs/ARCHITECTURE_AUDIT.md`. Measurements: `docs/BENCHMARKS.md`.

## Run

```
npm ci --prefix engine && npm run build:tars --prefix engine   # once: builds the vendored Agent TARS core
python3 hands_app.py
```

## Test

```
python3 -m pytest -q scripts/tars_loop_selftest.py   # real Chrome, real X11 (Xvfb), real TARS tools
python3 scripts/tars_bench.py                         # writes docs/BENCHMARKS.md
```
