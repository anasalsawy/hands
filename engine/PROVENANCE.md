# Vendored engine: UI-TARS (unmodified)

`ui-tars-desktop/` in this folder is the **verbatim, unmodified** source of
ByteDance's UI-TARS Desktop, copied file-for-file. It is the real engine.
Nothing here is reimplemented, paraphrased, or "the same thing" — it is their code.

- Upstream: https://github.com/bytedance/UI-TARS-desktop
- Commit:   3ac2cb8d946d78d5ff92ad17b318eaafa18c178c  (main, release 0.3.1)
- License:  Apache-2.0 (see `ui-tars-desktop/LICENSE`)

## What is copied (all of it)
- `packages/ui-tars/sdk`            — `GUIAgent.ts` (the perceive -> act loop), `Model.ts`
- `packages/ui-tars/action-parser`  — `actionParser.ts` (Thought/Action parsing, coordinate grounding)
- `packages/ui-tars/operators/*`    — `nut-js` desktop operator, browser + adb operators
- `packages/ui-tars/shared`         — types, constants (SYSTEM_PROMPT), utils
- `packages/ui-tars/visualizer`     — the execution visualizer
- `apps/ui-tars`                    — the desktop app: `runAgent.ts` (model wiring),
                                      `ScreenMarker.ts` (blue water-flow overlay),
                                      `setOfMarks.ts` (red marker SVGs)
- everything else in the upstream tree, exactly as shipped

## What is NOT modified
No file under `ui-tars-desktop/` is edited, **except one additive seam** that
makes Persona a first-class provider (the change you asked for):

- `apps/ui-tars/src/main/store/types.ts` — adds one enum member `persona = 'Persona'` to `VLMProviderV2`.
- `apps/ui-tars/src/main/utils/agent.ts` — maps that member to `UITarsModelVersion.V1_0`.

Everything else is byte-for-byte upstream. These two edits only *add* a provider;
they do not alter the engine's loop, parser, operator, or overlays. To update,
re-copy from the commit above and re-apply these two additions.

## Active Hands loop (Omni-TARS / UI-TARS-2 architecture)

`persona-loop.ts` imports the real `ComposableAgent`, `AgentComposer`, and
`AgentPlugin` directly from `ui-tars-desktop/multimodal/omni-tars/core/src/`.
It does not import that package's index, which also exports unrelated parsers.
The supporting `@tarko/agent` runtime is pinned to the matching published 0.3.1
release in `package-lock.json`; its native tool-call engine uses Persona's
existing structured tool calls rather than a model-specific UI-TARS-1 parser.
The existing V1 provider enum additions above are not used by this path.

The adapter injects a custom client before running the upstream loop. Every
completion goes through `persona/tars_loop.py` into the existing R lobe, and
every tool goes to the explicit lane's `HandsHub`. The model label `persona-r`
and loopback port 1 in the adapter are inert: no provider transport is used.
Provider credentials and Node loader injection variables are removed from the
child environment. This is a local orchestration process, not a sandbox.

The Omni-TARS application in the upstream tree labels its endpoint UI-TARS-2
(`multimodal/omni-tars/omni-agent/tarko.config.ts`). This integration reuses that
application's composable loop architecture, **not** UI-TARS-2 model weights,
training, or private capabilities. Persona remains the only inference owner.

### Dependency caveat

The published 0.3.1 SDK has transitive audit advisories rooted in Zod v3. There
is no backward-compatible patched Zod v3 release reported by `npm audit` in
this environment. They are not suppressed; do not treat this dependency tree
as security-cleared. UUID is overridden to the compatible 11.1.1 release to
remove its bounds-check advisory. The adapter exposes no HTTP server and does
not accept remote clients; only Persona supplies its tool schemas and config.
