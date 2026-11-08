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

The Hands <-> engine seam (the "Link") lives in `hands/persona/link.py` and points
the engine's `model.baseURL` at Persona's R-lane endpoint — the engine's own loop is
used as-is.
