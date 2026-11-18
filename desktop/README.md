# Hands — desktop app

The native Windows app for the court deck. `hands/desktop` is its own package: it shares
no files, no build, and no identity with the companion app (`ui/webview`, `ai.sawii.companion`).
This one is **Hands**, `ai.sawii.hands`, and it is a real `.exe`, not a page served by the backend.

The window is a thin skin over `persona/court.py`. Every number on screen comes from an HTTP
or WebSocket call to a Persona backend. Nothing here reimplements the deck, the Link, or the
engine, and no API key lives in this directory.

## Layout

```
hands/desktop/
  index.html, vite.config.ts, tsconfig.json
  src/
    main.tsx, App.tsx, styles.css
    lib/connection.ts      backend URL + persona key (local webview storage only)
    lib/client.ts          typed client for every route the app touches
    hooks/useDeck.ts       live deck state (WS /v1/court-events + 5s REST refresh)
    components/
      DeckView.tsx         the courts grid, and deck-level controls
      CourtCard.tsx        one card: spawn / goal / steer / stop / remove / voice
      LiveSurface.tsx      the live desktop stream inside a card (WS /v1/computer/stream)
      MotorView.tsx        UI-TARS engine + Link + computer health
      ChatView.tsx         POST /v1/chat/completions (SSE), /v1/interrupt, /v1/timeline
      VoiceView.tsx        WS /v1/voice session, GET /v1/voice-status
      PhoneView.tsx        GET /v1/calls, POST /v1/calls/outbound, /end
      BrowserView.tsx      WS /v1/browser/stream
      ConnectView.tsx      first run: URL + persona key
  scripts/backend.mjs      starts `python stitch.py local` for you
  scripts/build.mjs        icons -> typecheck -> bundle -> exe, one step at a time
  scripts/make-icon.mjs    regenerates src/assets/app-icon.png
  scripts/env.mjs          toolchain + disk report; shows whether link.exe exists
  scripts/smoke.mjs        drives the deck end to end against a real backend
  scripts/routes.mjs       unauthenticated probe: 401 = mounted, 404 = absent
  scripts/probe.mjs        is a backend listening on a port
  src-tauri/               Tauri 2 shell (productName "Hands", identifier ai.sawii.hands)
```

## Run in development

Two terminals.

Backend — from `hands/`:

```powershell
python stitch.py local --root C:\Users\you\Projects --port 8800
```

It prints the persona key: `Created local persona id=1 key=sp_…`. Copy it. If a persona
already exists, pass one explicitly with `--persona-key` or set `SAWII_PERSONA_KEY`.

Front end — from `hands/desktop/`:

```powershell
npm install
npm run dev          # http://127.0.0.1:5181  (open in a browser, or run tauri dev)
npx tauri dev        # the native window, same dev server
```

First launch shows the connect screen. Enter `http://127.0.0.1:8800` and the key. Press
**Settings** later to change either. Both are kept in the webview's own storage.

`node scripts/backend.mjs --root C:\Users\you\Projects` is a shortcut for the backend command;
`node scripts/backend.mjs --remote https://host` just prints what to type into Settings.

## Build the exe

`npm run build` produces the existing UI in `dist/` with relative asset paths.
Tauri embeds that directory directly; the Python backend is still a separate
process. A normal browser must serve the bundle over HTTP (for example,
`npm run preview`), since ES modules cannot load from `file://` in Chrome.

```powershell
cd hands\desktop
npm install
node scripts/env.mjs     # reports node/npm/cargo/rustc/link.exe/cl.exe/disk
npx tauri build
```

Or step by step, which is safer on a machine under memory pressure:

```powershell
node scripts/build.mjs           # icons, typecheck, bundle, cargo fetch, exe
node scripts/build.mjs typecheck # or one step at a time
```

### What the build needs on Windows

`node scripts/env.mjs` prints this machine's state. The front end needs only Node. The native
step additionally needs the **MSVC C++ toolchain and Windows SDK**, because the
`x86_64-pc-windows-msvc` target links with `link.exe`. rustup does not supply it, and
`cargo --version` succeeding proves nothing about it — the failure looks like:

```
error: linker `link.exe` not found
note: the msvc targets depend on the msvc linker but `link.exe` was not found
```

Install it, then sign out and back in so the new PATH applies:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

That is roughly 4 GB of download and 6 GB on disk once installed, plus about 1.5 GB for the
release target directory. Check `node scripts/env.mjs` for free space before starting. A
`link.exe` from Build Tools 2022 plus the Windows 10/11 SDK is what `cl.exe`/`link.exe`
resolving means.

WebView2 is already present on Windows 11, so nothing further is needed to run the app.

If `src/assets/app-icon.png` is missing or you want a different icon:
`node scripts/make-icon.mjs` then `npx tauri icon src/assets/app-icon.png` (icons are
generated, not committed).

### Output

Once the linker is installed, the artifacts land in `src-tauri/target/release/`:

| file | what it is |
| --- | --- |
| `release/Hands.exe` | the runnable app |
| `release/bundle/nsis/Hands_0.1.0_x64-setup.exe` | installer |
| `release/bundle/msi/Hands_0.1.0_x64_en-US.msi` | MSI |

Nothing under `target/` is committed, so verify with:

```powershell
Get-ChildItem src-tauri\target\release -Filter Hands.exe -Recurse
Get-ChildItem src-tauri\target\release\bundle -Filter *.exe,*.msi -Recurse
```

## Verifying against a real backend

No browser or exe needed — this drives the same routes in the same order the UI does:

```powershell
# terminal 1, from hands\
python stitch.py local --port 8878 --no-voice --root C:\Users\you\Projects

# terminal 2, from hands\desktop\
node scripts/smoke.mjs --base http://127.0.0.1:8878 --key <the key stitch printed>
node scripts/routes.mjs http://127.0.0.1:8878    # unauthenticated: 401 mounted, 404 absent
node scripts/probe.mjs 8800 8878                 # is anything even listening
```

`smoke.mjs` proves auth is enforced, creates two cards, spawns both, asserts they landed on
**two distinct work lines**, steers one, cancels one, reads the event socket, deletes one.
Exit code 0 means the deck contract is intact.

### Getting a usable persona key

`stitch.py local --persona-key X` passes X to the **engine** for `/v1/hands` auth. It does not
create a persona, so X authenticates against nothing and every route answers `401 invalid
persona key`. The key the app needs is the one the backend registers in its own store:

- started without `--persona-key`: the log line `Created local persona id=… key=sp_…`
- started with `--persona-key`: that key must already exist in the store (an existing persona)

The admin route `POST /admin/personas` mints one directly, but needs `SAWII_ADMIN_TOKEN`.

## Configuration

| setting | where it lives | default |
| --- | --- | --- |
| Backend URL | connect screen / Settings, then webview storage | `http://127.0.0.1:8800` |
| Persona key | connect screen / Settings, then webview storage | none |
| `VITE_HANDS_API_BASE` | build-time default URL | empty |
| `VITE_HANDS_PERSONA_KEY` | build-time default key | empty — **never commit a real key** |
| `HANDS_PORT`, `HANDS_ROOT`, `HANDS_APPROVE`, `HANDS_DATA_DIR` | read by `scripts/backend.mjs` | 8800, empty, ask, ~/.hands |

Backend flags worth knowing: `--root` is the allow-list of folders the engine may touch
(repeat it, one per folder), `--approve` is `ask` | `writes` | `auto` and defaults to `ask`.
This app never relaxes either; it only calls the routes.

## What talks to what

| screen | routes |
| --- | --- |
| Courts | `GET/POST /v1/courts`, `GET/DELETE /v1/courts/{id}`, `POST …/spawn`, `…/goal`, `…/cancel`, `…/voice`, `WS /v1/court-events`, `WS /v1/computer/stream` |
| Motor | `GET /v1/hands`, `GET /v1/computer`, `GET /v1/link`, `GET /v1/computer/capture`, `GET /v1/computer/windows` |
| Chat | `POST /v1/chat/completions`, `POST /v1/interrupt`, `GET /v1/timeline` |
| Voice | `WS /v1/voice`, `GET /v1/voice-status` |
| Phone | `GET /v1/calls`, `POST /v1/calls/outbound`, `POST /v1/calls/{sid}/end` |
| Browser | `WS /v1/browser/stream`, `GET /v1/computer` |

The deck routes need `SAWII_COURTS=1` (the default) and the desktop stream needs
`SAWII_COMPUTER_STREAM=1` (also the default). If either is turned off, the relevant panel
says so instead of showing a picture that is not real.

## Known gaps

- Retargeting a card in place would need `deck.retarget`, which has no route yet. The
  **target** control on a card only changes what this window captures; the card keeps its
  stored target until it is recreated.
- Court cards show the lane id, the Link adapter and its strategy. The engine's OS pid is
  not in `Court.view()`, so it is not shown — `GET /v1/link` reports per-adapter lane counts
  instead.
- The browser pane renders frames; it does not browse. Driving the page belongs to the hub.
- Voice needs a microphone grant from the webview. If it is refused, the panel says so and
  the typed-turn path still works.
- Unsigned: the build is not code-signed, so SmartScreen will warn on first run.