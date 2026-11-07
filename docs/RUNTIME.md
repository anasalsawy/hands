# Runtime — how to actually run Sawii Persona ("Hands")

> Verified against source at `C:\Users\anasa\hands-gh`, branch `main`, commit `19d5678`.
> Flags, env vars, ports, routes and the WebSocket protocol below were read directly from
> `stitch.py`, `sawii_hands.py`, `persona/config.py`, `persona/api/app.py`, and
> `presets/ui-tars-deepinfra.yaml`. Nothing here is guessed.

---

## 0. What runs where

- **`stitch.py`** is the top-level entrypoint. It starts Persona (the mind) as a server and
  wires the Link + engine(s) into it. Two subcommands: `local` and `remote`.
- **`sawii_hands.py`** is the engine (the "hands"). It runs on the machine whose shell/files
  she may touch (your PC, or a cloud container). It connects **back to** Persona over the
  `/v1/hands` WebSocket. The shared Persona server never runs a shell for anyone.
- The Persona server itself is a Starlette/ASGI app built by `persona/api/app.py` (`create_app`),
  served by `uvicorn`.

Requirements: Python 3.10+, `websockets` (engine), `uvicorn` + the persona dependencies
(server). OCI: `pip install websockets`; for computer-use: `npm install -g open-computer-use`
and run its `doctor` command.

---

## 1. `stitch.py` — CLI reference

```
python stitch.py local  [flags]
python stitch.py remote [flags]
```

### `local` — Persona + engine in one process

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--port` | int | `8800` | Port the Persona server listens on. |
| `--root` | repeatable str | `[]` | Folders the engine may access. Repeat for more. |
| `--approve` | `ask`/`writes`/`auto` | `ask` | Approval mode for the engine. |
| `--deepinfra-key` | str | `$DEEPINFRA_API_KEY` | DeepInfra key (sets `DEEPINFRA_API_KEY`). |
| `--persona-key` | str | `$SAWII_PERSONA_KEY` | Reuse an existing persona key (else one is minted). |
| `--data-dir` | str | `~/.hands` | Where the DB and files live (sets `SAWII_STORE_PATH`). |
| `--agent` | repeatable str | `[]` | Register an external agent. Format below. |

What `local` does, in order (`run_local`):
1. Sets `SAWII_STORE_PATH = <data-dir>/sawii.db` and, if given, `DEEPINFRA_API_KEY`.
2. `load_settings()` → `Runtime(settings)` → `create_app(settings, rt)`.
3. Mints a persona (`create_persona`) if `--persona-key`/`SAWII_PERSONA_KEY` is empty, so
   spawned engines have a key to authenticate with (falls back to a generated key with a
   warning if the DB isn't reachable).
4. Creates a `LinkRegistry`, sets `rt.mind.links`, and calls `links.set_hub(rt.mind.hands, pid)`.
5. Registers the built-in engine (`sawii_hands.py`) named `hands-engine`, place `computer`,
   with the URL `http://127.0.0.1:<port>`, the persona key, roots, and approve mode.
6. Registers each `--agent` spec.
7. **Replaces `rt.mind.lanes.start` with an auto-spawn hook**: whenever Persona starts a new
   lane that has no host delegation (`line.delegated == False`), it spawns a `sawii_hands.py`
   engine for that lane and marks `line.host_status = "spawned"`.
8. Injects a `GET /v1/link` route returning `links.info()`.
9. Serves the app with `uvicorn` on `0.0.0.0:<port>` and installs SIGINT/SIGTERM handlers that
   call `links.shutdown_all()`.

### `remote` — connect an engine to an existing Persona server

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--persona-url` | str | **required** | The Persona server base URL. |
| `--persona-key` | str | **required** | The persona key (`sp_...`). |
| `--root` | repeatable str | `[]` | Folders the engine may access. |
| `--approve` | `ask`/`writes`/`auto` | `ask` | Approval mode. |
| `--agent` | repeatable str | `[]` | Register an external agent. |

`run_remote` builds a `Hands(...)` engine and calls `serve(url, key, hands, place="computer")` —
one persistent engine connected to the remote server. It does **not** start a local server.

### `--agent` spec format

```
name:api_url:api_key:model[:agent_type]
```

- `api_key` may be an env-var reference like `$OPENAI_API_KEY` (resolved from the environment).
- Colons inside `http://`/`https://` are handled; the parser rejoins the URL.
- `agent_type` defaults to `openai`. Examples (from the script docstring/epilog):

```powershell
python stitch.py local --root ~/Projects --deepinfra-key di-...
python stitch.py local --root ~/Projects --agent browser-use:http://localhost:7860:$KEY:model
python stitch.py local --root ~/Projects --agent openai-helper:https://api.openai.com/v1:sk-...:gpt-4o
python stitch.py remote --persona-url https://my-persona --persona-key sp_... --root ~/Projects
```

---

## 2. `sawii_hands.py` — engine CLI reference

Run this on the machine she may work on.

```
python sawii_hands.py --url https://<your-sawii> --key sp_... --root ~/Projects --approve ask
```

Every flag can also come from the environment, so the **same program runs headless as a cloud
sandbox** (a separate container with no app secrets and no DB).

| Flag | Env var | Default | Meaning |
|------|---------|---------|---------|
| `--url` | `SAWII_HANDS_URL` | — | Persona base URL (`https://`→`wss://`, `http://`→`ws://`). |
| `--key` | `SAWII_HANDS_KEY` | — | Persona key. |
| `--root` | `SAWII_HANDS_ROOT` (os.pathsep-separated) | `[]` | Allowed folders. Repeat or use pathsep. |
| `--approve` | `SAWII_HANDS_APPROVE` | `ask` | `ask` / `writes` / `auto`. |
| `--place` | `SAWII_HANDS_PLACE` | `computer` | `computer` or `cloud`. |
| `--computer-use` | `SAWII_HANDS_COMPUTER_USE=1` | off | Enable screen + mouse/keyboard via Open Computer Use. |
| `--lane` | `SAWII_HANDS_LANE` | `""` | Optional execution lane id (lane-bound worker). |

`--url` and `--key` are required. At least one `--root` is required (the engine raises
`SystemExit("give at least one --root folder")` if none).

### Tools the engine exposes

`shell`, `read_file`, `write_file` (overwrite/append), `edit_file` (exact single replacement),
`list_files` (depth 1–4), `search_files` (regex, optional glob). With `--computer-use`, the
`ocu_*` tools discovered from the installed Open Computer Use MCP server are added. There is
**no delete tool**; shell could still delete, which is why shell asks by default.

### Approval semantics

- `approve == "auto"` → everything runs.
- `approve == "writes"` → reads (`read_file`, `list_files`, `search_files`, `ocu_list_apps`,
  `ocu_get_app_state`) run automatically; writes and shell ask.
- `approve == "ask"` (default) → every shell command and every write asks at the terminal.
- **Cloud special case:** if `--place cloud` and `--approve ask`, approve is forced to `auto`
  (nobody is at a terminal in the sandbox).

### Safety

`Hands.resolve()` realpaths every path and refuses anything outside the `--root` folders
(`PermissionError: ... is outside the folders you allowed`). Shell runs `powershell -NoProfile`
on Windows and `bash -lc` elsewhere, in the resolved cwd, with a capped timeout (default 120 s,
max 600 s).

---

## 3. The `/v1/hands` WebSocket protocol (verified)

The engine connects the moment it starts and reconnects with exponential backoff (1 s → 30 s).

**Handshake & frames:**
```
runner -> server  {"type":"hello","key","tools":[...],"place":"computer"|"cloud","lane":"...","host":"<hostname> (<OS>)"}
server -> runner  {"type":"ready","persona":<name>,"tools":[...]}     # else the engine exits ("refused")
server -> runner  {"type":"call","id","name","arguments":{...}}
runner -> server  {"type":"result","id","ok":true,"output":"<json string>"}
server -> runner  {"type":"cancel","id"}                              # the turn let go of that call
runner -> server  {"type":"status","kind":"approval","text":...}      # waiting on the person (per hands.py)
```

- Calls run concurrently (`asyncio.create_task`); a per-call `threading.Event` is the cancel token.
- On `cancel`, the token is set and (if computer-use) `ocu.close()` is called.
- The server advertises each engine's tools to the model as `computer_<tool>` (place `computer`)
  or `cloud_<tool>` (place `cloud`). Two places may be connected at once; a newer connection for
  a place replaces the older one.
- The server authenticates the `hello` by looking the key up (`persona_by_key`); the key's
  persona must exist or the connection is refused.
- Message size cap: `max_size = 2**24` (16 MB) on the engine side.

---

## 4. Environment variables (`SAWII_*`)

Config is read by `persona/config.py` (`Settings`). A local `.env` is auto-loaded by
`load_dotenv()` (cwd `.env` or `<repo>/../.env`; existing env vars always win). Every model is a
plain OpenAI-compatible endpoint — point it anywhere by changing base URL + model.

### Core / storage
| Var | Default | Notes |
|-----|---------|-------|
| `SAWII_DATA_DIR` | `./data` | Data dir. |
| `SAWII_DB_PATH` | `<data-dir>/persona.db` | SQLite file. |
| `SAWII_STORE_PATH` | — | Set by `stitch.py local` to `<data-dir>/sawii.db`. |
| `SAWII_WORKSPACE_ROOT` | `<data-dir>/workspaces` | Deep-lane workspaces. |
| `SAWII_TRACE` / `SAWII_TRACE_DIR` | off / `<data>/trace` | Diagnostic JSONL trace of every call. |
| `SAWII_ADMIN_TOKEN` | `""` | Admin API token. |
| `SAWII_PUBLIC_BASE_URL` | `http://localhost:8800` | For links/redirects. |
| `SAWII_CORS_ORIGINS` | `*` | Allowed origins (bearer-key auth, never cookies). |

### Lobes / models
| Var | Default | Notes |
|-----|---------|-------|
| `SAWII_A_HOST` | (Featherless) | `coreweave` or `tinker` select the mouth host. |
| `SAWII_A_BASE_URL` / `SAWII_A_API_KEY` / `SAWII_A_MODEL` | Featherless `Qwen/Qwen3.8-Flash-Next` | A (voice). |
| `COREWEAVE_API_KEY` | — | Needed when `A_HOST=coreweave`. |
| `TINKER_API_KEY` | — | Needed when `A_HOST=tinker`. |
| `SAWII_B_*` | `openai/gpt-oss-120b`, 90 s | B (adjutant). |
| `SAWII_C_*` | `Compumacy/Psych_Qwen_32B`, 120 s | C (psych observer). |
| `SAWII_EMBED_*` | `Qwen/Qwen3-Embedding-8B`, 20 s | Embeddings. |
| `SAWII_EMBEDDING_PROVIDER` | `gemini` | `gemini` / `deepinfra` / other. |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | — | Gemini embeddings. |
| `DEEPINFRA_API_KEY` | — | Any `api.deepinfra.com` base URL takes this key (outranks stale provider keys). |
| `SAWII_DISABLE_THINKING` | `true` | Ask providers to skip hidden reasoning (big latency win). |
| `SAWII_STEER_MODEL` (+`_BASE_URL`/`_API_KEY`) | off | Expensive "director only" model. |
| `SAWII_CONSULT_MODELS` (+`_BASE_URL`/`_API_KEY`, `_WAIT_MS`) | off | Parallel consultant models. |

### Cursor alias / host worker (R-side)
| Var | Default | Notes |
|-----|---------|-------|
| `SAWII_INFER_A_BASE_URL` | — | **The opt-in switch.** Set to `https://api.deepinfra.com/v1/openai` to enable the CURSOR-DEEPINFRA host worker. |
| `SAWII_INFER_A_MODEL` | off | Overrides the worker's model id. |
| `SAWII_CURSOR_ALIAS` | `sawii-fast` | Default alias for that surface; must be a table key. |
| `DEEPINFRA_API_KEY` | — | Required when the worker is on. |

Alias table (`CURSOR_MODEL_ALIASES`): `sawii-fast` → `deepseek-ai/DeepSeek-V4.1-Flash` (baseline),
`sawii-fast-v4`, `sawii-fast-0731`, `sawii-fast-1017`, `mimo-fast` → `XiaomiMiMo/MiMo-V2.6-Flash`,
`glm-flash` → `zai-org/GLM-5.3-Flash`. It is a **whitelist**: anything not a row resolves to `""`.
Aliases route only the host/execution lane (R); Persona's own voice keeps its model.

### Behaviour
`SAWII_DUAL` (`true`), `SAWII_VISION` (`false`), `SAWII_ANTICIPATE` (`true`),
`SAWII_INTERRUPT_MODE` (`steer`), `SAWII_A_TOOLS` (`true`), `SAWII_LANE_ROUNDS` (`6`),
`SAWII_SHOW_VERDICT` (`true`), `SAWII_SHOW_INNER` (`true`), `SAWII_TAIL_TURNS` (`0`=full),
`SAWII_RECALL_BAND` (`0.30`), `SAWII_OPEN_SESAME_THRESHOLD` (`0.78`),
`SAWII_PICTURE_FIRST_MEMORY` (`true`), `SAWII_LIVE_CHECKS_MAX` (`3`), `SAWII_PSYCHE_HALF_LIFE_DAYS` (`21`).

### Initiative / sleep
`SAWII_HEARTBEAT_SECONDS` (`60`), `SAWII_REFLECT_EVERY_TICKS` (`30`),
`SAWII_QUIET_HOURS` (`22:30-08:00`), `SAWII_DAILY_OUTREACH_CAP` (`4`), `SAWII_SLEEP_HOUR` (`3`),
`SAWII_BACKGROUND` (`true`), `SAWII_SERVE_CONSOLE` (`true`), `SAWII_CUP_CHARS` (`400000`).

### Voice / channels
`SAWII_VOICE_OUTPUT` (`speech`), `SAWII_VOICE_GATE` (`stream`|`whole`),
`SAWII_VOICE_NATIVE_AUDIO`, `SAWII_ASR_*` (`whisper-1`), `SAWII_TTS_*` (Featherless
`ResembleAI/chatterbox`), `SAWII_TTS_VOICE`. Twilio: `SAWII_TWILIO_ACCOUNT_SID`,
`SAWII_TWILIO_AUTH_TOKEN`, `SAWII_TWILIO_FROM`, `SAWII_TWILIO_WHATSAPP_FROM`. Telegram:
`SAWII_TELEGRAM_BOT_TOKEN`, `SAWII_TELEGRAM_WEBHOOK_SECRET`. SMTP: `SAWII_SMTP_*`,
`SAWII_EMAIL_INBOUND_SECRET`.

### Gateway / browser / calls
`SAWII_GATEWAY=1` enables `/api/*` accounts+routes. `SAWII_GATEWAY_DB`, `SAWII_SESSION_TTL`,
`SAWII_ALLOW_REGISTRATION`, `SAWII_RATE_LIMIT_PER_MINUTE`. Browser pane: `SAWII_BROWSER_PANE`
(`1`), `SAWII_BROWSER_DRY_RUN` (`0`), `SAWII_BROWSER_CDP_URL`. Calls: `SAWII_CALLS_ENABLED` (`1`).

### Engine (mirrors sawii_hands.py)
`SAWII_HANDS_URL`, `SAWII_HANDS_KEY`, `SAWII_HANDS_ROOT`, `SAWII_HANDS_APPROVE`,
`SAWII_HANDS_PLACE`, `SAWII_HANDS_COMPUTER_USE`, `SAWII_HANDS_LANE`, and `SAWII_PERSONA_KEY`.

---

## 5. The UI-TARS desktop preset

`presets/ui-tars-deepinfra.yaml` configures **UI-TARS Desktop** to drive the screen with a
DeepInfra OpenAI-compatible model (used alongside / instead of the built-in Open Computer Use).

```yaml
name: DeepInfra OpenAI-compatible (GLM-5.3-Flash)
language: en
vlmProvider: Hugging Face for UI-TARS-1.5
vlmBaseUrl: https://api.deepinfra.com/v1/openai
vlmApiKey: YOUR_DEEPINFRA_API_KEY
vlmModelName: zai-org/GLM-5.3-Flash
useResponsesApi: false
operator: Local Computer Operator
maxLoopCount: 100
loopIntervalInMs: 1000
```

Import via UI-TARS **Settings → Import Preset Config → choose this file**; replace
`YOUR_DEEPINFRA_API_KEY` first. Alternative `vlmModelName`: `Qwen/Qwen2.5-VL-32B-Instruct` or
`zai-org/GLM-5.3-Flash`. There is no dedicated DeepInfra provider label, so the provider name is
cosmetic — the real endpoint is `vlmBaseUrl`.

---

## 6. Ports, routes and the web/companion UI

### Port
- Persona server: **8800** by default (`stitch.py local --port`, and `SAWII_PUBLIC_BASE_URL`
  default `http://localhost:8800`). Runs on `0.0.0.0`.

### Key HTTP/WS routes (from `persona/api/app.py` `create_app`)
- `POST /v1/chat/completions` — OpenAI-compatible chat (streaming supported).
- `POST /v1/responses` — OpenAI Responses API adapter (`persona/api/responses.py`).
- `WS /v1/voice` — realtime voice.
- `WS /v1/hands` — engine WebSocket; `GET /v1/hands` — hands status.
- `GET /v1/events` — live SSE; `GET /v1/messages` — outbox long-poll.
- `GET /health`, `GET /health/subsystems`; `GET /v1/models`, `GET /v1/products`.
- `POST /v1/interrupt`, `POST|GET /v1/commands`, `GET /v1/commands/{id}`, `GET /v1/timeline`.
- `POST /v1/cursor/subagent-event`; `GET /v1/agents`; `GET /v1/persona/screen`.
- Adaptive: `GET|POST /v1/adaptive/tasks`, `/v1/adaptive/tasks/{id}`.
- Persona views: `/v1/persona`, `/v1/persona/presets`, `/v1/persona/clock`,
  `/v1/persona/conversation-board`, `/v1/persona/execution-blueprint`,
  `/v1/persona/execution-board` (+ legacy `/v1/persona/picture`, `execution-sketchpad`),
  `/v1/persona/emotion`, `/v1/persona/memory`, `/v1/persona/loops`, `/v1/persona/assistant`,
  `/v1/persona/anchor`, `/v1/persona/sleep`, `/v1/persona/subconscious`.
- Knowledge: `GET|POST /v1/knowledge`, `DELETE /v1/knowledge/{id}`.
- Voice clone: `GET|POST|DELETE /v1/voice/clone`.
- Deep lane: `POST /v1/deep`, `GET /v1/deep/{job_id}`.
- Tests: `POST /v1/test/turn`, `POST /v1/test/benchmark`.
- Hooks: `POST /hooks/twilio`, `/hooks/telegram`, `/hooks/email`.
- Lane questions: `POST /v1/lane/questions`.
- `GET /v1/link` — injected by `stitch.py local` (Link status/info).
- Gateway (when `SAWII_GATEWAY=1`): `/api/auth/register|login|me|logout`, `/api/persona`.
- Browser pane (when `SAWII_BROWSER_PANE!=0`): registered by `gateway.browser.register_browser_routes`.

### Serving the web/companion UI
- `GET /` redirects to `/app/` when `SAWII_SERVE_CONSOLE` is on (`serve_console`, default true);
  otherwise it returns `{"ok": true, "app": "gateway", "health": "/health"}`.
- `GET /app` is the built-in web app (`web_app`) — chat, voice, live timer, pop-ups.
- If a built companion build exists at `persona/web/companion/index.html`, it is mounted at
  `/app` (`Mount("/app", StaticFiles(...))`). That directory is produced from the separate
  `ui/` repo (built to `ui/dist` and copied under `persona/web/companion`). **The companion web
  UI source is not in this repository** — this repo serves the developer console and the API.
- In production the operator usually sets `SAWII_SERVE_CONSOLE=false` so the user-facing app is
  the gateway (`companion-app`), avoiding two competing UIs.

---

## 7. Step-by-step startup (Windows / PowerShell)

### A. Install
```powershell
cd C:\Users\anasa\hands-gh
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt    # or: pip install starlette uvicorn websockets
# engine-only on a second machine:
pip install websockets
# optional computer use:
npm install -g open-computer-use
open-computer-use doctor
```

### B. Configure (optional `.env` in the repo root, or real env vars)
```powershell
$env:DEEPINFRA_API_KEY = "di-..."
$env:SAWII_INFER_A_BASE_URL = "https://api.deepinfra.com/v1/openai"   # enable Cursor host worker
$env:SAWII_CURSOR_ALIAS = "sawii-fast"
# $env:GOOGLE_API_KEY = "..."       # Gemini embeddings (default)
# $env:SAWII_TRACE = "1"            # only for diagnosis; writes whole conversation to disk
```

### C. Run — LOCAL (Persona + engine in one process)
```powershell
python stitch.py local `
  --root "C:\Users\anasa\Projects" `
  --deepinfra-key $env:DEEPINFRA_API_KEY `
  --port 8800 `
  --data-dir "C:\Users\anasa\.hands" `
  --approve ask
```
Expect log lines: `Created local persona id=… key=sp_…`, `─── Hands stitched ───`, the server
URL, engine path, roots, agents, key, and `Link info: http://0.0.0.0:8800/v1/link`. The server
is now on `http://localhost:8800`; the console is at `http://localhost:8800/app/`.

### D. Run — REMOTE (engine connects to an existing Persona)
On the machine whose files she may touch:
```powershell
python sawii_hands.py `
  --url https://your-persona.example `
  --key sp_YOUR_PERSONA_KEY `
  --root "C:\Users\anasa\Projects" `
  --approve ask
# or the equivalent via stitch:
python stitch.py remote `
  --persona-url https://your-persona.example `
  --persona-key sp_YOUR_PERSONA_KEY `
  --root "C:\Users\anasa\Projects" `
  --approve ask
```
Expect: `connected - <persona> can now use: shell, read_file, …` then `folders: … approve: ask`.

### E. Talk to her
- Console: open `http://localhost:8800/app/`.
- OpenAI-compatible client:
```powershell
$body = @{ model = "sawii-persona"; messages = @(@{ role = "user"; content = "hi" }) } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8800/v1/chat/completions `
  -Method Post -Headers @{ Authorization = "Bearer sp_YOUR_KEY"; "X-Sawii-Client" = "companion" } `
  -ContentType "application/json" -Body $body
```
(`X-Sawii-Client: companion` selects the companion product — mind + her computer/browser.)

### F. Verify
```powershell
Invoke-RestMethod http://localhost:8800/health
Invoke-RestMethod http://localhost:8800/v1/link        # Link status (local mode only)
Invoke-RestMethod http://localhost:8800/v1/hands       # hands connection status
Invoke-RestMethod http://localhost:8800/health/subsystems
```

---

## 8. Minimal "just the engine, no server" smoke test

The engine needs a Persona server to connect to; there is no standalone mode. To confirm the
engine itself is sane without a full stack, you can still exercise its tool logic by importing
`Hands`:

```powershell
python -c "from sawii_hands import Hands; import os; h=Hands([os.getcwd()],'auto'); print(h.t_list_files('.', 1))"
```

Without a reachable `--url`/`--key` the `serve()` loop will simply retry with backoff and print
`disconnected (...); retrying in N s` — that output is expected and not an error in the tooling.

---

## 9. Data layout and cleanliness

- Default data dir (local mode): `C:\Users\anasa\.hands` (`--data-dir`); DB at
  `<data-dir>/sawii.db` (set via `SAWII_STORE_PATH`). Default (non-stitch) data dir: `./data`
  with DB `<data-dir>/persona.db`.
- Deep-lane workspaces: `<data-dir>/workspaces`.
- Trace (if enabled): `<data-dir>/trace`.
- **Repo hygiene:** the clone carries committed `persona/**/__pycache__/*.pyc`. Some reference
  modules with no `.py` source at this commit; they are stale build artifacts and should be
  removed and git-ignored. They are not required to run anything.

---

## 10. Troubleshooting quick reference

| Symptom | Likely cause / fix |
|---------|-------------------|
| Engine keeps `disconnected (...); retrying` | Wrong `--url`/`--key`, or no persona for that key. Check `--key` and that Persona is up. |
| Server logs `refused: {...}` from engine | Key not found by `persona_by_key`. Create a persona (`/admin/personas`) or use `stitch.py local`, which mints one. |
| Engine exits `give at least one --root folder` | Pass at least one `--root`. |
| Tools run but nothing happens on screen | `--computer-use` not enabled, or `open-computer-use` not installed/healthy. |
| Shell refused | Approval mode `ask`; answer `y` at the terminal, or use `writes`/`auto` for trusted folders. |
| `is outside the folders you allowed` | Path escaped the `--root` set; add the folder as a `--root`. |
| No `/v1/link` route | You launched the server directly (uvicorn) rather than via `stitch.py local`. |
| Two UIs / editing the wrong one | Set `SAWII_SERVE_CONSOLE=false` on the deployed backend so the gateway is the only UI. |
| Cursor alias ignored | `SAWII_INFER_A_BASE_URL` not set to DeepInfra, so the host worker is `None`. Set it + `DEEPINFRA_API_KEY`. |
