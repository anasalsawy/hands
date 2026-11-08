# Hands — Application Brief

*Hand this to anyone building, designing, or shipping the app. It is the whole picture in one file.*

---

## 1. What this is

**Hands** is a desktop app that gives **Persona** (an AI mind) hands on a computer.

- **Persona = the mind.** One shared context, one conversation history, one line of thought. It thinks and coordinates.
- **Hands = the body.** The app that hosts Persona and drives a real desktop/browser through it.
- **The engine** (`sawii_hands.py`) = the muscles. A model-less tool executor. No LLM, no API key. It runs shell/file/screen actions and reports back.

The product is a **deck of cards**. Each card is one live desktop instance running its own task, and many run **at the same time**. The user states a goal; the mind splits it into parallel lines of work and each line gets its own engine automatically. Concurrency is the mind's decision — **not** a button the user presses.

---

## 2. The two lanes (this is the core mental model)

Persona exposes exactly two kinds of lane. Both share the **same mind** — same memory, same tools, same persona.

| Lane | What it is | Who drives it |
| --- | --- | --- |
| **L** | The **conversation** lane. Voice and chat. | The mind itself |
| **R** | The **execution** lanes (R1, R2, R3…). Desktop/browser work. | One engine each |

- **L** = the mind talking. It thinks through a Featherless model by default.
- **R** = the mind acting. Each R lane is one `sawii_hands.py` process doing work on the machine.

**L is one lane with two mouths** (same mind, same memory):

- **L1** = the **fast** mouth — a small local model (llama.cpp, qwen2.5-1.5b on `127.0.0.1:8080`). Speaks first, ~0.4–1.2 s. Handles short social turns only.
- **L2** = the **deep** mouth — the full Featherless model. Produces the real answer.

They **race**. L1 speaks a short bridge while L2 is thinking; when L2's answer arrives it continues, and the echo is dropped. One mind, two speeds. The person hears a natural first line almost instantly, and the real answer right behind it.

**What L1 must never do:** invent a fact, a memory, a name, a number, or a tool result. Anything needing memory, a fact, or an action **abstains** (replies `PASS`) and lets L2 answer. This gate is enforced in code (`needs_deep_path`), not by asking the small model to behave — it will lie if you only ask nicely.

---

## 3. Architecture (how the pieces connect)

```
                          ┌───────────────────────────────┐
                          │        Persona  (mind)         │
                          │  shared context · one history  │
                          │      L (talk)  ·  R (work)     │
                          └───────┬───────────────┬────────┘
                                  │               │
                     ┌────────────┘               └─────────────┐
                     ▼                                          ▼
             ┌───────────────┐                          ┌───────────────┐
             │  L  lane       │                          │  R lanes       │
             │  L1 fast       │                          │  R1  R2  R3 …  │
             │  L2 deep       │                          │                │
             └───────────────┘                          └───────┬────────┘
                                                                 │  Link
                                                     ┌───────────┴───────────┐
                                                     ▼                       ▼
                                             ┌───────────────┐       ┌───────────────┐
                                             │ sawii_hands R1 │       │ sawii_hands R2 │
                                             │  (engine)      │       │  (engine)      │
                                             └───────────────┘       └───────────────┘
```

**The Link** (`persona/link.py`) is the adapter between the mind and an engine. It picks one of two strategies:

1. **Spawn** — no native delegation. Link launches one `sawii_hands.py` per R lane and connects over WebSocket.
2. **Hijack** — the external agent already has sub-agents (Hermes `delegate_task`, Cursor `Task`, Claude `Agent`). Link uses that existing pathway instead of inventing a new one.

Hands ships with its own engine and speaks the same Link protocol, so it can also plug into external agent frameworks. That is the partnership angle: **drop Persona in as the inference backend and get automatic concurrency the host doesn't have.**

---

## 4. What the user does in the app

The app window has six tabs across the top:

| Tab | What it's for |
| --- | --- |
| **Courts** | The **deck** — the main screen. Cards, one live desktop each. |
| **Motor** | Engine + Link + computer health. Shows what can actually act. |
| **Chat** | Text conversation with the mind. |
| **Voice** | Live voice session (mic). |
| **Phone** | Outbound/inbound calls. |
| **Browser** | Live browser stream. |

### The Courts tab (the main flow)

This is the product. Everything else supports it.

**To start work:**

1. Type an optional **card title**, pick a **surface** (`desktop`, `window`, `region`, `browser`).
2. Click **Add card and start** — creates a card *and* spins up its desktop instance in one move.
   (Or **Add card** to place it and start later.)
3. Type a **goal** on the card — plain language, e.g. *"open the invoices folder and total this month."*
4. Click **Send goal**. That starts the instance if it isn't running, otherwise it **steers** the running one.

**Each card shows, live:**

- the card's status (`idle` → `spawning` → `running` / `waiting` / `stopped`),
- its own **live desktop stream** (a real picture of what that instance is doing right now),
- its **work line** (which R lane), its **engine** (which Link + adapter), and its **surface**.

**Each card's controls:** `Start instance` · `Send goal` / `Steer` · `Stop` · `Voice` · `Remove`, plus a `target` toggle to change what that card watches.

**The point of the grid:** several cards run **simultaneously**, each with its own engine and its own work line. That is the whole product — concurrency you can watch.

### The key rule

**The user does not orchestrate concurrency.** The user states a goal. When the mind splits a goal into parts, each part **automatically** gets its own R lane and its own engine process, and the card shows the truth of that. The deck reflects what the mind decided — it is a view, not a control panel for spawning.

---

## 5. What's already built (do not rebuild these)

- `desktop/` — the **Tauri 2 native Windows app**. Real `.exe`, identifier `ai.sawii.hands`. React 19 + Vite front end, 9 components: `DeckView`, `CourtCard`, `LiveSurface`, `MotorView`, `ChatView`, `VoiceView`, `PhoneView`, `BrowserView`, `ConnectView`.
- `stitch.py` — the entry point that wires Persona + engine + Link into one running process.
- `sawii_hands.py` — the engine (shell, files, `ocu_*` screen tools).
- `persona/` — the mind, bundled. Coordinates L and R, owns the deck (`persona/court.py`).
- `voice_sidecar.py` — the voice sandbox (STT/TTS/barge-in), started on `:18766`.
- Selftests: `scripts/court_selftest.py`, `scripts/court_http_selftest.py`, `scripts/split_engines_selftest.py`, `scripts/run_checks.py`.

**The interface the UI binds to (the deck contract):**

```
REST  GET    /v1/courts                 list the deck
      POST   /v1/courts                 create a card
      GET    /v1/courts/{id}            one card
      DELETE /v1/courts/{id}            remove a card
      POST   /v1/courts/{id}/spawn      start its instance (own lane + engine)
      POST   /v1/courts/{id}/goal       send a goal (starts if idle, else steers)
      POST   /v1/courts/{id}/cancel     stop (card stays)
      POST   /v1/courts/{id}/voice      hand its last update to L
SSE   GET    /v1/courts/{id}/events     this persona's deck
WS           /v1/court-events           one socket, every card
WS           /v1/voice                  the L voice session
```

The front end never reimplements the deck. It calls these and renders what comes back.

---

## 6. How to run it

**Backend + mind** (from `hands/`):

```powershell
python stitch.py local --root C:\Users\you\Projects --port 8800
```

It prints the persona key: `Created local persona id=1 key=sp_…`. Copy it — that's the one credential and it's only shown once.

**The app** (from `hands/desktop/`):

```powershell
npm install
npm run dev            # browser preview at 127.0.0.1:5181, or:
npx tauri dev          # the real native window
```

First launch shows the **Connect** screen — enter the backend URL (`http://127.0.0.1:8800`) and the persona key. Both are kept in the app's own storage. **Settings** changes them later.

**Fast lane (L1)** — keep the local model warm so no turn ever pays for a cold load:

```powershell
powershell -File persona\voice\bench\start_fast_model.ps1
```

**Build the `.exe`:**

```powershell
cd hands\desktop
node scripts\env.mjs        # checks node/npm/cargo/rustc/link.exe/cl.exe/disk
npx tauri build
```

> **Blocker on a fresh machine:** the native build needs the **MSVC C++ toolchain + Windows SDK** (`link.exe`), which rustup does not provide. Without it you get `linker 'link.exe' not found`. Install it, then sign out/in:
> ```powershell
> winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
> ```
> ~4 GB download, ~6 GB on disk, plus ~1.5 GB for the release target. Output lands in `desktop\src-tauri\target\release\Hands.exe`.

---

## 7. The rules a builder must not break

1. **One mind.** Do not fork the context, history, or line of thought. L and every R read the same shared timeline.
2. **The engine has no LLM and no provider key.** It is a pure tool executor. Only Persona's key reaches it (for `/v1/hands` auth). The key that powers thinking is Persona's `--deepinfra-key` / Featherless credential, and it stays in the mind.
3. **The engine/VLM is R.** For screen work, R asks for a screenshot + state and *R itself* decides the next click. Do not add a second VLM inside the engine.
4. **Concurrency is the mind's.** Never add a user-facing "how many agents" slider. Split → auto-spawn is automatic.
5. **L1 speaks, never invents.** Anything factual/actionable abstains to L2. The gate is code.
6. **Fail honestly.** A failed model turn must go to `failed` with the real error, and must **never** be written to history as a fake apology ("Sorry, my thoughts got cut off…").
7. **Reuse, don't rebuild.** Voice (`persona/voice/realtime.py`, `/v1/voice`), the orb, the dialer, and the Tauri shell already exist. Attach around them.

---

## 8. Working systems, in plain terms

- **You type or speak a goal** → the mind understands it.
- **The mind splits it** into one or more lines of work.
- **Each line gets its own engine** automatically, shown as a card with a live desktop picture.
- **They run at the same time**, each on its own work line, each with its own safety allow-list (`--root`) and approval mode (`ask` / `writes` / `auto`; shell always asks by default; there is **no delete tool**).
- **L answers you** the whole time — instantly with L1, fully with L2. Same mind, so what L says and what R does never drift apart.

---

## 9. Repository layout

```
hands/                      the app  (github.com/anasalsawy/hands)
  desktop/                  Tauri 2 native app (React + Vite)
  persona/                  the mind (bundled copy)
  stitch.py                 wire it all together (entry point)
  sawii_hands.py            the engine (model-less tool executor)
  voice_sidecar.py          voice sandbox (:18766)
  ocu_mcp.py                Open Computer Use client (screen/mouse/keyboard)
  presets/                  UI-TARS Desktop import file (not read by this app)
  scripts/                  selftests + probes
```

`sawii-persona/` (the standalone library repo) holds the same mind as `src/sawii_persona/`. **The application lives in `hands`.** Persona is the brain; Hands is the body.
