# Modules — Sawii Persona ("Hands")

> Source of truth: GitHub clone `C:\Users\anasa\hands-gh`, branch `main`, commit `19d5678`.
> 138 Python modules under `persona/` plus root `stitch.py` and `sawii_hands.py`.
> Descriptions, entry points and imports are taken from the actual source (docstrings, class/def
> signatures, `import` lines). "Line count" is physical lines at this commit.

Legend: **Entry** = public classes/functions; **Imports** = intra-repo imports only
(third-party/stdlib omitted).

---

## Root scripts

### `stitch.py` (329 lines)
Wires every element into one running app: starts the Persona server, seeds personas, attaches
the Link and hands, and (crucially) performs **lane start auto-spawn**. `run_local` and
`run_remote` are the two top-level entrypoints; `_register_agent_from_spec` builds Link agent
profiles from `--agent` specs.
**Entry:** `run_local`, `run_remote`, `_register_agent_from_spec`, `_shutdown`, `main`.
**Imports:** (driver imports inside functions).

### `sawii_hands.py` (304 lines)
Sawii's hands on *your* computer: a small agent app that runs shell commands and file
operations, then connects back to the Persona server over the `/v1/hands` WebSocket. Tools:
`shell`, `read_file`, `write_file`, `edit_file`, `list_files`, `search_files`; optional Open
Computer Use (`ocu_*`). Approval modes `ask`, `writes`, `auto`. **It is the process the server
never runs for you** — you run it yourself.
**Entry:** `Hands`, `serve`, `main`. **Imports:** none intra-repo.

---

## Top-level package

### `persona/__init__.py` (3)
Package docstring: *"Sawii Persona — a three-lobe AI companion."*

### `persona/config.py` (494)
All runtime configuration from the environment (prefix `SAWII_`). Defines `Settings`, the
four model `ModelEndpoint`s, newline-safe personas seed, and the **Cursor alias whitelist**
(`resolve_cursor_alias`, `cursor_alias_for`, `cursor_default_model`, `cursor_worker_endpoint`).
The alias table is a closed whitelist — unknown strings return `""`.
**Entry:** `Settings`, `ModelEndpoint`, `load_settings`, `resolve_cursor_alias`, `cursor_alias_for`, `cursor_default_alias`, `cursor_default_model`, `cursor_worker_endpoint`, `load_dotenv`.
**Imports:** none.

### `persona/cli.py` (64)
Command line: `serve`, create a persona, run a sleep cycle by hand.
**Entry:** `main`. **Imports:** `.config`.

### `persona/bootstrap.py` (91)
First-run bootstrap: create a persona from a JSON file if the store has none.
**Entry:** `run`. **Imports:** `.config`, `.identity`, `.store`.

### `persona/products.py` (72)
The two products. `COMPANION_FEATURES` vs `INFERENCE_FEATURES` and `describe()`. The companion
gets her computer/browser/memory and may learn weights; a visit does not update weights
(2026-10-21 lock). Detected from `X-Sawii-Client: companion`.
**Entry:** `describe`. **Imports:** none.

### `persona/profiles.py` (88)
Who the persona is — always written by the user. `PRESETS`, `build_clone`, and `persona_frame`
(the frame that tells the model who it is). Never written to memory.
**Entry:** `build_clone`, `preset_list`, `persona_frame`. **Imports:** none.

### `persona/identity.py` (61)
Persona = API key. The key is the identity; only its hash is stored. `create_persona`,
`persona_by_key`, `get_persona`, `find_by_contact`, `new_key`, `hash_key`, `public_view`.
**Entry:** as above. **Imports:** `.store`.

### `persona/store.py` (554)
Durable storage — the single source of truth (SQLite WAL). `Store` owns all tables and
`now`/`jdump`/`jload` helpers. Tables include `personas`, `turns`, `events`, `layers`,
`hypotheses`, `entities`, `loops`, `rules`, `verdicts`, `anchor_runs`, `updates`, `thoughts`,
`outbox`, `juice`, `finetune_jobs`, `character_anchors`, `rulings` (+FTS), `escalations`,
`live_commands`, `persona_decisions`, `shared_timeline`, knowledge/scholar tables.
**Entry:** `Store`, `now`, `jdump`, `jload`. **Imports:** none.

### `persona/trace.py` (182)
The trace: one JSONL line for everything that happens inside Persona (redacted, JSON-safe).
**Entry:** `Trace`, `setup`, `trace`, `event`. **Imports:** none.

### `persona/subsystems.py` (103)
Subsystem health registry — honest green/red status for every part of the mind.
**Entry:** `SubsystemHealth`, `Registry`. **Imports:** none.

### `persona/timesense.py` (199)
The persona's sense of time: its own clock and timezone, gaps between visits, part-of-day,
quiet hours, stopwatch.
**Entry:** `Clock`, `fmt_duration`, `local_dt`, `valid_zone`, `place_from_zone`, `part_of_day`, `stopwatch`, `describe_gap`, `in_quiet_hours`. **Imports:** none.

### `persona/context.py` (212)
The context stream — how one continuous line of thought is assembled (stable cached prefix +
live part last, so chat templates with a single leading system turn work). Owns `Scratch`
(per-persona, never persisted) and the adjutant/conscience/integrity/loops blocks.
**Entry:** `Scratch`, `integrity_block`, `relevant_loops`, `loops_block`, `build_messages`, `_loop_words`. **Imports:** `.brains.prompts`, `.profiles`, `.store`, `.timesense`.

### `persona/emotion.py` (156)
Emotional memory as a **chord**: many feelings at once, each with intensity, over 8 axes.
`blend`, `resonance`, `normalize`, `render`, `parse_declaration`, `color`.
**Entry:** as above. **Imports:** none.

### `persona/embedder.py` (182)
Embeddings for context broadening. `Embedder`, `hash_embed` (deterministic fallback),
`to_blob`/`from_blob`, `cosine`, `cosine_matrix`.
**Entry:** `Embedder`, `hash_embed`, `cosine`, `cosine_matrix`, `to_blob`, `from_blob`. **Imports:** none.

### `persona/deepinfra_embedding.py` (90)
DeepInfra EmbeddingGemma-2 client: shared 768-D text/audio semantic memory.
**Entry:** `DeepInfraEmbeddingLLM`. **Imports:** `.llm`.

### `persona/gemini_embedding.py` (143)
Gemini Embedding-2 native API client for shared text/audio semantic vectors.
**Entry:** `GeminiEmbeddingLLM`. **Imports:** `.llm`.

### `persona/llm.py` (522)
OpenAI-compatible model clients: chat, streaming with tool calls, embeddings. `LLM` (ABC),
`OpenAICompatLLM`, `ScriptedLLM` (tests), `ChatResult`, `ToolCall`, `extract_json`,
`strip_reasoning`, `usage_of`.
**Entry:** as above. **Imports:** `.config`.

### `persona/llm_tinker.py` (271)
Tinker SDK-based Inkling text/tool inference for the L/R kernel.
**Entry:** `TinkerInklingLLM`, `final_text`, `raw_turn`, `parsed_tool_calls`, `_field`. **Imports:** `.llm`.

### `persona/kv_sessions.py` (147)
Durable llama-server KV checkpoints for scoped Persona generation. `KVSlots`, `KVSessionLLM`,
`KVCapacityError`.
**Entry:** as above. **Imports:** none.

### `persona/mind.py` (2568)
**The Mind: one turn of the wheel, end to end.** Assembles the turn (context, recall, clock
via `timesense`, project orientation), runs the dual loop through `_dual_speak`, wires the live
lobes (`_LiveLobes`), performs post-turn work (adjutant review, memory write, verdict, aging),
and owns the `Bus` for live client events. `respond`/`_respond_inner` are the main turn paths;
`chat` is the programmatic turn; `interrupt` steers an active turn; `answer_question` answers a
lane question; `close` shuts down.
**Entry:** `Mind`, `Bus`, `_LiveLobes`, `_blend_moods`, `_log_failure`, `_union`, `_covered`, `_turn_metrics`.
**Imports:** `.brains`, `.brains.consults`, `.brains.meter`, `.brains.prompts`, `.brains.voice_a`, `.config`, `.context`, `.embedder`, `.host_lanes`, `.identity`, `.kv_sessions`, `.life.fastweight`, `.life.subconscious`, `.life.update`, `.llm`, `.memory.recall`, `.memory.records`, `.project_orientation`, `.store`, `.timesense`, `.voice.native_input`.

### `persona/inference_lanes.py` (1350)
**Lanes for any agent: the voice talks while every line of work keeps moving.** `Line` (one
line of work, persisted with a lane id like `ln12345678`), `LaneBook` (registry), `run_lanes`
(scheduler), `Transcript`/`read_transcript`/`story` (narrative of a lane), `ClaimGate` +
`write_targets`/`claims_only`/`same_work`/`group_starts` (write-conflict collision detection and
parallelism), `lane_mark`/`marked_lane` (lane markers in text).
**Entry:** `Line`, `LaneBook`, `line_of`, `run_lanes`, `Round`, `ClaimGate`, `Transcript`, `read_transcript`, `story`, `write_targets`, `claims_only`, `same_work`, `group_starts`, `lane_mark`, `marked_lane`.
**Imports:** `.brains.voice_a`, `.host_delegation`, `.llm`.

### `persona/link.py` (1293)
**Link — Persona's universal adapter for connecting to any external agent.** One integration
point; three strategies (hijack / spawn / api_bridge) chosen automatically by `Link.detect`.
Drivers: `HermesDriver`, `CursorDriver`, `ClaudeCodeDriver`, `ClineDriver`, `GenericHijackDriver`
(hijack); `OpenAIBridgeDriver`, `BrowserUseDriver` (api_bridge); spawn uses
`sawii_hands.py`/agent binaries. `AgentProfile`, `LinkedLane`, `Link`, `LinkRegistry`.
**Entry:** as above. **Imports:** `.host_delegation`.

### `persona/hands.py` (260)
**Her hands on the person's own computer / in a cloud sandbox.** `HandsHub` holds each persona's
connected engine(s) for places (`computer`, `cloud`), authenticates by key (`persona_by_key`),
exposes tools to the model as `computer_<tool>` / `cloud_<tool>`, routes `call`/`result`/`cancel`
frames, and resolves pending tool futures. `hands_ws` is the FastAPI WebSocket endpoint.
**Entry:** `HandsHub`, `Runner`, `hands_ws`, `hands_status`, `_show_screen`, `_waiting_on_person`. **Imports:** none.

### `persona/host_lanes.py` (94)
Explicit Persona lane contract and asynchronous request scope: `parse_lane`, `lane_lock_key`,
`scope_key`, `use_host_tool_kernel`, `additional_host_task`, `choose_host_inference`,
`cursor_alias_model_id`.
**Entry:** as above. **Imports:** none.

### `persona/host_delegation.py` (126)
Host tool schema validation, action deduplication and output boundary: `is_host_background_tool`,
`route_host_output`, `_checked_action`, `_completed_actions`.
**Entry:** as above. **Imports:** none.

### `persona/live_commands.py` (444)
**Durable, concurrent live-command control plane** for execution lanes. `LiveCommands` with
submit/get/list/active_lanes/mark_applied/retry_pending/recover_on_startup. A `202` means only
*accepted* — steer/cancel reach remote tools eventually. `target_type ∈ {auto,goal,lane,host}`,
`action ∈ {steer,cancel}`; `CommandValidation`, `CommandConflict`.
**Entry:** `LiveCommands`, `CommandValidation`, `CommandConflict`. **Imports:** none.

### `persona/cursor_bridge.py` (72)
Cursor subagent lifecycle receipts into Persona's single shared mind. Honest about
verification: a completion is recorded as *unverified* (`"Cursor <status> (unverified)"`), and a
new task only becomes a Persona lane when explicitly marked `[persona-lane lnXXXXXXXX]`.
**Entry:** `receive_cursor_subagent`. **Imports:** none.

### `persona/desktop.py` (15)
Stub: Desktop GUI moved to Open Computer Use in the authenticated desktop shell.
**Entry:** `Desktop`. **Imports:** none.

### `persona/computer.py` (174)
Her computer, inside the app: `Computer` tool surface (schema-driven) bridging the mind to the
computer-use engine.
**Entry:** `Computer`, `_schema`. **Imports:** none.

### `persona/documents.py` (242)
Documents she makes for the person: real files handed over as links. Markdown → PDF rendering
(font/shaping detection), `safe_name`.
**Entry:** `Documents`, `render`, `render_pdf`, `safe_name`, `_find_fonts`, `_shaping_available`. **Imports:** none.

### `persona/knowledge.py` (275)
Files the person gives her — retrieved on the turn, not remembered as fact. Ingest/chunk/distill
into "voices", list/delete sources, retrieve passages, build the passages block.
**Entry:** `ingest`, `distill`, `retrieve`, `passages_block`, `for_turn`, `voices`, `list_sources`, `delete_source`, `read_upload`, `chunk_text`, `knowledge_facts`. **Imports:** `.embedder`, `.store`.

### `persona/plugins.py` (243)
Her tool belt, and the tools she makes: `Plugins` (calc, fetch_url, self-authored tools).
**Entry:** `Plugins`, `fetch_url`, `_schema`, `_calc`. **Imports:** `.documents`.

### `persona/picture.py` (342)
The picture: everything, at three depths, small enough to be read every step.
`PersonPicture`, `TaskPicture`, `touched`, `_one_line`.
**Entry:** as above. **Imports:** none.

### `persona/picture_board.py` (333)
Durable Execution Board: tool outcomes and live project state shared by L/R.
**Entry:** `PictureBoard`, `_cap_rows`. **Imports:** `.picture`, `.sketch_graph`.

### `persona/sketch_graph.py` (143)
Deterministic, editable project sketch graph. `SketchGraph`, `clean`, `position`.
**Entry:** as above. **Imports:** none.

### `persona/perception.py` (65)
Read-only per-persona browser/computer evidence for the companion L lane. `screen_frame`.
**Entry:** `screen_frame`. **Imports:** none.

### `persona/project_orientation.py` (72)
Non-blocking, versioned project orientation derived from shared-timeline evidence.
**Entry:** `ProjectOrientation`. **Imports:** none.

### `persona/shared_timeline.py` (123)
Authoritative, append-only, persona-scoped conversation and tool event timeline. The one record
every lobe/lane reads from.
**Entry:** `SharedTimeline`. **Imports:** none.

### `persona/conversation_board.py` (385)
B's internal Conversation Board (not a drawing or profile): maintains the live read of the
conversation as monotonic patches for L.
**Entry:** `ConversationBoard`, `_limit`. **Imports:** `.llm`.

### `persona/execution_blueprint.py` (128)
B-authored cumulative execution guidance for L and R, separate from tool receipts.
**Entry:** `ExecutionBlueprint`. **Imports:** `.conversation_board`, `.llm`.

### `persona/decision_map.py` (175)
Decision-choice telemetry for L/R training, separate from Quranic curricula. `DecisionJournal`
exports **reviewed-successes only** for SFT; `safe_context`.
**Entry:** `DecisionJournal`, `safe_context`. **Imports:** none.

---

## `persona/kernel/` — the dual loop and its machinery

### `persona/kernel/__init__.py` (36)
*"The dual loop: one voice, two writings, one model."* Exports `.caps`, `.dual`, `.writings`.

### `persona/kernel/dual.py` (3571) — **the core**
The dual loop itself. **One voice, two writings, one model**: the speaking writing (L) streams
to the person while the silent writing (R) drafts the next thought/tool calls at sentence
boundaries without breaking the cached prompt prefix.

Internal structure (verified):
- `HoldFilter` — holds partial tokens so a sentence can be re-aimed before it is spoken
  (`feed`/`flush`/`handoff`/`task`/`intent`/`hold`).
- `MetaFilter` — strips meta/self-referential output (`is_meta`, `echoes`, `restates`,
  `reasks`, `recaps`, `says_again`, `paraphrases`).
- `Beat` (one loop unit), `DualResult` (`switches()`, `trace()`), `_Gate` (output gate that can
  release a corrected reply).
- `DualLoop`: `run()` (the ~940-line main loop), `stream()`, `interrupt`, `hear_inner`,
  `_drain_inner`, `_speak`, draft-ahead (`_arm_draft`/`_wait_draft`/`_refresh_draft`/`_choose_draft`/`_hand_whole`),
  `_hands` (turn tool calls into execution), `_think`/`_think_inner`/`_silent_chat`/`_prefetch`
  (silent writing), verification wiring (`_check`/`_apply_verdict`/`_publish_late_verification`),
  tool bookkeeping (`_land_hands`, `_kernel_todo`, `tool_checklist`, `tool_evidence`).
- Module helpers: `trim_history`, `_parallel_safe`, `_hand_groups`, `_reground_name`,
  `_screen_tool`, `_reads_screen`, `parse_todo`, `task_parts`, `drop_dangling_bridge`,
  `strip_interjection`, `restates`/`reasks`/`recaps`/`says_again`/`paraphrases`, `is_meta`, `echoes`.
**Imports:** `..llm`, `.caps`, `.floor`, `.parts`, `.screen_plan`, `.screen_prove`, `.writings`.

### `persona/kernel/caps.py` (177)
What a host may give the loop, and what it may withhold: `MemoryCap`, `ToolsCap`, `VerifyCap`,
`InsightCap`, `VerificationResult`, `CapabilityBundle`, `NullCapabilities`.
**Entry:** as above. **Imports:** none.

### `persona/kernel/writings.py` (75)
Two internal passes of one voice (README.md). `Writing`, `writings()`.
**Entry:** `Writing`, `writings`. **Imports:** none.

### `persona/kernel/floor.py` (30)
Cheap floor selection. Readiness never interrupts the public owner. `requested_starter`,
`starting_lobe`.
**Entry:** as above. **Imports:** none.

### `persona/kernel/parts.py` (298)
Two parts at once: each lobe drives one independent part of a task. `split_parts`, `run_parts`,
`can_run_in_part`, `lane_of`, `label`.
**Entry:** as above. **Imports:** `..llm`.

### `persona/kernel/lane_routing.py` (185)
The routing contract for the dual-lobe mind, server side: `router_block`,
`lane_addressing_rules`, `needs_block`, `is_conversation_lobe`, `lane_prompt`, `normalize_lane`,
`split_steering`, `addressed_steer`, `may_act_on_user_turn`.
**Entry:** as above. **Imports:** none.

### `persona/kernel/adaptive.py` (389)
Durable task state, process-safe resource ownership and adaptive strategy scores. `Task`,
`TaskLedger`, `Assignment`, `AdaptiveCoordinator`, `LeaseBusy`.
**Entry:** as above. **Imports:** none.

### `persona/kernel/goal_agent.py` (710)
Model-driven persistent goals using Sawii's own cheap lobe and verifier.
**Entry:** `GoalAgent`, `_audit_safe`, `_json`, `_failed`. **Imports:** `.adaptive`, `.conflict_coordinator`, `.lane_routing`, `.task_runtime`.

### `persona/kernel/task_runtime.py` (395)
Evidence-gated task runner: L stays conversational, R1..Rn execute elastically.
`AdaptiveTaskRunner`, `WorkItem`, `WorkResult`.
**Entry:** as above. **Imports:** `..decision_map`, `..life.update`, `.adaptive`, `.conflict_coordinator`, `.execution_pool`.

### `persona/kernel/execution_pool.py` (82)
Persona-wide R execution lanes. `ExecutionPool`.
**Entry:** `ExecutionPool`. **Imports:** none.

### `persona/kernel/conflict_coordinator.py` (113)
Conflict-aware action admission for independent R workers. `ActionIntent`,
`ResourceConflictCoordinator`, `SupersededAction`.
**Entry:** as above. **Imports:** none.

### `persona/kernel/screen_plan.py` (141)
The earliest NeuralAgent loop, inside this engine: plan → prove-on-screen. `ScreenPlan`,
`parse_plan`, `parse_next_action`, `marked_done`, `ground_index`.
**Entry:** as above. **Imports:** none.

### `persona/kernel/screen_prove.py` (141)
"Not finished until the screen shows it" — made mechanical. `screen_proves`,
`latest_screen_text`, `prove_or_wait`, `evidence_terms`.
**Entry:** as above. **Imports:** none.

### `persona/kernel/openai_surface.py` (295)
The dual loop as an OpenAI-compatible endpoint: `split_messages`, `stream_chunks`,
`completion_body`, `trace_of`, `create_app`.
**Entry:** as above. **Imports:** `.caps`, `.dual`.

### `persona/kernel/hosts/__init__.py` (9)
Hosts: adapters that plug the kernel into somebody else's machinery. Exports `.companion`.

### `persona/kernel/hosts/companion.py` (279)
Host adapter: the companion app's `Mind` → the kernel's capabilities. `RecallMemory`,
`HostTools`, `MeterVerify`, `TwoSidedInsight`, `bundle_for_companion`, `correction`.
**Entry:** as above. **Imports:** `...brains`, `...brains.voice_a`, `..caps`.

---

## `persona/brains/` — the lobes

### `persona/brains/voice_a.py` (381)
**A — the voice.** Streams to the person, uses tools mid-sentence. `FeelFilter`/`strip_feel`
(emotional chord declarations), boundary helpers (`at_boundary`, `first_boundary`,
`last_boundary`), `inner_voice`, `VoiceResult`, `speak`.
**Entry:** as above. **Imports:** `..llm`, `.prompts`.

### `persona/brains/adjutant.py` (78)
**B — the adjutant (operations).** Runs after A answers, never blocks the person.
`review`, `transcript`.
**Entry:** as above. **Imports:** `..llm`, `..timesense`, `.prompts`.

### `persona/brains/observer.py` (54)
B — the psyche: the feeling, perceptive half of the mind. `read_psyche`.
**Entry:** `read_psyche`. **Imports:** `..llm`, `..store`, `.adjutant`, `.prompts`.

### `persona/brains/verifier.py` (69)
Thin async bridge around the meter module. `verify`, `meter_dict`.
**Entry:** as above. **Imports:** `..llm`, `.meter`.

### `persona/brains/meter.py` (1110)
The saved full verifier prompt shared by whichever lobe is not speaking, plus the verdict
engine and renderers. `Meter`, `Basis`, `compute_meter`, `rate`, `review_with_retry`,
`parse_review`, `pending`, `unrated`, `carryforward_for_a`, `strip_meter`, and
`render_html/markdown/text/ansi`, `to_header`, `summary_line`, `detail_lines`, `legacy_view`,
`footer`.
**Entry:** as above. **Imports:** none.

### `persona/brains/live.py` (50)
The other lobes, live: what B and C do *while* A is still speaking. `gut_read`, `live_check`.
**Entry:** as above. **Imports:** `..llm`, `.prompts`.

### `persona/brains/consults.py` (88)
Parallel minds: the moment a question arrives, consultant models start.
`Consult`, `ConsultBoard`.
**Entry:** as above. **Imports:** `..llm`, `.prompts`.

### `persona/brains/rememberer.py` (117)
B as the rememberer: a second mind that searches the past while the first one talks.
`remember`, `context_for`, `Recollection`.
**Entry:** as above. **Imports:** `..llm`, `..memory.recall`, `..timesense`, `.prompts`.

### `persona/brains/prompts.py` (410)
**Every prompt in the system, in one place.** No defs; import `..emotion`.

---

## `persona/memory/`

### `persona/memory/records.py` (453)
The memory store: one record per event, three authored layers, plus entities/loops/rules.
`Memory`, `content_words`, `polarity`, `parse_due`, `vec_matrix`.
**Entry:** as above. **Imports:** `..embedder`, `..store`, `..timesense`, `.grounding`.

### `persona/memory/recall.py` (386)
Recall: broad in, ranked out, stitched, labeled, dated. `Recall`, `RecallResult`, `render_event`,
`format_similarity_alerts`. Loose band `RECALL_BAND` (0.30); "open sesame" near-match named
unasked at 0.78.
**Entry:** as above. **Imports:** `..embedder`, `..store`, `..timesense`, `.records`.

### `persona/memory/grounding.py` (75)
**Quote-or-die: nothing is attributed to the user unless their exact words exist.**
`ground`, `_clean_quote`.
**Entry:** as above. **Imports:** none.

### `persona/memory/search.py` (94)
One retrieval entrypoint: semantic + lexical long-term memory, document passages, journal.
`search_all`, `_terms`, `_journal`.
**Entry:** as above. **Imports:** `..knowledge`, `.records`.

### `persona/memory/evaporate.py` (135)
The overflowing cup: what the mind has already absorbed, it may stop carrying. `evaporate`,
`occupancy`, `candidates`, `gist_of`, `trace_of`. Raw cup `CUP_CHARS` (400k).
**Entry:** as above. **Imports:** `..store`.

### `persona/memory/__init__.py` (1)
Package marker.

---

## `persona/life/` — everything slow, never on a turn

### `persona/life/sleep.py` (269)
Sleep: the nightly circadian pass. `Sleep`.
**Entry:** `Sleep`, `_strike`. **Imports:** `..brains.prompts`, `..llm`, `..memory`, `..profiles`, `..store`, `..timesense`, `.finetune`.

### `persona/life/subconscious.py` (186)
The subconscious buffer (ND-SMC / CSMC): keep the day's raw "juice". `Subconscious`,
`chord_distance`, `style_score`, `style_markers`.
**Entry:** as above. **Imports:** `..embedder`, `..store`.

### `persona/life/initiative.py` (283)
Initiative: the persona is always on and speaks first when it has a reason (heartbeat, quiet
hours, outreach caps). `Initiative`.
**Entry:** `Initiative`. **Imports:** `..brains`, `..brains.prompts`, `..channels.adapters`, `..identity`, `..llm`, `..profiles`, `..store`, `..timesense`.

### `persona/life/weights.py` (370)
The two copies she speaks from (AL fast / AR slow). `WeightStore`, `Learner`, `Backend`,
`ToyBackend`, `backend_from_env`, `LearnResult`, `memory_texts`, `fact_texts`, `cool_toward_base`,
`sync`, `gap`, `norm`, `cap`, `zeros_like`.
**Entry:** as above. **Imports:** none.

### `persona/life/update.py` (150)
Aging: the per-turn weight update. `Updater`, `reward_for`, `strategy_feedback`. Honest no-op
when `UPDATE_URL` is empty.
**Entry:** as above. **Imports:** `..store`.

### `persona/life/fastweight.py` (156)
Fast-weight delta: per-turn learning via L/R divergence. `FastWeightDelta`,
`LRDivergenceMeasurer`, `EMASynchronizer`.
**Entry:** as above. **Imports:** `..store`.

### `persona/life/finetune.py` (299)
The dream: bake the day's juice into A's weights (plug and play). `FineTuner`, `compress_text`.
**Entry:** as above. **Imports:** `..brains.prompts`, `..config`, `..llm`, `..profiles`, `..store`, `.subconscious`.

### `persona/life/anchor.py` (78)
The anchor: the wrongness dataset C keeps, and its rehearsal during sleep. `wrong_cases`,
`anchor_run`.
**Entry:** as above. **Imports:** `..brains`, `..store`.

### `persona/life/character_anchor.py` (247)
Character anchor — the tonal counterpart to the factual wrongness anchor.
`recognize_tonal_correction`, `maybe_record_correction`, `cases`, `character_anchor_run`.
**Entry:** as above. **Imports:** `..brains`, `..store`.

### `persona/life/speaking.py` (172)
She speaks from AL and AR, and the step writes to those same two copies. `Speaking`.
**Entry:** `Speaking`. **Imports:** `.mouth`, `.weights`.

### `persona/life/mouth.py` (50)
The mouth that can change and still bill per token. `is_coreweave`, `artifact_ref`,
`publish_adapter`.
**Entry:** as above. **Imports:** none.

### `persona/life/__init__.py` (1)
Package marker.

---

## `persona/voice/`

### `persona/voice/realtime.py` (1119)
Realtime voice — **the lag budget is everything**. `VoiceSession`, `SpeechIO`, `SentenceChunker`,
`tts_body`, `open_ended`, `parse_frame`, `_clause_cut`.
**Entry:** as above. **Imports:** `..config`, `.native_input`.

### `persona/voice/duplex.py` (76)
Latest-audio listener + responder preparation, isolated from task execution. `DuplexVoice`.
**Entry:** `DuplexVoice`. **Imports:** `.native_input`.

### `persona/voice/native_input.py` (166)
Bounded original-audio evidence — **a recording is not a user quotation**. `NativeAudio`,
`decode_native_audio`, `AudioReviewer`, `AudioAttachedLLM`.
**Entry:** as above. **Imports:** none.

### `persona/voice/phone_stream.py` (394)
Bidirectional Twilio Media Streams bridge for the companion phone surface. `TwilioMediaStream`,
mulaw↔PCM helpers, `_PhoneSpeech`, `_native_r_listen`.
**Entry:** as above. **Imports:** `.native_input`, `.realtime`.

### `persona/voice/calls.py` (788)
Twilio phone-call surface — Sawii places and receives real calls. `CallHub`, `CallSession`,
`CallEvent`, `TwilioConfig`, TwiML helpers, `register_call_routes`, `calls_*` routes,
`twilio_voice/response/status`, `CallStream`.
**Entry:** as above. **Imports:** none.

### `persona/voice/clone.py` (79)
A voice the conversation layer speaks in: save/load/status/clear.
**Entry:** as above. **Imports:** none.

### `persona/voice/tinker_audio.py` (47)
Legacy opt-in transcription helper; normal native turns never call this. `native_audio_part`,
`transcribe_native`.
**Entry:** as above. **Imports:** `.native_input`.

---

## `persona/scholar/` — Arabic corpus and deterministic fiqh arithmetic

### `persona/scholar/normalize.py` (180)
Making Arabic searchable without changing a letter: `normalize`, `strip_marks`, `fold_letters`,
`digits_to_latin`, `punct_to_latin`, `tokens`, `stems`, `content_hash`.
**Entry:** as above. **Imports:** none.

### `persona/scholar/robots.py` (97)
Robots.txt, read honestly. `Robots`, `_compile`.
**Entry:** as above. **Imports:** none.

### `persona/scholar/store.py` (463)
The scholar's corpus tables, in the same SQLite file as everything else. `ScholarStore`,
`open_store`, `_needs_review`, `_match_terms`.
**Entry:** as above. **Imports:** `.normalize`.

### `persona/scholar/calc/mirath.py` (358)
Mirath (mawarith) — shares of an estate, in **exact fractions** (`fractions.Fraction`):
`distribute`, `per_head`, `_fards`, `_hajb`, `_umariyya`, `_apply_awl`, `_give_residue`,
`_radd`, `_validate`, `_assert_sums`. `MirathError`.
**Entry:** `distribute`, `per_head`, `MirathError`. **Imports:** none.

### `persona/scholar/crawl/base.py` (319)
The polite core shared by the three harvests: robots first, per-host pacing, resume journal.
`Harvester`, `Policy`, `Fetched`, `Journal`, `base_type`, `accepts`, `human`, `suffix`.
**Entry:** as above. **Imports:** `..robots`, `.urls`.

### `persona/scholar/crawl/urls.py` (77)
Urls, journal, policy — boring on purpose: resumability is one append-only file.
`normalize_url`, `host_of`, `sha256`, `decode_html`, `now`.
**Entry:** as above. **Imports:** none.

### `persona/scholar/crawl/fatwas.py` (365)
Fatwa harvest: seed/list pages → fatwa pages → (question, answer) records. `FatwaRecord`,
`parse_fatwa`, `split_qa`, `extract_fatwas`, `find_links`, `next_pages`, `crawl_fatwa_site`,
`by_ruling`, `is_wp_post`, `is_fatwa_link`.
**Entry:** as above. **Imports:** `..normalize`, `.base`, `.urls`.

### `persona/scholar/crawl/books.py` (118)
Book harvest: a list of PDF URLs → stored PDFs, plus what they are. `BookSpec`, `load_book_list`,
`read_book`, `crawl_books`, `_pdfminer_text`.
**Entry:** as above. **Imports:** `..normalize`, `.base`.

### `persona/scholar/crawl/videos.py` (133)
Video harvest: a channel/playlist → stored audio + per-video journal. `VideoItem`, `list_videos`,
`harvest_videos`, `_download_audio`, `_have_video`.
**Entry:** as above. **Imports:** `.base`, `.urls`.

### `persona/scholar/crawl/__main__.py` (138)
CLI: one polite run of the whole harvest. `python -m sawii_persona.scholar.crawl`.
**Entry:** `main`, `_harvest`, `_split`, `_load_env_file`.
**Imports:** `.base`, `.fatwas`.

### `persona/scholar/crawl/__init__.py` (30) / `persona/scholar/__init__.py` (9) / `persona/scholar/calc/__init__.py` (18)
Package markers with docstrings defining the scholar's subsystems and the owner's fiqh ruling as code.

---

## `persona/gateway/` — the home layer (accounts, sessions, browser pane)

### `persona/gateway/__init__.py` (37)
*"Gateway — the home layer around Sawii's mind."* Exports `.auth`, `.db`, `.middleware`,
`.personas`, `.routes`.

### `persona/gateway/auth.py` (229)
Accounts and bearer sessions: bcrypt password hashing, bearer tokens, registration/login,
rate limiting, `require_user`. `AuthError`, `hash_password`, `verify_password`, `create_session`,
`user_for_token`, `revoke_token`, `revoke_all`, `register`, `login`, `rate_limit`,
`bearer_from_request`, `require_user`, `validate_credentials`, `constant_eq`.
**Entry:** as above. **Imports:** `.db`.

### `persona/gateway/db.py` (127)
SQLite persistence for the gateway (home layer). `DB`, `get_db`, `reset_db_for_tests`,
`set_db_for_tests`, `now`.
**Entry:** as above. **Imports:** none.

### `persona/gateway/middleware.py` (71)
Request-level middleware for the gateway: request-id, structured JSON logs.
`RequestIdMiddleware`, `safe_header_dump`.
**Entry:** as above. **Imports:** none.

### `persona/gateway/personas.py` (106)
Per-user persona minting (A2) — provision a persona on account creation.
`provision_persona_for_user`, `get_persona_for_user`, `load_seed`, `reset_seed_cache_for_tests`.
**Entry:** as above. **Imports:** `..identity`, `.db`.

### `persona/gateway/routes.py` (206)
Gateway routes — the `/api/*` surface the companion UI talks to. `/api/auth/register|login|me|logout`,
`/api/persona`. `register_gateway_routes`, `auth_register`, `auth_login`, `auth_me`,
`auth_logout`, `persona_me`.
**Entry:** as above. **Imports:** `.auth`, `.middleware`, `.personas`.

### `persona/gateway/browser/__init__.py` (35)
Browser gateway — Shape B wrap of `browser_use` for the companion app. Exports `.insight`,
`.routes`, `.session`, `.stream`, `.tools`.

### `persona/gateway/browser/session.py` (410)
Browser session contract and explicit test doubles. `BrowserSessionProto`, `Snapshot`,
`read_page`, `DryRunSession`, `RealSession`, `PlaywrightSession`, `create_browser_session`,
`_chrome_path`.
**Entry:** as above. **Imports:** none.

### `persona/gateway/browser/tools.py` (210)
`BrowserTools` — a `ToolsCap` wrapping the browser session. `schemas(mode)` is asymmetric:
**L** gets action tools (`browser_navigate/click/type/scroll/back/wait_for_selector`) plus
observation; **R** gets observation only (`browser_snapshot`).
**Entry:** `BrowserTools`, `_render_snapshot`. **Imports:** `.session`.

### `persona/gateway/browser/insight.py` (58)
`BrowserPaneInsight` — an `InsightCap(mode="L")` that reads the page.
**Entry:** `BrowserPaneInsight`. **Imports:** none.

### `persona/gateway/browser/stream.py` (110)
Screenshot broadcasting — one producer, many WebSocket subscribers. `BrowserBroadcast`.
**Entry:** `BrowserBroadcast`. **Imports:** `.session`.

### `persona/gateway/browser/tasks.py` (143)
Persona-scoped background Browser-Use tasks; L chat never awaits their run().
`BrowserTasks`, `MindBrowserLanes`, `task_schemas`.
**Entry:** as above. **Imports:** none.

### `persona/gateway/browser/routes.py` (316)
HTTP + WebSocket routes for the browser pane. `BrowserHub`, `BrowserStream`, `browser_state`,
`browser_action`, `browser_snapshot`, `browser_task`, `register_browser_routes`.
**Entry:** as above. **Imports:** `.session`, `.stream`, `.tools`.

### `persona/gateway/browser/edited.py` (118)
Companion session backed by the exact edited Browser-Use SDK from Railway.
`EditedBrowserSession`.
**Entry:** `EditedBrowserSession`. **Imports:** `.session`.

---

## `persona/api/` — HTTP / WebSocket surface

### `persona/api/app.py` (1568)
The app's HTTP/WebSocket surface: Starlette/FastAPI app `create_app`/`main_app`, `Runtime`
(holds the mind + subsystems), auth via `persona_by_key`, `/v1/chat/completions`,
`/v1/responses`, `/v1/hands` WS, voice WS, gateway mount, rate limiting.
**Entry:** `create_app`, `main_app`, `Runtime`, `RateLimiter`, `_bearer`, `_client_tools`, `_lobes`, `_surface`, `_channel`.
**Imports:** `..channels.adapters`, `..config`, `..lanes.deep`, `..life`, `..life.finetune`, `..life.initiative`, `..life.sleep`, `..mind`, `..profiles`, `..store`, `..testing`, `..timesense`, `..voice.realtime`.

### `persona/api/responses.py` (278)
OpenAI Responses API adapter backed by Persona's real dual-lobe chat kernel.
`responses_endpoint`, `from_responses_request`, `from_chat_response`, `from_chat_stream`,
`AdaptedRequest`, `_content`, `_text_item`, `_call_item`, `_event`, `_base_response`.
**Entry:** as above. **Imports:** none.

### `persona/api/__init__.py` (1)
Package marker.

---

## `persona/lanes/`

### `persona/lanes/deep.py` (192)
The workshop (deep lane): heavy multi-step work runs here, in the background.
`DeepLane`, `LLMWorker`, `OpenHandsWorker`, `workspace_for`, `attach`.
**Entry:** as above. **Imports:** `..brains.prompts`, `..llm`, `..store`.

---

## `persona/channels/`

### `persona/channels/adapters.py` (121)
Every way a human can be reached: app, SMS, WhatsApp, phone call, email, Telegram.
`Channels`, `DeliveryError`, `twilio_signature_ok`.
**Entry:** as above. **Imports:** `..config`, `..store`.

---

## `persona/testing/`

### `persona/testing/runner.py` (113)
The test harness (READY-PLAN Step 6): real turns, real numbers. `run_test_turn`,
`run_benchmark`, `TestTurnResult`, `_stats`, `_weights`.
**Entry:** as above. **Imports:** none.

### `persona/testing/latency.py` (99)
Latency measurement utilities for the testing interface. `LatencyBreakdown`, `Timer`, `measure`.
**Entry:** as above. **Imports:** none.

### `persona/testing/__init__.py` (6)
Testing interface docstring; exports `.runner`.

---

## Training modules (top-level)

### `persona/training_bundle.py` (569)
Offline, provenance-preserving assembly; never a trainer or rights attestation.
`assemble`, `verify_bundle`, `check_leakage`, `authored_examples`, `dynamic_messages`,
`validate_dynamic`, `admitted_examples`, `launcher_partitions`, `write_bundle`, `encoded`,
`jsonl`, `read_rows`, `BundleError`.
**Entry:** as above. **Imports:** `.tajweed_training`.

### `persona/tajweed_training.py` (280)
Role-preserving, opt-in Tajweed training data contracts. `plan_training`, `validate_sample`,
`choose_lobe_models`, `reference_sample`, `split_for_group`, `TrainingDataError`.
**Entry:** as above. **Imports:** none.

### `persona/agentic_training.py` (180)
Evidence-gated browser/desktop and L/R coordination training. `validate_execution_evidence`,
`coordination_report`, `_reject`, `_tool_messages`, `_is_failure`.
**Entry:** as above. **Imports:** none.

### `persona/connected_policy.py` (275)
Goal-conditioned, jointly-rewarded L/R coordination policy. `GoalConditionedPolicy`,
`LearnedCoordinator`, `TeamChoice`, `train_verified_goal_policy`.
**Entry:** as above. **Imports:** none.

### `persona/connected_training.py` (147)
Actual connected L/R/C online episode evaluator. `run_connected_episode`, `team_reward`,
`JointOutcome`, `digest`.
**Entry:** as above. **Imports:** none.

---

## Completeness check

Modules explicitly required by the brief are all covered above: `link.py`, `mind.py`,
`inference_lanes.py`, `host_lanes.py`, `host_delegation.py`, `hands.py`, `store.py`,
`identity.py`, `timesense.py`, `shared_timeline.py`, `sketch_graph.py`, `decision_map.py`,
`execution_blueprint.py`, `context.py`, `conversation_board.py`, `cursor_bridge.py`,
`desktop.py`, `computer.py`, `config.py`, `plugins.py`, `profiles.py`, `products.py`,
`subsystems.py`, `emotion.py`, `perception.py`, `picture.py`, `picture_board.py`,
`project_orientation.py`, `knowledge.py`, `kv_sessions.py`, `live_commands.py`, `llm.py`,
`llm_tinker.py`, `embedder.py`, `deepinfra_embedding.py`, `gemini_embedding.py`,
`channels/adapters.py`, `lanes/deep.py`, `memory/*`, `api/app.py`, `api/responses.py`,
`gateway/*` (incl. `gateway/browser/*`), `kernel/*`, `brains/*`, `life/*`, `voice/*`,
`testing/*`, `scholar/*`, training modules.

> **Cleanup note:** the repo carries committed `persona/**/__pycache__/*.pyc`. Some bytecode
> references modules with no `.py` at this commit (e.g. `world_map`, `architecture/*`,
> `brains/director`) — stale artifacts, not source. Only `.py` files are documented here.
