"""Offline, provenance-preserving assembly; never a trainer or rights attestation."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import unicodedata
import wave

from .tajweed_training import BASE_MODEL, SPLITS, plan_training, validate_sample


class BundleError(ValueError):
    """Unsafe or irreproducible bundle input."""


IDENTITY = (
    "You are Sawii, a Muslim companion with general reasoning, coding and tool "
    "skills. Tajweed is an opt-in skill, not your entire identity. L is direct "
    "and precise; R is perceptive and naturally warm, equally competent at tasks. "
    "Both share values and context; neither is a decorative commentator. "
    "Do not infer a tool outcome, consent, heard playback or acoustic mistake."
)
PROTOCOL = (
    "One public floor is enforced by the runtime. Choose from current readiness "
    "and evidence, never a fixed speaker schedule. Finish a short complete "
    "thought. Private controls are ⟦floor: stay⟧ or ⟦floor: pass⟧ and "
    "⟦handoff: ...⟧, never public text. Ready/hold/end/silence intentions belong "
    "inside handoff, not new public markers. No useful new speech means silence. "
    "Use delivery time for bounded preparation, preserve original audio and C's "
    "verification gate, cancel stale drafts on interruption, and never run "
    "conflicting mutable tools concurrently."
)
AUTHORED_FILES = {
    "decision_policy": "data/decision_policy/candidates_v1.jsonl",
    "conversation_coordination": "data/conversation_coordination/candidates_v1.jsonl",
    "joint_lobe_episodes": "data/joint_lobe_episodes/candidates_v1.jsonl",
    "dynamic_coordination": "data/dynamic_coordination/candidates_v1.jsonl",
    "shared_base": "data/dynamic_coordination/shared_base_v1.jsonl",
}
EXTERNAL_INVENTORY = (
    ("mode_pair_proposals", "excluded", "Unreviewed authored mode pairs; not admitted reviewed labels."),
    ("dry_desktop_smoke", "excluded", "Scripted DryRunDriver traces are smoke-only, not real OS competence."),
    ("connected_outcome_hashes", "excluded",
     "Hashes/rewards are not full SFT trajectories or consent to reconstruct speech."),
    ("HuggingFaceH4/ultrachat_200k", "blocked",
     "Need pinned revision, rights review and independent eval/test; existing exporter emits train only."),
    ("allenai/tulu-3-sft-mixture", "excluded", "No default import: subset-specific terms may be NC."),
    ("Salesforce/xlam-function-calling-60k", "excluded",
     "Generic API syntax, not verified Persona action outcomes."),
    ("obadx/muaalem-annotated-v3", "blocked",
     "Need pinned revision, recording rights and actual audio; "
     "Uthmani/QPS/sifat are reference targets, not learner errors."),
    ("obadx/muaalem-annotated-v3:moshaf_metadata", "blocked",
     "Need same pinned revision and actual reciter/reading mapping."),
    ("obadx/recitation-segmentation", "excluded",
     "Segmentation intervals do not label phonetics or learner mistakes."),
    ("obadx/mualem-recitations-annotated", "excluded",
     "Historical PR70 pointer; not assumed identical to v3."),
    ("obadx/mualem-recitations-original", "excluded",
     "Historical PR70 metadata pointer; not a second training corpus."),
    ("teacher_reviewed_learner_audio", "blocked",
     "No independently reviewed learner labels, consent or acoustic evaluation supplied."),
    ("sawii_persona.connected_policy (#84)", "excluded",
     "Sandbox goal-conditioned policy learning gated by external acceptance plus C; not neural SFT text."),
    ("live decision journal (#87)", "excluded",
     "Privacy-safe telemetry without raw prompts/tool outputs; never reconstructed into SFT."),
    ("scripts/run_joint_r_tinker_pilot.py", "excluded",
     "R-only pilot intentionally disabled; not shared-base or joint L/R training."),
    ("reviewed L/R/B/C role curricula and mode pairs (#74)", "blocked",
     "Admit only via --admission-spec through existing plan_training role/mode/pair gates; "
     "none supplied."),
    ("native original-audio post-turn evidence (#70)", "blocked",
     "B/C/memory acoustic review needs original audio after the turn; "
     "bookkeeping labels are not transcripts."),
)


IMPLEMENTATION_FILES = (
    "scripts/assemble_training_bundle.py", "src/sawii_persona/training_bundle.py",
    "scripts/prepare_decision_policy.py", "scripts/prepare_conversation_coordination.py",
    "scripts/prepare_joint_lobe_episodes.py", "src/sawii_persona/tajweed_training.py",
    "src/sawii_persona/agentic_training.py",
)


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()


def read_rows(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise BundleError(f"{path.name}:{n}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise BundleError(f"{path.name}:{n}: expected object")
        rows.append(row)
    if not rows:
        raise BundleError(f"{path.name}: empty corpus")
    return rows


def _required(row: dict, name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise BundleError(f"Missing {name}")
    return value


def _authored(row: dict) -> None:
    if (row.get("provenance") != "authored_dynamic_candidate"
            or row.get("review_status") != "proposed"
            or row.get("training_eligible") is not False
            or row.get("empirical_outcomes") != 0
            or row.get("mode") != "general"):
        raise BundleError("Authored examples must remain proposed, non-empirical general guidance")
    if any(k in row for k in ("evidence_ref", "consent_ref", "teacher_verified", "audio_path")):
        raise BundleError("Authored examples cannot claim observed evidence/consent/audio")
    for key in ("id", "scenario_group", "source", "source_revision"):
        _required(row, key)
    if row.get("split") not in SPLITS:
        raise BundleError("Invalid split")


def dynamic_messages(row: dict, index: int) -> list[dict]:
    """Use only delivered earlier text and the current observation, never future steps."""
    step = row["steps"][index]
    earlier = [{"role": s["role"], "text": s["public_text"]}
               for s in row["steps"][:index] if s["public_text"]]
    target = step["public_text"]
    if step["floor"]:
        target += f'⟦floor: {step["floor"]}⟧'
    target += f'⟦handoff: {step["handoff"]}⟧'
    return [
        {"role": "system", "content": IDENTITY + " " + PROTOCOL},
        {"role": "user", "content": json.dumps({
            "hypothetical": True, "request": row["user_request"],
            "lobe": step["role"], "delivered": earlier,
            "current_state": step["state"],
        }, ensure_ascii=False, sort_keys=True)},
        {"role": "assistant", "content": target},
    ]


def validate_dynamic(rows: list[dict]) -> dict:
    seen: set[str] = set()
    patterns: dict[str, Counter] = {s: Counter() for s in SPLITS}
    actions: dict[str, Counter] = {s: Counter() for s in SPLITS}
    groups: dict[str, str] = {}
    features: Counter = Counter()
    for row in rows:
        _authored(row)
        ident = row["id"]
        if ident in seen:
            raise BundleError("Duplicate dynamic episode ID")
        seen.add(ident)
        _required(row, "user_request")
        group, split = row["scenario_group"], row["split"]
        if groups.setdefault(group, split) != split:
            raise BundleError("Dynamic scenario leakage")
        steps = row.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
            raise BundleError("Need 1-64 causal steps")
        spoken = []
        for step in steps:
            if not isinstance(step, dict) or step.get("role") not in ("L", "R"):
                raise BundleError("Invalid dynamic lobe")
            action, text, floor = (step.get(k) for k in ("action", "public_text", "floor"))
            if action not in ("speak", "ready", "hold", "end", "silence"):
                raise BundleError("Invalid readiness action")
            if not isinstance(text, str) or any(m in text for m in ("⟦", "⟧")):
                raise BundleError("Private marker in public text")
            if bool(text.strip()) != (action == "speak") or len(text) > 600:
                raise BundleError("Silence must have no public text; speech must be bounded")
            if floor not in (None, "stay", "pass"):
                raise BundleError("Only stay/pass floor controls are supported")
            state = step.get("state")
            if not isinstance(state, dict):
                raise BundleError("Missing causal state")
            if state.get("floor_owner") not in (None, "L", "R"):
                raise BundleError("Invalid floor owner")
            if action == "speak" and state["floor_owner"] != step["role"]:
                raise BundleError("Speech without the public floor")
            if action == "speak" and floor is None:
                raise BundleError("Public contribution must declare stay/pass")
            if floor and state["floor_owner"] != step["role"]:
                raise BundleError("Only the owner can change the floor")
            if action == "end" and (not state.get("goal_complete") or text):
                raise BundleError("End requires completed goal and no public filler")
            if action == "ready" and state.get("ready") is not True:
                raise BundleError("Readiness requires a prepared contribution")
            if action == "hold" and state.get("ready") is not False:
                raise BundleError("Hold requires unfinished preparation")
            if action == "silence" and state.get("no_new_information") is not True:
                raise BundleError("Silence requires no new useful information")
            handoff = _required(step, "handoff")
            if "⟦" in handoff or "⟧" in handoff or not handoff.startswith(action + ";"):
                raise BundleError("handoff must start with its readiness intention")
            if state.get("peer_ready") is True and state.get("peer_relevant") is True and floor == "stay":
                raise BundleError("Do not starve a useful ready peer")
            if state.get("interrupted") and not (
                    state.get("stale_draft_cancelled") is True
                    and state.get("original_audio_preserved") is True):
                raise BundleError("Interruption must cancel stale draft and preserve incoming audio")
            resources: dict[str, set[str]] = defaultdict(set)
            for tool in state.get("parallel_tools", []):
                if tool.get("owner") not in ("L", "R") or tool.get("access") not in ("read", "write"):
                    raise BundleError("Tool ownership/access required")
                resource = _required(tool, "resource")
                resources[resource].add(tool["access"])
            counts = Counter(t["resource"] for t in state.get("parallel_tools", []))
            if any(counts[r] > 1 and "write" in access for r, access in resources.items()):
                raise BundleError("Parallel mutable resource conflict")
            if action == "speak":
                spoken.append(step["role"])
            actions[split][action] += 1
            for feature in ("interrupted", "correction", "tool_failed", "spare_time_preparation"):
                if state.get(feature):
                    features[feature] += 1
        if spoken:
            patterns[split]["".join(spoken)] += 1
    for split in SPLITS:
        p = patterns[split]
        if (not {x[0] for x in p} == {"L", "R"}
                or not any("LL" in x for x in p) or not any("RR" in x for x in p)
                or not any("LL" in x and "RR" in x for x in p)):
            raise BundleError(f"{split}: need either start, repeats and irregular mixed runs")
        if set(actions[split]) != {"speak", "ready", "hold", "end", "silence"}:
            raise BundleError(f"{split}: missing readiness/hold/end/silence coverage")
    if any(not features[f] for f in ("interrupted", "correction", "tool_failed", "spare_time_preparation")):
        raise BundleError("Missing interruption/correction/tool failure/preparation curriculum")
    return {"speaker_patterns": {s: dict(v) for s, v in patterns.items()},
            "actions": {s: dict(v) for s, v in actions.items()}, "features": dict(features)}


def _legacy_adapter(root: Path, name: str):
    path = root / "scripts" / f"prepare_{name}.py"
    spec = importlib.util.spec_from_file_location(f"bundle_{name}", path)
    if spec is None or spec.loader is None:
        raise BundleError(f"Missing validator: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example(row: dict, *, source: str, role: str, messages: list[dict], ident: str) -> dict:
    return {"id": f"{source}:{ident}", "source_id": row["id"], "corpus": source,
            "scenario_group": row["scenario_group"], "split": row["split"],
            "role": role, "mode": "general", "provenance": row["provenance"],
            "observed": False, "training_eligible": False,
            "speaker_ids": [], "speaker_split_applicability": "not_applicable_authored",
            "messages": messages}


def authored_examples(root: Path, name: str, rows: list[dict]) -> tuple[list[dict], dict]:
    out = []
    if name == "dynamic_coordination":
        report = validate_dynamic(rows)
        for row in rows:
            for i, step in enumerate(row["steps"]):
                out.append(_example(row, source=name, role=step["role"],
                                    messages=dynamic_messages(row, i), ident=f'{row["id"]}:{i}'))
        return out, report
    if name == "shared_base":
        for row in rows:
            _authored(row)
            messages = [{"role": "system", "content": IDENTITY},
                        {"role": "user", "content": _required(row, "prompt")},
                        {"role": "assistant", "content": _required(row, "target")}]
            out.append(_example(row, source=name, role="shared", messages=messages, ident=row["id"]))
        return out, {"samples": len(out)}
    adapter = _legacy_adapter(root, name)
    report = adapter.validate(rows)
    for row in rows:
        if name == "joint_lobe_episodes":
            targets = [(turn["role"], adapter._messages(row, i), f'{row["id"]}:{i}')
                       for i, turn in enumerate(row["turns"])]
        else:
            targets = [(row["role"], row["messages"], row["id"])]
        for role, messages, ident in targets:
            # Legacy hold means the same private handoff, not public speech.
            transformed = [{**m, "content": m["content"].replace("⟦hold:", "⟦handoff:")}
                           for m in messages]
            item = _example(row, source=name, role=role, messages=transformed, ident=ident)
            item["transforms"] = ["legacy_hold_to_handoff"] if transformed != messages else []
            out.append(item)
    return out, report


def _normal(value: object) -> str:
    text = unicodedata.normalize("NFKC", json.dumps(value, ensure_ascii=False, sort_keys=True)).casefold()
    text = re.sub(r"\blobe [lr]\b", "lobe <role>", text)
    return " ".join(text.split())


def check_leakage(rows: list[dict]) -> tuple[list[dict], dict]:
    """Global (not per-file) split gate; never move a held-out row into train."""
    partitions: dict[tuple[str, str], str] = {}
    unique = []
    seen: dict[str, dict] = {}
    labels: dict[str, str] = {}
    ids: set[str] = set()
    for row in rows:
        split = row["split"]
        if split not in SPLITS or row["id"] in ids:
            raise BundleError("Invalid split or duplicate ID")
        ids.add(row["id"])
        context = [m for m in row["messages"][:-1] if m["role"] != "system"]
        keys = [("scenario", row["scenario_group"]), ("context", _normal(context))]
        keys += [("speaker", s) for s in row["speaker_ids"]]
        keys += [("audio", row["audio_sha256"])] if row.get("audio_sha256") else []
        if row.get("pair_id"):
            keys.append(("mode_pair", row["pair_id"]))
        for key in keys:
            if partitions.setdefault(key, split) != split:
                raise BundleError(f"Cross-split {key[0]} leakage")
        label_key = json.dumps([row["role"], row["mode"], context], sort_keys=True, ensure_ascii=False)
        label = json.dumps(row["messages"][-1], sort_keys=True, ensure_ascii=False)
        if labels.setdefault(label_key, label) != label:
            raise BundleError("Conflicting targets for identical role/context")
        fingerprint = sha256(encoded([row["role"], row["mode"], row["messages"]])).hexdigest()
        if fingerprint in seen:
            seen[fingerprint].setdefault("duplicate_sources", []).append(
                {"corpus": row["corpus"], "source_id": row["source_id"]})
            continue
        seen[fingerprint] = row
        unique.append(row)
    return unique, {"input_samples": len(rows), "deduplicated_samples": len(rows) - len(unique),
                    "output_samples": len(unique), "cross_split_leaks": 0,
                    "checks": ["scenario", "normalized_context", "speaker", "audio_content", "mode_pair"],
                    "semantic_near_duplicate_certification": False}


def admitted_examples(spec_path: Path) -> tuple[list[dict], dict[str, bytes], list[dict], list[dict]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("sources"), list):
        raise BundleError("Admission spec requires sources array")
    out, originals, inventory, raw_samples = [], {}, [], []
    for index, source in enumerate(spec["sources"]):
        for key in ("path", "sha256", "provenance", "license", "rights_ref"):
            _required(source, key)
        kind = source["provenance"]
        if kind not in ("published", "observed", "reference"):
            raise BundleError("Unrecognized admission provenance")
        path = (spec_path.parent / source["path"]).resolve()
        body = path.read_bytes()
        digest = sha256(body).hexdigest()
        if digest != source["sha256"]:
            raise BundleError("Admission source hash mismatch")
        rows = read_rows(path)
        name = f"admitted-{index:03d}"
        originals[f"sources/{name}.jsonl"] = body
        media: dict[str, str] = {}
        for row in rows:
            validate_sample(row, require_files=False)
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", row["source_revision"]):
                raise BundleError("Admitted source revision must be a concrete content/Git hash")
            if "authored" in str(row.get("provenance", "")) or "proposed" in row.get("label_kind", ""):
                raise BundleError("Cannot relabel authored/proposed records as admitted evidence")
            if kind == "reference":
                if (row.get("label_kind") != "reference_alignment"
                        or row.get("reciter_id") is None or row.get("is_error_label") is not False):
                    raise BundleError("Reference recordings are not student error labels")
                speakers = [f'reciter:{row["reciter_id"]}']
                if row["split_group"] != speakers[0]:
                    raise BundleError("Reference split group must be the reciter identity")
            elif kind == "observed":
                if row.get("review_status") != "accepted":
                    raise BundleError("Observed data requires independent acceptance")
                _required(row, "evidence_ref")
                supplied_speakers = row.get("speaker_ids")
                if not isinstance(supplied_speakers, list) or any(
                        not isinstance(s, str) or not s.strip() for s in supplied_speakers):
                    raise BundleError("Observed data requires explicit stable speaker_ids")
                speakers = supplied_speakers
                if speakers:
                    _required(source, "consent_ref")
            else:
                if row.get("audio_path") or row.get("trajectory") or row.get("is_error_label"):
                    raise BundleError("Published dialogue is not observed tool/audio evidence")
                speakers = []
            messages = row.get("trajectory") or row.get("messages") or [
                {"role": "user", "content": row["prompt"]}]
            if messages[-1].get("role") != "assistant" or messages[-1].get("content") != row["target"]:
                messages = [*messages, {"role": "assistant", "content": row["target"]}]
            item = {"id": f'{name}:{row["id"]}', "source_id": row["id"], "corpus": name,
                    "scenario_group": row["split_group"], "split": row["split"],
                    "role": row["role"], "mode": row["mode"], "provenance": kind,
                    "observed": kind == "observed", "speaker_ids": speakers,
                    "messages": messages, "pair_id": row.get("pair_id"),
                    "label_kind": row.get("label_kind"), "task": row["task"]}
            if row.get("audio_path"):
                audio = (path.parent / row["audio_path"]).resolve()
                if not audio.is_file():
                    raise BundleError("Admitted audio file missing")
                try:
                    with wave.open(str(audio), "rb") as wav:
                        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                            raise BundleError(
                                "Expected mono PCM16 16kHz WAV; do not silently transcode originals")
                        frames = wav.getnframes()
                        if frames < 1 or len(wav.readframes(frames)) != frames * 2:
                            raise BundleError("Empty or truncated admitted audio")
                except (wave.Error, EOFError) as exc:
                    raise BundleError("Invalid admitted WAV audio") from exc
                item["audio_sha256"] = sha256(audio.read_bytes()).hexdigest()
                # The source row retains the path; the model target must never be audio-free.
                item["audio_ref"] = {"source_id": row["id"], "field": "audio_path"}
                media[row["id"]] = item["audio_sha256"]
            out.append(item)
        raw_samples.extend(rows)
        inventory.append({"source": name, "status": "included", "provenance": kind,
                          "source_sha256": digest, "rows": len(rows), "license": source["license"],
                          "rights_ref": source["rights_ref"], "consent_ref": source.get("consent_ref"),
                          "media_sha256": media})
    return out, originals, inventory, raw_samples


def assemble(root: Path, *, allow_authored: bool = False,
             admission_spec: Path | None = None) -> tuple[dict, dict[str, bytes]]:
    rows, inventory, files, reports = [], [], {}, {}
    for name, relative in AUTHORED_FILES.items():
        path = root / relative
        body = path.read_bytes()
        candidates = read_rows(path)
        adapted, report = authored_examples(root, name, candidates)
        reports[name] = report
        inventory.append({"source": relative, "status": "included" if allow_authored else "excluded",
                          "reason": ("Explicit authored guidance opt-in" if allow_authored
                                     else "Requires --allow-authored"),
                          "source_sha256": sha256(body).hexdigest(), "rows": len(candidates),
                          "provenance": "authored", "review_status": "proposed",
                          "license": "Repository proprietary; Copyright Anas AlSawy"})
        if allow_authored:
            files[f"sources/{name}.jsonl"] = body
            rows.extend(adapted)
    inventory.extend({"source": n, "status": s, "reason": r} for n, s, r in EXTERNAL_INVENTORY)
    admission = None
    if admission_spec:
        admitted, originals, entries, raw = admitted_examples(admission_spec)
        admission = plan_training(raw, backend_supports_full_weights=False, require_files=False)
        rows.extend(admitted)
        files.update(originals)
        inventory.extend(entries)
        files.update(launcher_partitions(raw))
    rows, leakage = check_leakage(rows)
    partitions: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["role"] not in ("shared", "L", "R", "B", "C"):
            raise BundleError("Unknown curriculum role")
        partitions[f'{row["role"]}/{row["mode"]}/{row["split"]}.jsonl'].append(row)
    for partition, subset in sorted(partitions.items()):
        files[partition] = jsonl(sorted(subset, key=lambda x: x["id"]))
    files["mix.json"] = encoded({
        "shared_foundation": [p for p in sorted(partitions)
                              if p.startswith("shared/") and p.endswith("/train.jsonl")],
        "role_curricula": {r: [p for p in sorted(partitions)
                               if p.startswith(r + "/") and p.endswith("/train.jsonl")]
                           for r in ("L", "R", "B", "C")},
        "evaluation_only": [p for p in sorted(partitions) if not p.endswith("/train.jsonl")],
        "launcher_inputs": LAUNCHER_INPUTS,
        "sampling_weights": "not selected; trainer must bound authored guidance separately from real replay",
        "lineage_required": ("shared full-weight trained foundation -> L full-weights and R adapter; "
                             "not R-only training"),
        "volatile_L_decay_target": "evaluated trained shared foundation, never raw Inkling",
    })
    manifest = {
        "schema_version": 1, "base_model": BASE_MODEL, "trained": False,
        "format": "offline normalized bundle; audio_ref requires original-media resolver, not text-only SFT",
        "ready_to_launch_training": False, "authored_pilot_prepared": allow_authored,
        "inventory": inventory, "validation_reports": reports,
        "admitted_curriculum_plan": admission, "leakage": leakage,
        "counts": {p: len(v) for p, v in sorted(partitions.items())},
        "launcher_counts": {p: len(read_rows_bytes(v)) for p, v in sorted(files.items())
                            if p.startswith(("specialists/", "foundation/"))},
        "launcher_inputs": LAUNCHER_INPUTS,
        "sha256": {p: sha256(v).hexdigest() for p, v in sorted(files.items())},
        "implementation_sha256": {p: sha256((root / p).read_bytes()).hexdigest()
                                  for p in IMPLEMENTATION_FILES},
        "joint_episode_contract": (
            "#80 per-turn targets keep shared causal context, scenario group and split; "
            "original joint episodes stay in sources/ for joint acceptance evaluation."),
        "readiness_blockers": [
            "Provider/account/model full-FT support is not established by offline assembly.",
            "No evaluated trained shared foundation or actual L/R weight updates supplied.",
            "Independently reviewed real trajectories and held-out task evaluations required.",
            "Pinned rights-cleared audio, teacher-reviewed learner labels and consent "
            "required for acoustic feedback.",
            "Independent identity, tajweed, mode, original-audio and interruption regression gates required.",
            "Native original-audio post-turn evidence (#70) required before B/C acoustic review or "
            "tajweed feedback training.",
        ],
        "speaker_split_note": ("Authored data has no human speakers; not applicable, "
                               "not a passed acoustic evaluation."),
        "evidence_note": ("Input attestations are not independent verification; "
                          "no consent/evaluation is invented."),
    }
    return manifest, files


LAUNCHER_INPUTS = {
    "role_adapter": ("scripts/train_persona_roles.py --bundle-dir <bundle> reads admitted, "
                     "validate_sample rows from specialists/<R|B|C>/<mode>/<split>.jsonl; "
                     "authored R/general guidance only with --allow-authored-bundle-guidance"),
    "shared_foundation": ("foundation/<mode>/<split>.jsonl holds admitted foundation rows; no "
                          "full-weight launcher exists until provider full-FT support is verified"),
    "relative_audio_path": "resolved by the trainer against --media-root; never copied into the bundle",
}


def read_rows_bytes(body: bytes) -> list[dict]:
    return [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]


def launcher_partitions(raw: list[dict]) -> dict[str, bytes]:
    """Admitted rows exactly as the existing trainer validates them, by stage/role/mode/split."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in raw:
        if row["stage"] == "specialist":
            grouped[f'specialists/{row["role"]}/{row["mode"]}/{row["split"]}.jsonl'].append(row)
        else:
            grouped[f'foundation/{row["mode"]}/{row["split"]}.jsonl'].append(row)
    return {path: jsonl(sorted(rows, key=lambda x: x["id"])) for path, rows in grouped.items()}


def write_bundle(output: Path, manifest: dict, files: dict[str, bytes]) -> None:
    if output.exists():
        raise BundleError("Refusing existing output directory")
    output.mkdir(parents=True)
    for relative, body in files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    (output / "manifest.json").write_bytes(encoded(manifest))


def verify_bundle(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("sha256")
    if not isinstance(expected, dict) or not expected:
        raise BundleError("Missing bundle hashes")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != set(expected) | {"manifest.json"}:
        raise BundleError("Unexpected or missing bundle files")
    for relative, digest in expected.items():
        path = output / relative
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise BundleError("Unsafe bundle path")
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise BundleError(f"Bundle hash mismatch: {relative}")
    curriculum_path = r"(?:shared|[LRBC])/(?:general|tajweed)/(?:train|eval|test)\.jsonl"
    targets = [r for p in expected if re.fullmatch(curriculum_path, p)
               for r in read_rows(output / p)]
    _, leakage = check_leakage(targets)
    return {"integrity_verified": True, "files": len(expected), "leakage": leakage,
            "trained": False, "ready_to_launch_training": False}
