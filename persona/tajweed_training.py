"""Role-preserving, opt-in Tajweed training data contracts.

This module prepares *plans*, not model weights. Training may only start after
dataset rights, speaker-disjoint splits, role coverage and backend support pass.
In particular a reference reciter is never an automatically labeled learner error.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

BASE_MODEL = "thinkingmachines/Inkling-Small"
ROLES = ("L", "R", "B", "C")
MODES = ("general", "tajweed")
ROLE_TASKS = {
    "L": {"general": frozenset({"conversation", "tool_execution", "interruption",
                                     "browser_control", "desktop_control", "lobe_coordination"}),
          "tajweed": frozenset({"recitation_coaching", "live_feedback"})},
    "R": {"general": frozenset({"reasoning", "planning", "tool_execution",
                                     "browser_control", "desktop_control", "lobe_coordination"}),
          "tajweed": frozenset({"rule_reasoning", "qiraat_disambiguation"})},
    "B": {"general": frozenset({"psyche", "memory", "reflection"}),
          "tajweed": frozenset({"learner_progress", "learning_reflection"})},
    "C": {"general": frozenset({"verification", "tool_result_audit"}),
          "tajweed": frozenset({"assessment_audit", "acoustic_verification"})},
}
ACOUSTIC_JUDGMENTS = frozenset({"recitation_coaching", "live_feedback", "acoustic_verification"})
AGENTIC_TRACES = frozenset({"tool_execution", "tool_result_audit", "browser_control", "desktop_control", "lobe_coordination"})
SPLITS = ("train", "eval", "test")


class TrainingDataError(ValueError):
    """An input cannot safely be used for Tajweed training."""


def split_for_group(group: str) -> str:
    """Stable 80/10/10 group-disjoint split; never randomize each verse."""
    if not isinstance(group, str) or not group.strip():
        raise TrainingDataError("A stable reciter or episode split_group is required")
    bucket = int(hashlib.sha256(("tajweed-v1:" + group).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "eval" if bucket < 90 else "test"


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingDataError(f"{field} is required")
    return value.strip()


def reference_sample(row: dict, *, audio_path: str, riwaya: str,
                     revision: str, rights_cleared: bool,
                     target_kind: str = "text") -> dict | None:
    """Normalize a real obadx Muaalem v3 annotated row, not a fabricated Tajweed label.

    The dataset supplies aligned Qur'an transcript/reciter metadata; no field
    proves whether a learner made a pronunciation error.
    """
    if row.get("has_quran") is not True:
        return None
    ratio = row.get("match_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.95 <= ratio <= 1.0:
        return None
    target_map = {
        "text": ("reference_recitation", str(row.get("uthmani") or "").strip()),
        "phonemes": ("reference_phonemes", str(row.get("phonemes") or "").strip()),
        "sifat": ("reference_sifat", json.dumps(row.get("sifat"), ensure_ascii=False)
                  if isinstance(row.get("sifat"), list) and row["sifat"] else ""),
    }
    if target_kind not in target_map:
        raise TrainingDataError("Unknown reference target_kind")
    task, target = target_map[target_kind]
    if not target:
        return None
    reciter = row.get("reciter_id")
    moshaf = _nonempty(str(row.get("moshaf_id") or ""), "moshaf_id")
    if isinstance(reciter, bool) or not isinstance(reciter, int):
        raise TrainingDataError("reciter_id must be an integer")
    reading = _nonempty(riwaya, "riwaya")
    revision = _nonempty(revision, "dataset revision")
    audio_path = _nonempty(audio_path, "audio_path")
    segment = _nonempty(str(row.get("segment_index") or ""), "segment_index")
    group = f"reciter:{reciter}"
    return {
        "id": hashlib.sha256(f"{revision}|{moshaf}|{segment}|{target_kind}".encode()).hexdigest()[:24],
        "stage": "foundation", "mode": "tajweed", "role": "shared",
        "task": task,
        "prompt": {
            "text": f"Listen to this Quran recitation in the reading {reading}. Transcribe the audible Quranic words only.",
            "phonemes": f"Listen to this Quran recitation in the reading {reading}. Return the Quran Phonetic Script (QPS) of the provided reference audio.",
            "sifat": f"Listen to this Quran recitation in the reading {reading}. Return the reference phonetic articulation (sifat) annotations as JSON.",
        }[target_kind],
        "target": target, "audio_path": audio_path,
        "riwaya": reading, "reciter_id": reciter, "moshaf_id": moshaf,
        "segment_index": segment, "split_group": group, "split": split_for_group(group),
        "source": "obadx/muaalem-annotated-v3",
        "source_revision": revision, "rights_cleared": rights_cleared,
        "label_kind": "reference_alignment", "is_error_label": False,
        "annotation_origin": "expert_recitation_text_derived_qps_not_student_errors",
    }


def validate_sample(sample: dict, *, require_files: bool = False) -> None:
    """Fail closed on role leakage, fabricated mistakes and rights ambiguity."""
    if not isinstance(sample, dict):
        raise TrainingDataError("Each sample must be an object")
    for field in ("id", "stage", "mode", "role", "task", "prompt", "target",
                  "split_group", "split", "source", "source_revision"):
        _nonempty(sample.get(field), field)
    if sample["split"] not in SPLITS or sample["split"] != split_for_group(sample["split_group"]):
        raise TrainingDataError("Split must be reproducible and group-disjoint")
    if sample.get("rights_cleared") is not True:
        raise TrainingDataError("Data use rights have not been affirmatively cleared")
    stage, role, mode, task = (sample[k] for k in ("stage", "role", "mode", "task"))
    if stage == "foundation":
        if role != "shared" or (mode, task) not in (
            ("tajweed", "reference_recitation"), ("tajweed", "reference_phonemes"),
            ("tajweed", "reference_sifat"), ("general", "general_agent_replay"),
            ("general", "mode_discrimination"), ("tajweed", "mode_discrimination")
        ):
            raise TrainingDataError("Foundation must contain shared audio competence or normal-agent replay")
    elif stage == "specialist":
        if role not in ROLES or mode not in MODES or task not in ROLE_TASKS[role][mode]:
            raise TrainingDataError("Task does not belong to this lobe and mode")
    else:
        raise TrainingDataError("Unknown training stage")
    if task == "mode_discrimination":
        if stage != "foundation" or role != "shared":
            raise TrainingDataError("Mode discrimination is a shared foundation task")
        _nonempty(sample.get("pair_id"), "mode pair_id")
        _nonempty(sample.get("user_intent"), "explicit user intent")
        if sample.get("expected_mode") != mode:
            raise TrainingDataError("Mode label must match explicit user intent")
        if sample.get("label_kind") != "reviewed_mode_pair":
            raise TrainingDataError("Mode labels require reviewed counterfactual pairs")
    if task in AGENTIC_TRACES:
        # Do not flatten an agent's tool execution to an invented textual answer.
        # Training requires actual function-call / observation pairs.
        trace = sample.get("trajectory")
        if not isinstance(trace, list) or not trace:
            raise TrainingDataError("Agentic sample needs a real tool-call trajectory")
        call_ids = set()
        observed_ids = set()
        for message in trace:
            if not isinstance(message, dict):
                raise TrainingDataError("Invalid trajectory message")
            if message.get("role") == "assistant":
                for call in message.get("tool_calls", []):
                    if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                        raise TrainingDataError("Invalid tool call in trajectory")
                    if call["id"] in call_ids:
                        raise TrainingDataError("Duplicate tool call ID")
                    call_ids.add(call["id"])
            if message.get("role") == "tool":
                observed_ids.add(message.get("tool_call_id"))
        if not call_ids or not observed_ids or call_ids != observed_ids:
            raise TrainingDataError("Each tool call must have its actual matching tool result")
    audio = sample.get("audio_path")
    if task in ("reference_recitation", "reference_phonemes", "reference_sifat"):
        if sample.get("label_kind") != "reference_alignment" or sample.get("is_error_label") is not False:
            raise TrainingDataError("Reference recitation cannot be relabeled as a learner error")
        _nonempty(sample.get("riwaya"), "riwaya")
        _nonempty(audio, "audio_path")
    if task in ("browser_control", "desktop_control", "lobe_coordination"):
        from .agentic_training import validate_execution_evidence
        validate_execution_evidence(sample)
    if task in ACOUSTIC_JUDGMENTS:
        if sample.get("label_kind") != "teacher_verified" or not sample.get("evidence_ref"):
            raise TrainingDataError("Acoustic feedback needs independent teacher-verified evidence")
        _nonempty(sample.get("riwaya"), "riwaya")
        _nonempty(audio, "audio_path")
    if sample.get("is_error_label") is True:
        if stage != "specialist" or mode != "tajweed" or sample.get("label_kind") != "teacher_verified":
            raise TrainingDataError("Learner mistakes require teacher-reviewed specialist labels")
        _nonempty(sample.get("evidence_ref"), "evidence_ref")
    if require_files and audio and not Path(audio).is_file():
        raise TrainingDataError(f"Audio file not found: {audio}")


def plan_training(samples: list[dict], *, backend_supports_full_weights: bool,
                  require_files: bool = False) -> dict:
    """Inspect stage coverage, not a fake 'trained' status.

    Deliberately requires evaluation AND untouched test groups for every stage.
    Specialist Tajweed and general variants are independently trained from the
    shared base, not stacked adapters or a prompt-shaped personality change.
    """
    seen = set()
    grouped: Counter = Counter()
    for sample in samples:
        validate_sample(sample, require_files=require_files)
        if sample["id"] in seen:
            raise TrainingDataError("Duplicate sample ID across stages or splits")
        seen.add(sample["id"])
        key = ("shared" if sample["stage"] == "foundation" else sample["role"],
               sample["mode"], sample["split"])
        grouped[key] += 1

    def stage(role: str, mode: str, weight_type: str) -> dict:
        counts = {s: grouped[role, mode, s] for s in SPLITS}
        missing = [f"{s} data" for s in SPLITS if not counts[s]]
        if weight_type == "full_weights" and not backend_supports_full_weights:
            missing.append("verified full-weight Inkling-Small training backend")
        return {"method": weight_type, "counts": counts,
                "ready": not missing, "blockers": missing}

    # Semantic mode pairs must be split together: a Quran-related software
    # request stays general, while an explicit recitation session is Tajweed.
    # Neither the religion/topic nor the presence of audio alone is a mode trigger.
    mode_pairs: dict[tuple[str, str], set[str]] = {}
    for sample in samples:
        if sample["task"] == "mode_discrimination":
            pair_key = (sample["split"], sample["pair_id"])
            mode_pairs.setdefault(pair_key, set()).add(sample["mode"])
    invalid_pairs = [p for p, modes in mode_pairs.items() if modes != set(MODES)]
    if invalid_pairs:
        raise TrainingDataError("Mode discrimination pairs need both general and Tajweed examples in the same split")
    pair_counts = {split: sum(1 for (s, _), modes in mode_pairs.items()
                               if s == split and modes == set(MODES)) for split in SPLITS}

    shared = {
        "model": BASE_MODEL, "general": stage("shared", "general", "full_weights"),
        "tajweed": stage("shared", "tajweed", "full_weights"),
    }
    reference_tasks = ("reference_recitation", "reference_phonemes", "reference_sifat")
    reference_counts = {kind: {split: sum(
        1 for row in samples if row["stage"] == "foundation"
        and row["task"] == kind and row["split"] == split
    ) for split in SPLITS} for kind in reference_tasks}
    shared["reference_targets"] = reference_counts
    shared["mode_discrimination_pairs"] = pair_counts
    shared["ready"] = shared["general"]["ready"] and shared["tajweed"]["ready"]
    if any(n == 0 for counts in reference_counts.values() for n in counts.values()):
        shared["ready"] = False
        shared["reference_blockers"] = ["Missing Uthmani/QPS/phonetic-sifat reference examples in one or more splits"]
    if any(n == 0 for n in pair_counts.values()):
        shared["ready"] = False
        shared["mode_blockers"] = ["No reviewed general/Tajweed mode pair in one or more train/eval/test splits"]
    variants = {}
    for role in ROLES:
        method = "full_weights" if role == "L" else "adapter"
        variants[role] = {mode: stage(role, mode, method) for mode in MODES}
        for mode in MODES:
            # No promoted role checkpoint without its trained foundation.
            if not shared["ready"]:
                variants[role][mode]["ready"] = False
                variants[role][mode]["blockers"].append("shared trained foundation not ready")
            # The Tajweed variant must retain competence on normal-role examples.
            if mode == "tajweed" and not variants[role]["general"]["ready"]:
                variants[role][mode]["ready"] = False
                variants[role][mode]["blockers"].append("normal-role regression anchor not ready")
    return {
        "base_model": BASE_MODEL, "shared_foundation": shared,
        "variants": variants,
        "ready_to_train": shared["ready"] and all(
            entry["ready"] for role in variants.values() for entry in role.values()
        ),
        "trained": False, "deployed": False,
        "note": "This is an admission plan, not a training run or an evaluation result.",
    }


def choose_lobe_models(default: dict | None, *, recitation_mode: bool,
                       l_checkpoint: str = "", r_checkpoint: str = "",
                       validated: bool = False) -> dict:
    """Keep normal checkpoints untouched; activate both qualified copies atomically.

    B/C are not switched here: acoustic review remains disabled until B/C can
    inspect independently preserved audio evidence. This is NOT training.
    """
    normal = dict(default or {})
    if not recitation_mode or not validated:
        return normal
    if not l_checkpoint.strip() or not r_checkpoint.strip():
        raise TrainingDataError("Validated Tajweed routing needs BOTH L and R checkpoints")
    if l_checkpoint.strip() == r_checkpoint.strip():
        raise TrainingDataError("L and R must have separately trained role checkpoints")
    return {**normal, "L": l_checkpoint.strip(), "R": r_checkpoint.strip()}
