"""The dream: bake the day's juice into A's weights (plug and play).

Order is the safety property:
    1. export verbatim  - juice rows -> archive (ndsmc-raw/2) + chat training file
                          mixed 80/20 with generative replay (the persona rehearsing itself,
                          so one intense day cannot overwrite who it is)
    2. mark exported    - the database records what provably left
    3. train            - provider: fireworks (LoRA SFT) | command (your own trainer) | none
    4. bake             - only on success; a failed night leaves the juice raw for tomorrow
    5. (stubborn memory) baked juice is NOT compressed and NOT evicted - the raw rows are kept forever,
       as what the persona became. The archive in ndsmc-raw/2 remains the byte-exact source.
When training succeeds the new adapter becomes this persona's A model automatically.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from datetime import datetime, timezone

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    import httpx2 as httpx  # type: ignore

from ..brains.prompts import REPLAY
from ..config import Settings
from ..llm import LLM, extract_json
from ..profiles import persona_frame
from ..store import Store, jdump, jload, now
from .subconscious import Subconscious

log = logging.getLogger(__name__)

FIREWORKS = "https://api.fireworks.ai/v1"
DONE_STATES = {"JOB_STATE_COMPLETED"}
FAIL_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_DELETED"}


def compress_text(raw: str, markers: list[str], limit: int = 90) -> str:
    """Lossy, deterministic, local. Keeps the most charged sentence's head + style tags."""
    sentences = re.split(r"(?<=[.!?…])\s+", raw.strip()) or [raw]
    charged = max(sentences, key=lambda s: (sum(c.isupper() for c in s) + s.count("!") + s.count("...")
                                            + len(s) / 40))
    head = charged[:limit] + ("…" if len(charged) > limit else "")
    return (head + (f" [{','.join(markers)}]" if markers else "")).strip()


class FineTuner:
    def __init__(self, settings: Settings, store: Store, subconscious: Subconscious, a_llm: LLM,
                 http: "httpx.AsyncClient | None" = None):
        self.s = settings
        self.store = store
        self.sub = subconscious
        self.a = a_llm
        self._http = http

    def http(self) -> "httpx.AsyncClient":
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=120)
        return self._http

    def _dir(self, pid: int) -> str:
        d = os.path.join(self.s.data_dir, "subconscious", str(pid))
        os.makedirs(d, exist_ok=True)
        return d

    async def replay_examples(self, persona: dict, n: int) -> list[dict]:
        if n <= 0:
            return []
        out: list[dict] = []
        tries = 0
        while len(out) < n and tries < 4:
            tries += 1
            k = min(20, n - len(out))
            res = await self.a.chat([{"role": "system", "content": REPLAY.format(n=k)},
                                     {"role": "user", "content": persona_frame(persona["definition"], persona["name"])}],
                                    temperature=0.9, max_tokens=2500, json_mode=True)
            data = extract_json(res.text) or {}
            for ex in data.get("examples") or []:
                if isinstance(ex, dict) and ex.get("user") and ex.get("assistant"):
                    out.append({"user": str(ex["user"]), "assistant": str(ex["assistant"])})
        return out[:n]

    async def consolidate(self, persona: dict) -> dict:
        pid = persona["id"]
        report: dict = {"persona": pid}
        await self.poll(pid)  # finish anything from previous nights first
        running = await self.store.one("SELECT id FROM finetune_jobs WHERE persona_id = ? AND status IN "
                                       "('pending','running')", pid)
        if running:
            report["status"] = "waiting_on_previous_job"
            return report
        rows = await self.store.all("SELECT * FROM juice WHERE persona_id = ? AND tier IN ('raw','exported') "
                                    "ORDER BY id", pid)
        report["juice"] = len(rows)
        if len(rows) < self.s.finetune_min_examples:
            report["status"] = "not_enough_juice"
            return report
        # 1. export verbatim
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = self._dir(pid)
        archive = os.path.join(d, "archive.jsonl")
        train_path = os.path.join(d, f"train-{stamp}.jsonl")
        system = persona_frame(persona["definition"], persona["name"])
        with open(archive, "a", encoding="utf-8") as fa:
            for r in rows:
                if r["tier"] == "raw":
                    fa.write(json.dumps({"schema": "ndsmc-raw/2",
                                         "timestamp": datetime.fromtimestamp(r["created_at"], timezone.utc).isoformat(),
                                         "raw_transcript": r["raw_transcript"], "reply": r["reply"],
                                         "speaker_style_vectors": jload(r["style_markers"], []),
                                         "emotional_variance_score": r["variance"]}, ensure_ascii=False) + "\n")
        # Adaptive replay ratio — the data-level analogue of EWC within the
        # current P1 (nightly SFT) path. When either anchor's score is weak
        # (near but not below the gate floor), we don't refuse the bake, but
        # we lean more heavily on replay (the persona rehearsing itself) so
        # the new juice can't dominate the next adapter. "Protect the prior
        # shape without silencing the day's growth" — Gemini's EWC intent,
        # expressed in the training mix we have control over.
        #
        # Base ratio ← settings. If (factual_anchor_score < 1.1*floor) OR
        # (character_anchor_score < 1.1*floor), bump replay by 50%. Natural
        # cap at 0.90 so some new juice always lands.
        base_ratio = float(self.s.replay_ratio)
        floor = float(self.s.juice_floor)
        a_score = float(persona.get("anchor_score") or 0.0)
        c_score = float(persona.get("character_anchor_score") or 0.0)
        weak = (0 < a_score < floor * 1.1) or (0 < c_score < floor * 1.1)
        eff_ratio = min(0.90, base_ratio * 1.5) if weak else base_ratio
        report["replay_ratio"] = round(eff_ratio, 3)
        report["replay_ratio_boosted"] = bool(weak)
        n_replay = int(round(len(rows) * eff_ratio / max(0.01, 1 - eff_ratio)))
        replay = await self.replay_examples(persona, n_replay)

        # Split the juice by which gate(s) kept it (per the Subconscious ingest
        # two-gate rule): variance drives the "base" adapter (correctness/
        # persona-base LoRA); shaped_me drives the "char" adapter (character/
        # relationship LoRA). A row kept by both lands in both files. For the
        # P1 serving stack we still submit ONE job (train_path), but the
        # character file (char_path) is written alongside so the P2 two-LoRA
        # serving stack can promote it independently without re-processing
        # the juice table.
        char_path = os.path.join(d, f"train-{stamp}-char.jsonl")
        with open(train_path, "w", encoding="utf-8") as ft, \
                open(char_path, "w", encoding="utf-8") as fc:
            for r in rows:
                if not (r["raw_transcript"] and r["reply"]):
                    continue
                row_msg = {"messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": r["raw_transcript"]},
                                        {"role": "assistant", "content": r["reply"]}]}
                # The base file includes variance-kept rows (and shaped-only
                # rows carried along until the two-LoRA stack lands, so no
                # signal is lost in the interim).
                ft.write(json.dumps(row_msg, ensure_ascii=False) + "\n")
                # The character file carries shaped_me rows plus their chord
                # before/after as metadata the preference-pair trainer can
                # use. The messages shape stays SFT-compatible so a Phase B
                # preference-pair trainer can be swapped in without changing
                # the writer.
                if int(r.get("shaped_me") or 0) == 1:
                    char_msg = dict(row_msg)
                    char_msg["sawii"] = {
                        "chord_delta": float(r.get("chord_delta") or 0),
                        "chord_before": jload(r.get("chord_before") or "{}", {}),
                        "chord_after": jload(r.get("chord_after") or "{}", {}),
                    }
                    fc.write(json.dumps(char_msg, ensure_ascii=False) + "\n")
            for ex in replay:
                ft.write(json.dumps({"messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": ex["user"]},
                                                  {"role": "assistant", "content": ex["assistant"]}]},
                                    ensure_ascii=False) + "\n")
        ids = [r["id"] for r in rows]
        # 2. mark exported
        qs = ",".join("?" for _ in ids)
        await self.store.run(f"UPDATE juice SET tier='exported', exported_at=? WHERE tier='raw' AND id IN ({qs})",
                             now(), *ids)
        job_id = await self.store.insert("finetune_jobs", persona_id=pid, provider=self.s.finetune_provider,
                                         dataset=train_path, juice_ids=jdump(ids), created_at=now(), updated_at=now())
        shaped_count = sum(1 for r in rows if int(r.get("shaped_me") or 0) == 1)
        report.update({
            "exported": len(ids), "replay": len(replay), "dataset": train_path, "job": job_id,
            "char_dataset": char_path, "shaped_me": shaped_count,
        })
        # 3. train
        provider = self.s.finetune_provider
        if provider == "command" and self.s.finetune_command:
            ok, detail = await self._run_command(train_path)
            await self._finish(job_id, pid, ok, detail, output_model="")
            report["status"] = "baked" if ok else "train_failed"
        elif provider == "fireworks" and self.s.fireworks_api_key and self.s.fireworks_account:
            try:
                ref = await self._fireworks_submit(pid, job_id, train_path)
                await self.store.update("finetune_jobs", job_id, job_ref=ref, status="running", updated_at=now())
                report["status"] = "submitted"
                report["job_ref"] = ref
            except Exception as exc:  # noqa: BLE001
                await self.store.update("finetune_jobs", job_id, status="failed", detail=str(exc)[:500],
                                        updated_at=now())
                report["status"] = "submit_failed"
                report["error"] = str(exc)[:300]
        else:
            await self.store.update("finetune_jobs", job_id, status="pending", detail="no trainer configured; "
                                    "dataset exported and waiting", updated_at=now())
            report["status"] = "exported_waiting_for_trainer"
        # stubborn memory: baked juice is kept raw forever; compress() is retained but never called here.
        return report

    async def _run_command(self, path: str) -> tuple[bool, str]:
        if os.name == "nt":
            # shlex.split is POSIX: it eats Windows path backslashes and
            # mangles quoted args. The shell handles both correctly.
            proc = await asyncio.create_subprocess_shell(
                f'{self.s.finetune_command} "{path}"', stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
        else:
            cmd = shlex.split(self.s.finetune_command) + [path]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        return proc.returncode == 0, out.decode(errors="replace")[-1000:]

    async def _finish(self, job_id: int, pid: int, ok: bool, detail: str, output_model: str) -> None:
        job = await self.store.one("SELECT * FROM finetune_jobs WHERE id = ?", job_id)
        await self.store.update("finetune_jobs", job_id, status="succeeded" if ok else "failed",
                                detail=detail[:1000], output_model=output_model, updated_at=now())
        ids = jload(job["juice_ids"], [])
        if ok and ids:
            qs = ",".join("?" for _ in ids)
            await self.store.run(f"UPDATE juice SET tier='baked', baked_at=? WHERE tier='exported' AND id IN ({qs})",
                                 now(), *ids)
            if output_model:  # plug and play: the persona now speaks with its own weights
                await self.store.update("personas", pid, a_model_override=output_model)

    # ---------------------------------------------------------------- fireworks
    def _fw_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.s.fireworks_api_key}"}

    async def _fireworks_submit(self, pid: int, job_id: int, path: str) -> str:
        acct = self.s.fireworks_account
        base = f"{FIREWORKS}/accounts/{acct}"
        dataset_id = f"sawii-p{pid}-j{job_id}-{int(now())}"
        with open(path, "rb") as fh:
            content = fh.read()
        count = content.count(b"\n")
        r = await self.http().post(f"{base}/datasets", headers=self._fw_headers(),
                                   json={"datasetId": dataset_id, "dataset": {"userUploaded": {}, "exampleCount": str(count)}})
        if r.status_code >= 400:
            raise RuntimeError(f"dataset create {r.status_code}: {r.text[:300]}")
        r = await self.http().post(f"{base}/datasets/{dataset_id}:upload", headers=self._fw_headers(),
                                   files={"file": (os.path.basename(path), content, "application/jsonl")})
        if r.status_code >= 400:
            raise RuntimeError(f"dataset upload {r.status_code}: {r.text[:300]}")
        output = f"accounts/{acct}/models/sawii-p{pid}-{int(now())}"
        r = await self.http().post(f"{base}/supervisedFineTuningJobs", headers=self._fw_headers(), json={
            "dataset": f"accounts/{acct}/datasets/{dataset_id}", "baseModel": self.s.fireworks_base_model,
            "outputModel": output, "epochs": 1, "loraRank": 8})
        if r.status_code >= 400:
            raise RuntimeError(f"sft create {r.status_code}: {r.text[:300]}")
        name = r.json().get("name") or ""
        return jdump({"name": name, "output": output})

    async def poll(self, pid: int | None = None) -> int:
        """Check running jobs; bake + switch the persona's model when one completes."""
        sql = "SELECT * FROM finetune_jobs WHERE status = 'running'"
        rows = await self.store.all(sql + (" AND persona_id = ?" if pid else ""), *([pid] if pid else []))
        finished = 0
        for job in rows:
            ref = jload(job["job_ref"], {}) or {}
            if job["provider"] != "fireworks" or not ref.get("name"):
                continue
            try:
                r = await self.http().get(f"{FIREWORKS}/{ref['name']}", headers=self._fw_headers())
                state = r.json().get("state", "") if r.status_code < 400 else ""
            except Exception as exc:  # noqa: BLE001
                log.warning("fireworks poll failed: %s", exc)
                continue
            if state in DONE_STATES:
                await self._finish(job["id"], job["persona_id"], True, state, ref.get("output", ""))
                finished += 1
            elif state in FAIL_STATES:
                await self._finish(job["id"], job["persona_id"], False, state, "")
                finished += 1
        return finished

    async def compress(self, pid: int) -> int:
        rows = await self.store.all("SELECT * FROM juice WHERE persona_id = ? AND tier = 'baked'", pid)
        for r in rows:
            await self.store.update("juice", r["id"], tier="compressed",
                                    compressed=compress_text(r["raw_transcript"], jload(r["style_markers"], [])),
                                    raw_transcript="", reply="", raw_chars=0, compressed_at=now())
        await self.sub.trim(pid)
        return len(rows)
