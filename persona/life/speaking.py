"""She speaks from AL and AR, and the step writes to those same two copies.

AL is the fast copy. The lobe that answers and presses tools speaks from
it. A home turn trains it on everything she remembers. It then decays
toward the base model, 0.002 of the remaining gap, not toward AR.

AR is the facts copy. The other lobe speaks from it. A home turn trains
it only on facts: the fact layer, who someone is, and what a tool
returned. It does not decay.

On a server that applies the step, the chat request names AL or AR and
POST {base}/adapters/step sends fast_texts, slow_texts, and
cool_toward=base. On CoreWeave the mouth is Qwen/Qwen3.8-27B until the
adapter files are published. A frozen host is skipped: Featherless, and a
token chat API (Fireworks, Together) that cannot apply the step. A
visit never calls this.

The current rule is README.md
"""
from __future__ import annotations

import json
import logging
import os

from .mouth import ALPHA, BASE_MODEL, COOL, is_coreweave, publish_adapter
from .weights import fact_texts, memory_texts

log = logging.getLogger(__name__)

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


class Speaking:
    def __init__(self, settings):
        self.s = settings
        self.publisher = None

    def active(self) -> bool:
        kind = os.environ.get("SAWII_WEIGHTS", "speaking").strip().lower()
        if kind in ("off", "0", "false", "no", "toy"):
            return False
        if "tinker.thinkingmachines.dev" in (self.s.a.base_url or "").lower():
            # Tinker checkpoint training is NOT the old synchronous adapter step.
            return False
        # Only an explicitly configured adapter-update server can apply
        # /adapters/step. A chat-completions API (including DeepInfra) cannot.
        # Never fabricate a learning URL by appending /adapters/step to a
        # third-party inference endpoint.
        if self.s.update_url:
            return httpx is not None
        if is_coreweave(self.s.a.base_url):
            return True
        return False

    def _path(self, pid: int) -> str:
        folder = os.path.join(self.s.data_dir, "weights", str(int(pid)))
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "instances.json")

    def _saved(self, pid: int) -> dict:
        path = self._path(pid)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def models(self, pid: int) -> dict[str, str] | None:
        """What the two lobes call. None until the copies have names.

        On the per-token mouth both lobes call the base model until the
        adapter files have been published.
        """
        saved = self._saved(pid)
        al = saved.get("AL") or self.s.al_model
        ar = saved.get("AR") or self.s.ar_model
        if "tinker.thinkingmachines.dev" in (self.s.a.base_url or "").lower():
            base = self.s.a.model
            return {"L": al or base, "R": ar or base}
        if is_coreweave(self.s.a.base_url):
            base = self.s.a.model or BASE_MODEL
            return {"L": al or base, "R": ar or base}
        if not al and not ar:
            return None
        base = self.s.a.model
        return {"L": al or base, "R": ar or base}

    def _url(self) -> str:
        if self.s.update_url:
            return self.s.update_url
        return self.s.a.base_url.rstrip("/") + "/adapters/step"

    def _adapter_dir(self, pid: int, name: str) -> str:
        override = os.environ.get(f"SAWII_{name}_DIR", "")
        if override:
            return override
        return os.path.join(self.s.data_dir, "weights", str(int(pid)), name)

    async def _publish(self, pid: int) -> dict:
        """Replace AL and AR with the adapter files on disk. Speech stays per token."""
        base = self.s.a.model or BASE_MODEL
        al_dir = self._adapter_dir(pid, "AL")
        ar_dir = self._adapter_dir(pid, "AR")
        if not (os.path.isdir(al_dir) and os.path.isdir(ar_dir)):
            return {"updated": False, "model": base,
                    "reason": "the mouth is the base model; the adapter files are not written yet"}
        entity = os.environ.get("SAWII_WANDB_ENTITY", "").strip()
        project = os.environ.get("SAWII_WANDB_PROJECT", "").strip()
        key = os.environ.get("SAWII_WANDB_API_KEY", "").strip() or os.environ.get("WANDB_API_KEY", "").strip()
        if not (entity and project and key):
            return {"updated": False, "model": base,
                    "reason": "the adapter files are local; the registry key is not set"}
        publish = self.publisher or publish_adapter
        try:
            al = publish(al_dir, "sawii-al", entity=entity, project=project, base_model=base, api_key=key)
            ar = publish(ar_dir, "sawii-ar", entity=entity, project=project, base_model=base, api_key=key)
        except Exception as exc:  # noqa: BLE001 - a missed step never blocks the next reply
            log.warning("speaking publish failed: %s", exc)
            return {"updated": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        meta = self._saved(pid)
        meta["AL"] = str(al)
        meta["AR"] = str(ar)
        meta["base"] = base
        meta["turns"] = int(meta.get("turns", 0)) + 1
        with open(self._path(pid), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        return {"updated": True, "AL": meta["AL"], "AR": meta["AR"], "turns": meta["turns"]}

    async def turn(self, db, pid: int) -> dict:
        if is_coreweave(self.s.a.base_url) and not self.s.update_url:
            return await self._publish(pid)
        if not self.active() or httpx is None:
            return {"updated": False, "reason": "the speaking model is not one we can write"}
        meta = self._saved(pid)
        fast_texts = await memory_texts(db, pid, meta)
        slow_texts = await fact_texts(db, pid, meta)
        if not fast_texts and not slow_texts:
            return {"updated": False, "reason": "memory is empty"}
        al = meta.get("AL") or self.s.al_model or "AL"
        ar = meta.get("AR") or self.s.ar_model or "AR"
        body = {"fast": al, "slow": ar, "base": self.s.a.model,
                "fast_texts": fast_texts, "slow_texts": slow_texts,
                "alpha": ALPHA, "cool": COOL, "cool_toward": "base"}
        headers = {"Content-Type": "application/json"}
        if self.s.a.api_key:
            headers["Authorization"] = f"Bearer {self.s.a.api_key}"
        try:
            async with httpx.AsyncClient(timeout=self.s.update_timeout) as client:
                response = await client.post(self._url(), json=body, headers=headers)
                response.raise_for_status()
                detail = response.json()
        except Exception as exc:  # noqa: BLE001 - a missed step never blocks the next reply
            log.warning("speaking step failed: %s", exc)
            return {"updated": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        if not isinstance(detail, dict):
            detail = {}
        meta["AL"] = str(detail.get("AL") or al)
        meta["AR"] = str(detail.get("AR") or ar)
        meta["turns"] = int(meta.get("turns", 0)) + 1
        with open(self._path(pid), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        return {"updated": True, "AL": meta["AL"], "AR": meta["AR"],
                "examples": len(fast_texts), "facts": len(slow_texts),
                "turns": meta["turns"]}
