"""The two copies she speaks from. The current rule is README.md

    AL, the fast copy, learns everything: every active layer and every
    exchange. Then it decays toward the base model (the adapter shrinks
    by 0.002 of itself). It does not decay toward AR.
    AR, the facts copy, learns facts only: the fact layer, entities, and
    a tool's name and result. It does not decay.
    Home only. A visit does not write. Nothing is retired from memory.

The 2026-10-21 rehearsal (``sync``, ``cap``, ``COOL = 0.05``) pulled R
toward L and cooled L toward R. That is not the mouth. Those functions
stay so the old arithmetic test still runs. ``Learner.turn`` follows
the rule above.

ToyBackend is a tiny byte model so the diet and the cool can be measured
without a GPU. It is not the mouth. The mouth is Qwen/Qwen3.8-27B on
CoreWeave (see life/mouth.py). Fireworks does not serve a LoRA per token.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# 2026-10-21 rehearsal only. The mouth uses life.mouth.COOL (0.002) and
# cools the fast copy toward the base, not toward the facts copy.
ALPHA = 0.05
COOL = 0.05


# --------------------------------------------------------------------- weights
Adapter = dict[str, np.ndarray]


def zeros_like(a: Adapter) -> Adapter:
    return {k: np.zeros_like(v) for k, v in a.items()}


def gap(l: Adapter, r: Adapter) -> Adapter:
    return {k: l[k] - r[k] for k in l}


def norm(a: Adapter) -> float:
    return float(np.sqrt(sum(float(np.sum(v * v)) for v in a.values())))


def cool_toward_base(adapter: Adapter, cool: float) -> Adapter:
    """Shrink a delta toward zero. Zero is the base model. The facts copy is not passed here."""
    keep = 1.0 - cool
    return {k: v * keep for k, v in adapter.items()}


def sync(l: Adapter, r: Adapter, *, alpha: float = ALPHA, cool: float = COOL) -> tuple[Adapter, Adapter]:
    """2026-10-21 rehearsal: R pulled toward L, L cooled toward R. Not the mouth."""
    g = gap(l, r)
    r2 = {k: r[k] + alpha * g[k] for k in r}
    l2 = {k: l[k] - cool * g[k] for k in l}
    return l2, r2


def cap(l: Adapter, r: Adapter, max_gap: float) -> tuple[Adapter, float]:
    """2026-10-21 rehearsal: pull L back toward R. The mouth does not call this."""
    g = gap(l, r)
    n = norm(g)
    if n <= max_gap or n == 0:
        return l, 1.0
    s = max_gap / n
    return {k: r[k] + s * g[k] for k in l}, s


# --------------------------------------------------------------------- backend
class Backend(Protocol):
    """Where the learning step runs and how base + L is served."""

    name: str

    def init_adapter(self) -> Adapter: ...
    def step(self, adapter: Adapter, texts: list[str], lr: float) -> tuple[Adapter, float, float]:
        """One learning step on ``texts``. Returns (new adapter, loss before, loss after)."""
        ...
    def loss(self, adapter: Adapter, texts: list[str]) -> float: ...


class ToyBackend:
    """A tiny byte-level next-character model: frozen base matrix W plus a
    low-rank adapter dW = B @ A. Real cross-entropy, real gradients. Small
    enough to run every turn on a CPU; the point is that the L/R loop above it
    is exactly the one her real model will use."""

    name = "toy"
    # Measured 2026-10-21: 0.5 barely moved L in 100 turns (5.547 -> 5.546);
    # 20 diverged (loss 26). 2.0 learns steadily on this model. Her real model
    # gets its own value - step size is a property of the model, not the design.
    default_lr = 2.0

    def __init__(self, rank: int = 8, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.V = 256
        self.rank = rank
        self.W = rng.normal(0, 0.02, (self.V, self.V)).astype(np.float64)   # frozen base
        self._seed = seed

    def init_adapter(self) -> Adapter:
        rng = np.random.default_rng(self._seed + 1)
        # Standard low-rank init: A small random, B zero, so base + L == base at start.
        return {"A": rng.normal(0, 0.1, (self.rank, self.V)), "B": np.zeros((self.V, self.rank))}

    @staticmethod
    def _pairs(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for t in texts:
            b = t.encode("utf-8", "ignore")[:4000]
            if len(b) >= 2:
                xs.extend(b[:-1])
                ys.extend(b[1:])
        return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)

    def _forward(self, ad: Adapter, x: np.ndarray):
        W = self.W + ad["B"] @ ad["A"]
        logits = W[x]                                   # (n, V)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return W, p

    def loss(self, adapter: Adapter, texts: list[str]) -> float:
        x, y = self._pairs(texts)
        if len(x) == 0:
            return 0.0
        _, p = self._forward(adapter, x)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    def step(self, adapter: Adapter, texts: list[str], lr: float) -> tuple[Adapter, float, float]:
        x, y = self._pairs(texts)
        if len(x) == 0:
            return adapter, 0.0, 0.0
        _, p = self._forward(adapter, x)
        before = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
        d = p.copy()
        d[np.arange(len(y)), y] -= 1.0
        d /= len(y)
        gW = np.zeros((self.V, self.V))
        np.add.at(gW, x, d)                             # dLoss/dW (rows touched by inputs)
        A, B = adapter["A"], adapter["B"]
        gB = gW @ A.T
        gA = B.T @ gW
        new = {"A": A - lr * gA, "B": B - lr * gB}
        after = self.loss(new, texts)
        return new, before, after


def backend_from_env() -> Backend | None:
    kind = os.environ.get("SAWII_WEIGHTS", "").strip().lower()
    if kind == "toy":
        return ToyBackend()
    if kind == "fireworks":
        log.warning("SAWII_WEIGHTS=fireworks: the Fireworks backend is not built yet "
                    "(needs the model + key and the per-turn training call verified). Learning is OFF.")
        return None
    return None


# --------------------------------------------------------------------- storage
class WeightStore:
    """L and R per persona as weight files, with saved versions for rollback."""

    def __init__(self, root: str):
        self.root = root

    def _dir(self, pid: int) -> str:
        d = os.path.join(self.root, "weights", str(pid))
        os.makedirs(d, exist_ok=True)
        return d

    def load(self, pid: int, backend: Backend) -> tuple[Adapter, Adapter, dict]:
        d = self._dir(pid)
        meta_path = os.path.join(d, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"turns": 0, "cursor": 0,
                                                                           "seen_layer_id": 0, "seen_turn_id": 0}
        if os.path.exists(os.path.join(d, "L.npz")) and os.path.exists(os.path.join(d, "R.npz")):
            L = dict(np.load(os.path.join(d, "L.npz")))
            R = dict(np.load(os.path.join(d, "R.npz")))
        else:
            L = backend.init_adapter()
            R = {k: v.copy() for k, v in L.items()}
        return L, R, meta

    def save(self, pid: int, L: Adapter, R: Adapter, meta: dict) -> None:
        d = self._dir(pid)
        np.savez(os.path.join(d, "L.npz"), **L)
        np.savez(os.path.join(d, "R.npz"), **R)
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        if meta["turns"] % 10 == 0:
            v = os.path.join(d, "versions", f"{meta['turns']:08d}")
            os.makedirs(v, exist_ok=True)
            np.savez(os.path.join(v, "L.npz"), **L)
            np.savez(os.path.join(v, "R.npz"), **R)
            json.dump(meta, open(os.path.join(v, "meta.json"), "w"))
            # Every version is kept (nothing is deleted automatically).

    def versions(self, pid: int) -> list[str]:
        p = os.path.join(self._dir(pid), "versions")
        return sorted(os.listdir(p)) if os.path.isdir(p) else []

    def rollback(self, pid: int, version: str) -> None:
        d = self._dir(pid)
        src = os.path.join(d, "versions", version)
        for f in ("L.npz", "R.npz", "meta.json"):
            with open(os.path.join(src, f), "rb") as a, open(os.path.join(d, f), "wb") as b:
                b.write(a.read())


# --------------------------------------------------------------------- memory
async def memory_texts(store, pid: int, meta: dict, *, fresh_limit: int = 24, slice_size: int = 24) -> list[str]:
    """What the fast copy learns: new memories at once, plus the next slice
    of everything (turns and every active layer). A superseded row is skipped.
    A reply the verifier C judged RED is never learned: the weight gate
    (issue #9) - the mouth does not rehearse what C caught as wrong.
    The facts copy uses fact_texts, not this."""
    fresh = []
    rows = await store.all(
        "SELECT id, text, quote FROM layers WHERE persona_id = ? AND status = 'active' AND id > ? "
        "ORDER BY id LIMIT ?", pid, int(meta.get("seen_layer_id", 0)), fresh_limit)
    for r in rows:
        fresh.append(_layer_text(r))
        meta["seen_layer_id"] = max(int(meta.get("seen_layer_id", 0)), int(r["id"]))
    trows = await store.all(
        "SELECT id, user_text, reply_text FROM turns WHERE persona_id = ? AND reply_text != '' AND id > ? "
        "AND id NOT IN (SELECT turn_id FROM verdicts WHERE persona_id = ? AND level = 'RED' AND turn_id IS NOT NULL) "
        "ORDER BY id LIMIT ?", pid, int(meta.get("seen_turn_id", 0)), pid, fresh_limit)
    for r in trows:
        fresh.append(_turn_text(r))
        meta["seen_turn_id"] = max(int(meta.get("seen_turn_id", 0)), int(r["id"]))

    # The rest of the memory, a slice at a time, wrapping around forever.
    total = (await store.one(
        "SELECT (SELECT COUNT(*) FROM layers WHERE persona_id = ? AND status = 'active') + "
        "(SELECT COUNT(*) FROM turns WHERE persona_id = ? AND reply_text != '' "
        "AND id NOT IN (SELECT turn_id FROM verdicts WHERE persona_id = ? AND level = 'RED' AND turn_id IS NOT NULL) ) AS n", pid, pid, pid))["n"]
    cycling = []
    if total:
        cur = int(meta.get("cursor", 0)) % total
        rows = await store.all(
            "SELECT text, quote, NULL AS user_text, NULL AS reply_text, 0 AS k, id FROM layers "
            "WHERE persona_id = ? AND status = 'active' "
            "UNION ALL SELECT '' , '', user_text, reply_text, 1 AS k, id FROM turns "
            "WHERE persona_id = ? AND reply_text != '' "
            "AND id NOT IN (SELECT turn_id FROM verdicts WHERE persona_id = ? AND level = 'RED' AND turn_id IS NOT NULL) "
            "ORDER BY k, id LIMIT ? OFFSET ?",
            pid, pid, pid, slice_size, cur)
        for r in rows:
            cycling.append(_turn_text(r) if r["k"] == 1 else _layer_text(r))
        meta["cursor"] = (cur + max(1, len(rows))) % total
    return [t for t in fresh + cycling if t.strip()]


def _layer_text(r) -> str:
    q = (r["quote"] or "").strip()
    return (r["text"] or "").strip() + (f'\n(they said: "{q}")' if q else "")


def _turn_text(r) -> str:
    return f"They: {r['user_text'] or ''}\nMe: {r['reply_text'] or ''}"


def _entity_text(r) -> str:
    account = (r["account"] or "").strip()
    return f"{r['kind'] or 'person'}: {r['name']}" + (f" ({account})" if account else "")


async def fact_texts(store, pid: int, meta: dict, *, fresh_limit: int = 24, slice_size: int = 24) -> list[str]:
    """What the facts copy learns. It does not decay.

    A fact is an active ``fact`` layer (what happened, including a tool's
    name and result once that was written as one), an entity (who
    someone is), or a passage from a file she was given, including the
    mind that file makes her. Notes, readings, collateral, hypotheses,
    and the raw exchange stay on the fast copy.
    """
    fresh = []
    rows = await store.all(
        "SELECT id, text, quote FROM layers WHERE persona_id = ? AND status = 'active' "
        "AND layer = 'fact' AND id > ? ORDER BY id LIMIT ?",
        pid, int(meta.get("fact_seen_id", 0)), fresh_limit)
    for r in rows:
        fresh.append(_layer_text(r))
        meta["fact_seen_id"] = max(int(meta.get("fact_seen_id", 0)), int(r["id"]))
    erows = await store.all(
        "SELECT id, name, kind, account FROM entities WHERE persona_id = ? AND id > ? ORDER BY id LIMIT ?",
        pid, int(meta.get("fact_seen_entity", 0)), fresh_limit)
    for r in erows:
        fresh.append(_entity_text(r))
        meta["fact_seen_entity"] = max(int(meta.get("fact_seen_entity", 0)), int(r["id"]))

    total = (await store.one(
        "SELECT (SELECT COUNT(*) FROM layers WHERE persona_id = ? AND status = 'active' AND layer = 'fact') + "
        "(SELECT COUNT(*) FROM entities WHERE persona_id = ?) AS n", pid, pid))["n"]
    cycling = []
    if total:
        cur = int(meta.get("fact_cursor", 0)) % total
        rows = await store.all(
            "SELECT text, quote, NULL AS name, NULL AS kind, NULL AS account, 0 AS k, id FROM layers "
            "WHERE persona_id = ? AND status = 'active' AND layer = 'fact' "
            "UNION ALL SELECT '', '', name, kind, account, 1 AS k, id FROM entities "
            "WHERE persona_id = ? ORDER BY k, id LIMIT ? OFFSET ?",
            pid, pid, slice_size, cur)
        for r in rows:
            cycling.append(_entity_text(r) if r["k"] == 1 else _layer_text(r))
        meta["fact_cursor"] = (cur + max(1, len(rows))) % total
    from ..knowledge import knowledge_facts
    books = await knowledge_facts(store, pid, meta, fresh_limit=fresh_limit, slice_size=slice_size)
    return [t for t in fresh + cycling + books if t.strip()]


# --------------------------------------------------------------------- the turn
@dataclass
class LearnResult:
    learned: bool
    examples: int = 0
    loss_before: float = 0.0
    loss_after: float = 0.0
    gap: float = 0.0
    capped: float = 1.0
    turns: int = 0
    backend: str = ""
    ms: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class Learner:
    """One home turn. Fast copy learns everything and cools toward the base. Facts copy learns facts and stays."""

    def __init__(self, backend: Backend, store: WeightStore, *, lr: float | None = None, max_gap: float = 5.0):
        self.backend = backend
        self.weights = store
        self.lr = lr if lr is not None else getattr(backend, "default_lr", 0.5)
        self.max_gap = max_gap

    async def turn(self, db, pid: int) -> LearnResult:
        from .mouth import COOL as MOUTH_COOL

        t0 = time.perf_counter()
        L, R, meta = self.weights.load(pid, self.backend)
        texts = await memory_texts(db, pid, meta)
        facts = await fact_texts(db, pid, meta)
        if not texts and not facts:
            return LearnResult(learned=False, backend=self.backend.name, notes=["memory is empty"])
        before = after = 0.0
        if texts:
            L, before, after = self.backend.step(L, texts, self.lr)
            L = cool_toward_base(L, MOUTH_COOL)
        if facts:
            R, _, _ = self.backend.step(R, facts, self.lr)
        meta["turns"] = int(meta.get("turns", 0)) + 1
        self.weights.save(pid, L, R, meta)
        return LearnResult(learned=True, examples=len(texts), loss_before=before, loss_after=after,
                           gap=norm(gap(L, R)), capped=1.0, turns=meta["turns"], backend=self.backend.name,
                           ms=round((time.perf_counter() - t0) * 1000),
                           notes=[f"facts {len(facts)}"])
