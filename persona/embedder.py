"""Embeddings for context broadening.

Primary: a real semantic embedder (default Featherless Qwen3-Embedding-8B).
Fallback: a deterministic local feature-hash embedder so the system never
blocks on an outage. Every stored vector is tagged with the model that made
it; vectors from a different model are treated as absent, never compared.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re

import numpy as np

log = logging.getLogger(__name__)

HASH_DIM = 512
HASH_MODEL = "local-hash-512"
_WORD = re.compile(r"[a-z0-9؀-ۿ']+")


def _bucket(token: str) -> tuple[int, float]:
    h = hashlib.blake2b(token.encode(), digest_size=8).digest()
    idx = int.from_bytes(h[:4], "little") % HASH_DIM
    sign = 1.0 if h[4] & 1 else -1.0
    return idx, sign


def hash_embed(text: str) -> list[float]:
    vec = np.zeros(HASH_DIM, dtype=np.float32)
    words = _WORD.findall((text or "").lower())
    for w in words:
        i, s = _bucket("w:" + w)
        vec[i] += s
        if len(w) > 4:  # crude stemming: shared prefixes
            i, s = _bucket("p:" + w[:5])
            vec[i] += 0.5 * s
        padded = f"#{w}#"
        for k in range(len(padded) - 2):
            i, s = _bucket("c:" + padded[k:k + 3])
            vec[i] += 0.25 * s
    n = float(np.linalg.norm(vec))
    return (vec / n).tolist() if n else vec.tolist()


def to_blob(vec: list[float] | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not na or not nb:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    q = query / (np.linalg.norm(query) or 1.0)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return (matrix @ q) / norms


class Embedder:
    """Remote semantic embedder with honest local fallback.

    Tracks last success / fallback times and the recent failure reason so
    /health/subsystems can tell at a glance whether semantic recall is live
    or we're degrading to hash matching.
    """

    def __init__(self, llm=None, *, timeout: float = 15.0):
        import time as _time
        self.llm = llm
        self.timeout = timeout
        self._remote_ok = llm is not None
        self._retry_remote_after = 0.0
        self._remote_failures = 0
        self._last_success_at: float | None = None
        self._last_fallback_at: float | None = None
        self._last_error: str = ""
        self._success_count: int = 0
        self._fallback_count: int = 0
        self._latencies_ms: list[int] = []
        self._last_embedding_ms: int | None = None
        self._time = _time
        # Set by Mind at construction time so each embed call reports health
        # into the central subsystems registry. Optional — the standalone
        # tracking fields above remain for callers that don't use the
        # registry.
        self.on_health = None  # type: ignore[assignment]

    @property
    def model(self) -> str:
        if self.llm is not None and self._remote_ok:
            return getattr(self.llm, "model", "remote")
        return HASH_MODEL

    def health(self) -> dict:
        """Snapshot of the embedder's live state for the subsystems endpoint."""
        now_mono = self._time.monotonic()
        return {
            "model": getattr(self.llm, "model", None) if self.llm else None,
            "live_model": self.model,
            "is_fallback": self.model == HASH_MODEL,
            "last_success_at": self._last_success_at,
            "last_fallback_at": self._last_fallback_at,
            "age_since_success_s": (now_mono - self._last_success_at) if self._last_success_at else None,
            "age_since_fallback_s": (now_mono - self._last_fallback_at) if self._last_fallback_at else None,
            "success_count": self._success_count,
            "fallback_count": self._fallback_count,
            "last_embedding_ms": self._last_embedding_ms,
            "median_last_50_embedding_ms": (sorted(self._latencies_ms)[len(self._latencies_ms)//2]
                                              if self._latencies_ms else None),
            "last_error": self._last_error[:200],
            "remote_configured": self.llm is not None,
            "remote_retry_in_s": max(0.0, self._retry_remote_after - now_mono),
        }

    async def embed(self, texts: list[str]) -> tuple[str, list[list[float]]]:
        texts = [t[:8000] for t in texts]
        model_name = getattr(self.llm, "model", "remote") if self.llm else None
        if self.llm is not None and self._time.monotonic() >= self._retry_remote_after:
            started_embedding = self._time.monotonic()
            try:
                vecs = await asyncio.wait_for(self.llm.embed(texts), self.timeout)
                if vecs and len(vecs) == len(texts):
                    self._remote_ok = True
                    self._remote_failures = 0
                    self._retry_remote_after = 0.0
                    self._last_embedding_ms = round((self._time.monotonic() - started_embedding) * 1000)
                    self._latencies_ms = (self._latencies_ms + [self._last_embedding_ms])[-50:]
                    self._last_success_at = self._time.monotonic()
                    self._success_count += 1
                    self._last_error = ""
                    if self.on_health is not None:
                        self.on_health.ok(extra={"model": model_name, "live_model": model_name})
                    return model_name, vecs
            except Exception as exc:  # outage: degrade, never block a turn
                self._remote_ok = False
                self._remote_failures += 1
                # Avoid repeated network timeouts on every conversational beat.
                # Retry periodically; preserve remote recovery automatically.
                self._retry_remote_after = self._time.monotonic() + min(
                    60.0, 2.0 ** min(self._remote_failures, 6))
                self._last_embedding_ms = round((self._time.monotonic() - started_embedding) * 1000)
                log.warning("embedder fallback to local hash: %s", exc)
                self._last_error = f"{type(exc).__name__}: {exc}"
                if self.on_health is not None:
                    self.on_health.fail(f"{type(exc).__name__}: {exc}",
                                         extra={"model": model_name, "live_model": HASH_MODEL,
                                                "is_fallback": True})
        self._last_fallback_at = self._time.monotonic()
        self._fallback_count += 1
        if self.on_health is not None and self.llm is None:
            # No remote configured at all — not an error, just baseline hash mode.
            self.on_health.extra.update({"model": None, "live_model": HASH_MODEL,
                                          "is_fallback": True, "remote_configured": False})
        return HASH_MODEL, [hash_embed(t) for t in texts]

    async def embed_one(self, text: str) -> tuple[str, list[float]]:
        model, vecs = await self.embed([text])
        return model, vecs[0]
