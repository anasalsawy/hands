"""The trace: one JSONL line for everything that happens inside Persona.

Built because a question could not be answered any other way: a person asked
"what was I saying 17 minutes ago" and she answered instantly and verbatim, and
nobody in the codebase could say which part of the system did that. Memory,
recall, lanes, prompts and models all contribute, and none of them left a
record deep enough to check. So this records all of it: every model call with
the exact prompt that went in, every tool with its arguments and result, every
memory read, every lane step, with timings.

Three rules, and they are what make it safe to run in production:

- **It never raises.** A logger that can break a turn is worse than no logger.
  Every write is wrapped, and if the disk is full the turn goes on and the
  trace is short.
- **It loses nothing on purpose.** Prompt text is stored whole. A trace that
  stores "system: 14321 chars" cannot answer what was in front of her; the
  char-count is stored as well, for when the text is not wanted.
- **One line per event, with a turn id.** So `grep turn=abc123` rebuilds a
  whole turn in order, and the answer to "what made that reply good" is a
  diff between two turns rather than a guess.

Enable with SAWII_TRACE=1 (or the `trace` setting). Files are
`<data_dir>/trace/trace-YYYYMMDD.jsonl`, rotated by day.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from typing import Any

# Which process wrote a line. Turn ids restart at 1 in every process, so a log
# shared by two runs is unreadable without this.
_PROC = os.getpid()

log = logging.getLogger(__name__)

# Which turn, which call, which line of work. Set by the turn loop and by the
# lane runner so a model call deep inside a lane is still attributable.
turn_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("sawii_trace_turn", default="")
pid_var: contextvars.ContextVar[int] = contextvars.ContextVar("sawii_trace_pid", default=0)
# Which of her minds is speaking to the model right now: "voice", "planning",
# "line ln...". Set by the lane runner inside each asyncio task, and read by
# the llm hook, so a model call in the middle of a fan-out is attributable
# without threading a label through every signature.
purpose_var: contextvars.ContextVar[str] = contextvars.ContextVar("sawii_trace_purpose", default="")


class Trace:
    """Append-only JSONL event log. Thread-safe, never raises."""

    def __init__(self, path: str | None = None, *, enabled: bool = True,
                 max_field: int = 200_000):
        self.enabled = bool(enabled and path)
        self.max_field = max_field
        self.dir = path or ""
        self._lock = threading.Lock()
        self._day = ""
        self._fh = None
        self.dropped = 0
        if self.enabled:
            try:
                os.makedirs(self.dir, exist_ok=True)
            except OSError:
                log.warning("trace: cannot create %s; tracing off", self.dir)
                self.enabled = False

    # ------------------------------------------------------------------ writing
    def _file(self):
        day = time.strftime("%Y%m%d")
        if self._fh is not None and day == self._day:
            return self._fh
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._day = day
        self._fh = open(os.path.join(self.dir, f"trace-{day}.jsonl"), "a", encoding="utf-8")
        return self._fh

    def event(self, kind: str, /, **fields: Any) -> None:
        """Record one thing that happened. `kind` is the verb: model_call, tool, recall...

        Never raises. A trace failure is logged once to the normal log and
        counted, because silently swallowing it is how a broken logger ends up
        looking like a quiet system.
        """
        if not self.enabled:
            return
        # `persona` is whose mind this is; `proc` is which process wrote it. A
        # trace file is append-only and outlives the process, and turn ids count
        # up from zero inside each one, so without `proc` two conversations in
        # the same file read as one impossible conversation.
        row = {"t": round(time.time(), 4), "mono": round(time.monotonic(), 4), "proc": _PROC,
               "kind": kind, "turn": turn_id_var.get(), "persona": pid_var.get()}
        if lane := purpose_var.get():
            row["lane"] = lane
        if not row["turn"]:
            row.pop("turn")
        if not row["persona"]:
            row.pop("persona")
        for k, v in fields.items():
            row[k] = self._fit(v)
        try:
            with self._lock:
                fh = self._file()
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                fh.flush()
        except Exception as exc:  # noqa: BLE001 - the turn must not care
            self.dropped += 1
            if self.dropped == 1:
                log.warning("trace write failed (%s); continuing without it", exc)

    def _fit(self, value):
        """Bound one field, keeping the head and tail so the shape survives."""
        if isinstance(value, str) and len(value) > self.max_field:
            cut = self.max_field
            return value[: cut // 2] + f"\n…{len(value) - cut} chars…\n" + value[-cut // 2:]
        if isinstance(value, dict):
            return {k: self._fit(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._fit(v) for v in value]
        return value

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None

    # ------------------------------------------------------------- the useful views
    def prompt_sections(self, messages: list[dict]) -> dict:
        """Sizes of the prompt, so a slow turn can be read as a shape.

        The one question this answers is "did the cache get a chance?" - the
        cached prefix is everything before the first `=== now ===`, and it only
        helps if it is identical to last turn's. So the stable prefix gets a
        digest here: two turns with the same digest had the same cache key.
        """
        system = next((str(m.get("content") or "") for m in messages if m.get("role") == "system"), "")
        if isinstance(system, list):  # pragma: no cover - audio parts
            system = json.dumps(system, ensure_ascii=False)
        head, _, _tail = system.partition("=== now ===")
        out = {"system_chars": len(system), "stable_chars": len(head),
               "stable_sha": _sha(head), "messages": len(messages)}
        for m in messages:
            if m.get("role") == "tool":
                out["tool_messages"] = out.get("tool_messages", 0) + 1
        return out


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


# The one the rest of the code reaches for. Replaced by the Runtime at startup;
# a no-op until then, so nothing needs to know whether tracing is on.
_active = Trace(enabled=False)


def setup(path: str | None, *, enabled: bool = True) -> Trace:
    global _active
    _active.close()
    _active = Trace(path, enabled=enabled)
    return _active


def trace() -> Trace:
    return _active


def event(kind: str, /, **fields: Any) -> None:
    _active.event(kind, **fields)