"""What a host may give the loop, and what it may withhold.

The rule that keeps this honest: **the loop must run on an API key alone.** If
anything here becomes required for a reply to come out, the boundary is in the
wrong place and the kernel has stopped being a kernel.

So every capability is optional and every one of them degrades to nothing. A
host that supplies no memory gets a mind with no past. A host that supplies no
tools gets a mind that only talks. Both are correct behaviour, not failures.

Capabilities are plain protocols (structural typing), not base classes - a host
should be able to satisfy one with whatever it already has, without importing
our inheritance tree.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Memory: the loop's past.
#
# The loop asks for memories and is handed text it can drop straight into
# context. It never learns whether that came from SQLite, Postgres, a vector
# store or a file - and it must not care.
# --------------------------------------------------------------------------
@runtime_checkable
class MemoryCap(Protocol):
    async def recall(self, query: str, *, limit: int = 6) -> list[str]:
        """Past lines relevant to ``query``, most relevant first, already rendered
        as text the voice can read. Empty list when nothing fits."""

    async def remember(self, text: str) -> None:
        """Keep ``text`` from this exchange. Must not raise on failure - the
        loop is speaking while this happens."""


# --------------------------------------------------------------------------
# Tools: the loop's hands.
#
# Deliberately shaped like OpenAI's own tool-calling contract, because that is
# the widest door available: anything that already speaks chat/completions can
# supply this without translation.
# --------------------------------------------------------------------------
@runtime_checkable
class ToolsCap(Protocol):
    def schemas(self, mode: str = "") -> list[dict]:
        """OpenAI function-tool schemas. Empty list means no hands.

        ``mode`` is the writing key that is about to act, "" when the host is
        not separating them. A host that serves the same belt to both ignores
        it, which is the default and still correct. A host that does separate
        them should give the speaking pass the whole belt and the silent pass
        only what it can look at - the quiet side of a mind reads, it does not
        press things.
        """

    async def run(self, name: str, arguments: dict) -> str:
        """Execute and return the result as text the voice can fold into
        speech. Return an error string rather than raising."""


# --------------------------------------------------------------------------
# Verification: the loop's second opinion.
#
# Midstream is the point. A verdict that arrives after the reply is a postmortem;
# a verdict that arrives while the next beat is still being composed can still
# change what gets said.
# --------------------------------------------------------------------------
@dataclass
class VerificationResult:
    """A host-neutral verifier result, including fields safe to show the person."""

    note: str = ""
    state: str = "READY"
    level: str = "GREEN"
    risk: int | None = None
    basis: str = ""
    finding_count: int = 0


@runtime_checkable
class VerifyCap(Protocol):
    async def verify(self, *, said: str, question: str) -> str | VerificationResult:
        """Judge what has been said so far.

        Return a short correction when something is wrong. Hosts with a meter
        may also return its public rating so the UI can show successful checks.
        A verifier that also accepts ``evidence: str`` gets the turn's newest
        tool results, so a claim taken from them is judged against them.
        """


# --------------------------------------------------------------------------
# Introspection: what the silent pass does with its spare time.
#
# This is the piece that makes the relay worth building. The pass that is NOT
# holding the microphone has exactly one job: be ready to speak next. It uses
# this hook to do the work that today happens serially, on the critical path.
#
# ``widening`` is the direct answer's private work (the fact, what the hands
# brought back). The other pass gets that memory as what the answer is about
# to leave out, plus the other way of seeing the same facts.
# --------------------------------------------------------------------------
@runtime_checkable
class InsightCap(Protocol):
    async def insight(self, *, mode: str, said: str, question: str) -> str:
        """A note for the pass that is about to speak. "" when there is nothing
        worth saying. ``mode`` is the writing key that will speak next."""


@dataclass
class CapabilityBundle:
    """Everything a host can offer. All optional, all default to nothing."""

    memory: MemoryCap | None = None
    tools: ToolsCap | None = None
    verify: VerifyCap | None = None
    insight: InsightCap | None = None
    # Anything a host wants to pass through to its own tool implementations
    # (credentials, a persona id, a webhook). The loop never reads it.
    host: Any = None

    @property
    def has_tools(self) -> bool:
        return self.tools is not None and bool(_safe_schemas(self.tools, ""))

    def schemas_for(self, mode: str = "") -> list[dict]:
        """The belt for one writing. Never raises: a host with broken tools is
        a host with no tools, and the reply still has to come out."""
        if self.tools is None:
            return []
        return _safe_schemas(self.tools, mode)

    def describe(self) -> dict:
        """What this host actually wired up. Useful in health endpoints and in
        the side-by-side test, so you can tell a capability gap from a bug."""
        return {
            "memory": self.memory is not None,
            "tools": self.has_tools,
            "verify": self.verify is not None,
            "insight": self.insight is not None,
        }


class NullCapabilities(CapabilityBundle):
    """An API key and nothing else. The loop must still think and alternate."""


def _safe_schemas(tools: ToolsCap, mode: str) -> list[dict]:
    """Call ``schemas`` with or without the mode argument.

    A host written before the writings were allowed different hands has a
    one-argument ``schemas``. Passing the mode to it would raise, and a
    capability that raises is a capability that is silently switched off - so
    inspect the signature and only pass what the host actually accepts.
    """
    try:
        accepts_mode = "mode" in inspect.signature(tools.schemas).parameters
    except (TypeError, ValueError):  # builtins, functools.partial, C extensions
        accepts_mode = False
    try:
        return list((tools.schemas(mode) if accepts_mode else tools.schemas()) or [])
    except Exception:  # noqa: BLE001 - a host with broken tools is a host with no tools
        return []


__all__ = [
    "CapabilityBundle",
    "InsightCap",
    "MemoryCap",
    "NullCapabilities",
    "ToolsCap",
        "VerifyCap",
    ]