"""Host adapter: the companion app's ``Mind`` -> the kernel's capabilities.

This file is the seam. Everything on one side is the kernel (which knows nothing
about personas, lanes, lobes or SQLite); everything on the other side is the app
that is already deployed. Nothing here is imported by the kernel, and nothing
here changes what the deployed turn loop does - it only exposes the same objects
through the four doors the kernel expects.

What the companion adds on top of a bare API key:

    memory   -> the real Recall (lanes, salience, mood resonance)
    tools    -> whatever the app already gives A, via the app's own executor
    verify   -> brains.verifier.verify, the meter
    insight  -> the psychological read for the right writing, memory widening
                for the left one

That last line is the responsibility shift made concrete. Today B and C are
separate lobes that must be *scheduled* around A's speech. Here they are not
lobes at all - they are two calls the silent pass makes with time it already had.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from ..caps import CapabilityBundle, VerificationResult
from ...brains import live, verifier
from ...brains.voice_a import A_TOOLS

log = logging.getLogger(__name__)

# A's tool list is the app's, not the kernel's. A host that wants a narrower set
# passes its own; a host that passes nothing gets the workshop's whole bench.
ToolRunner = Callable[[str, dict], Awaitable[str]]


class RecallMemory:
    """``MemoryCap`` over the app's Recall.

    The kernel asks in plain words and gets plain words back. It never sees a
    row id, a lane name or a table, which is what lets the same loop run against
    somebody else's database unchanged.
    """

    def __init__(self, mind, persona: dict, *, budget: int = 2200, exclude_turn_ids: set | None = None):
        self.m = mind
        self.persona = persona
        self.budget = budget
        self.exclude_turn_ids = exclude_turn_ids or set()

    async def recall(self, query: str, *, limit: int = 6) -> list[str]:
        try:
            res = await self.m.recall.recall(self.persona, query, budget=self.budget,
                                             exclude_turn_ids=set(self.exclude_turn_ids))
        except Exception as exc:  # noqa: BLE001 - a mind that cannot remember still speaks
            log.warning("kernel recall failed: %s", exc)
            return []
        text = getattr(res, "text", "") or ""
        if not text.strip():
            return []
        # One entry per surfaced moment: the blank-line blocks Recall renders.
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        return blocks[:limit] if blocks else [text.strip()]

    async def remember(self, text: str) -> None:
        """Kernel memory probes use unified semantic + lexical search."""
        if not (text or "").strip():
            return
        try:
            from ...memory.search import search_all
            await search_all(self.m, self.persona, text, budget=self.budget)
        except Exception as exc:
            log.warning("kernel unified memory probe failed: %s", exc)


class HostTools:
    """``ToolsCap`` over a schema list plus one async executor the host owns.

    The kernel only ever calls ``run(name, args)`` and gets text back. How that
    text was produced - a plugin webhook, the working lane, the local bench - is
    the host's business, which is exactly why the same loop drives any of them.
    """

    def __init__(self, schemas: list[dict], run: ToolRunner, observe=None):
        self._schemas = list(schemas or [])
        self._run = run
        # BR1: a read-only look at the page/screen the hands are working on,
        # for the silent writing to take during a tool wait. None = nothing to see.
        if observe is not None:
            self.observe = observe

    def schemas(self) -> list[dict]:
        return self._schemas

    async def run(self, name: str, arguments: dict) -> str:
        try:
            out = await self._run(name, arguments or {})
        except Exception as exc:  # noqa: BLE001 - a failed hand reports, it does not go silent
            log.warning("kernel tool %s failed: %s", name, exc)
            return f'{{"error": "{type(exc).__name__}"}}'
        return out if isinstance(out, str) else str(out)


def correction(finding: dict) -> str:
    """A finding the next beat can act on silently: a stated fact the record
    contradicts, with the record's words. Anything vaguer stays on the meter."""
    if not isinstance(finding, dict):
        return ""
    said = str(finding.get("claim_quote") or "").strip()
    truth = str(finding.get("basis_quote") or "").strip()
    if finding.get("evidence_status") != "CONTRADICTED" or not said or not truth:
        return ""
    fix = str(finding.get("fix") or "").strip()
    return f'Said "{said[:120]}", but the record says "{truth[:120]}".' + (f" {fix[:120]}" if fix else "")


class MeterVerify:
    """``VerifyCap`` over the meter.

    The level is for the meter only. A note comes back only when a finding is a
    concrete contradicted fact, so the next beat can say the right thing.
    """

    def __init__(self, mind, *, context: str = "", llm=None):
        self.m = mind
        # C is the verifier (owner's B/C swap, 2026-10-13; DECISIONS.md). This
        # used to default to mind.b, so the in-reply "C meter" ran on B's model.
        self.llm = llm if llm is not None else mind.c
        self.context = context

    async def verify(self, *, said: str, question: str, evidence: str = "") -> VerificationResult:
        if not (said or "").strip():
            return VerificationResult(state="UNRATED", level="", note="Nothing to check")
        try:
            m = await verifier.verify(
                self.llm, context=self.context, response_text=said, events=evidence or "",
                events_share=.35 if evidence else None,
                latest_request=question, on_health=self.m.health_registry.get("verifier"))
        except Exception as exc:  # noqa: BLE001 - unrated is not a reason to stop talking
            log.warning("kernel verify failed: %s", exc)
            return VerificationResult(state="UNRATED", level="", note="Check unavailable")
        state = str(getattr(m, "state", "UNRATED"))
        level = str(getattr(m, "level", ""))
        note = ""
        if state == "READY" and level != "GREEN":
            note = " ".join(c for c in map(correction, m.findings or []) if c)[:300]
        risk = getattr(m, "risk", None)
        return VerificationResult(
            note=note,
            state=state,
            level=level,
            risk=risk if isinstance(risk, int) else None,
            basis=str(getattr(m, "basis", ""))[:120],
            finding_count=len(m.findings or []),
        )


class TwoSidedInsight:
    """``InsightCap``: what the quiet pass is handed while the mouth is busy.

    ``mode == "L"`` (the direct answer is about to speak) -> the memory slice
    that holds the fact.

    ``mode == "R"`` (the other half is about to speak) -> that same memory,
    named as what the answer is about to leave out, plus the other reading of
    the same facts. How it was said is included only when a live read exists,
    because a flat "fine" can change the reading. It is not the job.
    """

    def __init__(self, mind, persona: dict, *, tail: list[dict] | None = None,
                 exclude_layer_ids: set | None = None, exclude_turn_ids: set | None = None):
        self.m = mind
        self.persona = persona
        self.tail = tail or []
        self.exclude_layer_ids = exclude_layer_ids or set()
        self.exclude_turn_ids = exclude_turn_ids or set()

    async def insight(self, *, mode: str, said: str, question: str) -> str:
        try:
            if mode == "R":
                return await self._left_out(question)
            return await self._widen(question)
        except Exception as exc:  # noqa: BLE001 - a silent pass that fails just stays silent
            log.warning("kernel insight (%s) failed: %s", mode, exc)
            return ""

    async def _widen(self, question: str) -> str:
        # Session-grounded companion: reuse broad initial context. Do not
        # initiate a second automatic embedding call during silent thinking.
        state = getattr(self.m, "_working_memory", {}).get(self.persona.get("id"))
        if state is not None:
            res = state.get("result")
            return str(getattr(res, "text", "") or "")[:1200]
        cands = await self.m.recall.candidates(
            self.persona, question, exclude_ids=set(self.exclude_layer_ids), limit=8,
            exclude_turn_ids=set(self.exclude_turn_ids))
        if not cands:
            return ""
        lines = []
        for c in cands[:4]:
            body = (c.get("text") or c.get("quote") or "").strip()
            if body:
                lines.append(f"- {body[:300]}")
        return "\n".join(lines)

    async def _left_out(self, question: str) -> str:
        """Memory first, then the other reading. A dead read does not wipe the memory."""
        parts: list[str] = []
        try:
            # Once grounded, deeper retrieval is chosen through an explicit
            # search_memory call, not triggered silently by each lobe beat.
            if getattr(self.m, "_working_memory", {}).get(self.persona.get("id")) is not None:
                alerts = []
                from ...memory.recall import format_similarity_alerts
            else:
                from ...memory.recall import format_similarity_alerts
                alerts = await self.m.recall.similarity_alerts(
                self.persona, question, threshold=getattr(self.m.s, "open_sesame_threshold", 0.78),
                exclude_turn_ids=set(self.exclude_turn_ids) or None)
            block = format_similarity_alerts(alerts)
            if block:
                parts.append(block)
        except Exception:  # noqa: BLE001 - a missed resemblance is not a failed reply
            pass
        memory = await self._widen(question)
        if memory:
            parts.append("From memory, what this answer is about to leave out:\n" + memory)
        read: dict = {}
        try:
            read = await live.gut_read(
                self.m.c, user_text=question, tail=self.tail,
                expect=self.persona.get("c_expect", "")) or {}
        except Exception:  # noqa: BLE001 - how it was said is extra, the memory still stands
            read = {}
        if read.get("surprise"):
            parts.append(f"The other reading (use it only if it changes the answer): {read['surprise']}")
        if read.get("tone"):
            parts.append(f"How it was said (let it shape your answer; never remark on it): {read['tone']}")
        return "\n".join(parts)


def bundle_for_companion(
    mind,
    *,
    persona: dict,
    turn_id: int | None = None,
    tail: list[dict] | None = None,
    context: str = "",
    tool_schemas: list[dict] | None = None,
    run_tool: ToolRunner | None = None,
    observe=None,
    memory: bool = True,
    verify: bool = True,
    insight: bool = True,
    extra: dict[str, Any] | None = None,
) -> CapabilityBundle:
    """Wire the companion's machinery into a kernel ``CapabilityBundle``.

    Every flag defaults on, and every capability degrades to nothing on failure,
    so a half-broken host still gets a working mind rather than an error page.
    ``run_tool`` is required for tools: the kernel will not guess how to reach
    the workshop.
    """
    exclude_turn = {turn_id} if turn_id is not None else set()

    tools = None
    if run_tool is not None:
        tools = HostTools(tool_schemas if tool_schemas is not None else A_TOOLS, run_tool, observe=observe)

    return CapabilityBundle(
        memory=RecallMemory(mind, persona, exclude_turn_ids=exclude_turn) if memory else None,
        tools=tools,
        verify=MeterVerify(mind, context=context) if verify else None,
        insight=TwoSidedInsight(mind, persona, tail=tail, exclude_turn_ids=exclude_turn) if insight else None,
        host=extra,
    )


__all__ = ["bundle_for_companion", "RecallMemory", "HostTools", "MeterVerify", "TwoSidedInsight"]