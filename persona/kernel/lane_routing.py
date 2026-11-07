"""lane_routing — the routing contract for the dual-lobe mind, server side.

Unity of mind means ONE shared timeline: every lobe sees system events, tool
results and the user's turns. That is awareness, not an invitation to act. Without
this contract a plain user message (no lane id) can be obeyed by several R lanes
at once, or ignored by the lane it was meant for.

The invariant this module encodes (mirrored verbatim by the host side, see
tasker-github/backend/utils/lane_bus.py — keep the two blocks in sync):

  * L is the only lobe the human speaks to directly. L classifies EVERY user
    message into one of: respond-in-chat / steer-to-lane / spawn-new-R /
    surface-needs_user-question, and routes as DATA (route_decision tool call or
    an `ln=<lane-id>=<instruction>` steering line), never as prose vibes.
  * R lanes act ONLY on instructions addressed to them (`[steering] ln=<id>=...`
    on the shared timeline, or the x-sawii-steer header) or on the task they were
    spawned for. A plain user message is READ-ONLY awareness for an R lane.
  * A busy R lane needs something from the human: it calls the `needs_user`
    host tool. That publishes a lane_needs_user event on the bus; L narrates it;
    the human's answer returns to THAT lane only, as `ln=<lane>=answer: ...`.

`split_steering()` is the executable half of the rule: given a turn's user
text, it says whether an R worker may treat it as an instruction (and which one),
or must park it for L.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

#: `[steering] ln=<lane>=<instruction>` — rail 1 of the persona_host_sdk contract.
_STEER_LINE = re.compile(r"\bln=([A-Za-z0-9]{1,16})=([^;\n]*)")


def router_block() -> str:
    """L's per-turn routing contract (mirror of lane_bus.router_block())."""
    return (
        "[lane routing] You are L, the only lobe the user speaks to directly. "
        "Execution lanes (R) share this timeline for awareness but MUST NOT act "
        "on a plain user message. Classify EVERY user message into exactly one "
        "of: respond-in-chat / steer-to-lane / spawn-new-R / surface-needs_user-"
        "question. Route as data, not vibes: call route_decision with "
        "{decision: chat|steer|new_task|surface_need, lane_id, instruction}, or "
        "emit steering as 'ln=<lane-id>=<instruction>'. A message that could "
        "apply to more than one lane gets ONE owner — the lane whose recent task "
        "context best matches — or you ask the user. NEVER broadcast the same "
        "instruction to several lanes. When a lane calls needs_user, its question "
        "appears in the conversation as a needs_user event: you MUST narrate it "
        "to the human in your next reply and never leave it unanswered; when the "
        "human answers, route the answer back to that lane only (answer_lane_need "
        "or ln=<lane>=answer: ...)."
    )


def lane_addressing_rules() -> str:
    """The R-lane addressing contract (mirror of lane_bus.lane_addressing_rules())."""
    return (
        "[lane addressing] Execution lanes act ONLY on instructions addressed to "
        "them — a '[steering] ln=<your-lane-id>=...' line (or the x-sawii-steer "
        "header) — or on the task they were spawned for. A plain user message has "
        "no lane id: read it for awareness (shared mind), but do NOT start, "
        "redirect, or repeat work because of it. L owns routing. If you need "
        "something from the human, call the needs_user tool; the answer comes "
        "back to you as ln=<your-lane-id>=answer: ..."
    )


def needs_block(asks: Iterable[dict]) -> str:
    """L-side surface for open needs_user questions (already narrated shapes)."""
    rows = [r for r in asks if str(r.get("question") or "").strip()][:8]
    if not rows:
        return ""
    lines = ["[lane needs_user] Your execution lanes are waiting on the human. "
             "You (L) MUST raise each one in your next conversational reply — "
             "never leave a needs_user question unanswered in narration. When the "
             "human answers, route it back to THAT lane only."]
    for r in rows:
        bits = [f"need_id={r.get('need_id', '')}", f"lane={r.get('lane') or 'unknown'}",
                "question: " + str(r["question"])[:512]]
        if r.get("blocker"):
            bits.append("blocker: " + str(r["blocker"])[:512])
        if r.get("choices"):
            bits.append("choices: " + " | ".join(str(c)[:120] for c in list(r["choices"])[:8]))
        lines.append("- " + "; ".join(bits))
    return "\n".join(lines)


def is_conversation_lobe(lobe: Optional[str]) -> bool:
    """Only L owns the human's words. Execution lobes are labelled R, R1, R2..."""
    return str(lobe or "").strip().upper().startswith("L")


def lane_prompt(lane_id: Optional[str], *, lobe: str = "R",
                task: str = "", steers: Iterable[str] = ()) -> str:
    """The per-turn system block one lane is allowed to obey.

    R gets the addressing contract plus ONLY the lines addressed to its own lane
    id; L gets the router contract. This is what "R lanes ignore direct user
    turns" looks like in text form.
    """
    if is_conversation_lobe(lobe):
        return router_block()
    head = lane_addressing_rules()
    you = normalize_lane(lane_id) or "?"
    lines = [head, f"[your lane id] {you}"]
    if task:
        lines.append("[your task — you may act on this] " + str(task)[:600])
    for s in steers:
        ln, txt = addressed_steer(str(s)[:600], you)
        if txt:
            lines.append(f"[steering for you] ln={ln}={txt}" if ln else
                         "[reminder] plain user turns are awareness only — do not act.")
    return "\n".join(lines)


# ── the executable half ───────────────────────────────────────────────────────

def normalize_lane(raw: Optional[str]) -> Optional[str]:
    """'abcd1234' | 'lnabcd1234' | 'LNABCD1234' | 'call_lnabcd1234_x' -> 'lnabcd1234'."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    m = re.match(r"^call_(ln[0-9a-f]{4,12})(?:_|$)", s)
    if m:
        return m.group(1)
    m = re.match(r"^(?:ln)?([0-9a-f]{4,12})$", s)
    if not m:
        return None
    body = m.group(1)
    return body if body.startswith("ln") else "ln" + body


def split_steering(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Split a timeline turn into ([(lane, instruction)], [plain user lines]).

    `ln=<lane>=...` segments are lane-addressed steering (rail 1) — each one
    belongs to exactly one lane. Everything else on the turn is a plain user
    message: conversational material for L, awareness-only for R.
    """
    t = str(text or "")
    hits = [(normalize_lane(m.group(1)) or m.group(1).lower(), m.group(2).strip())
            for m in _STEER_LINE.finditer(t)]
    if hits:
        rest = _STEER_LINE.sub("", t)
        rest = re.sub(r"\[steering\]", "", rest).strip(" ;\n")
        return [(ln, txt) for ln, txt in hits if txt], ([rest] if rest else [])
    stripped = t.strip()
    return [], ([stripped] if stripped else [])


def addressed_steer(text: str, lane_id: Optional[str]) -> tuple[Optional[str], str]:
    """The instruction addressed to `lane_id` on this turn, if any.

    A lane that passes this check may act. A lane that fails it must NOT: the
    plain-user branch is L's material, never an R instruction.
    """
    target = normalize_lane(lane_id)
    steers, plain = split_steering(text)
    if target:
        for ln, txt in steers:
            if normalize_lane(ln) == target and txt:
                return target, txt[:600]
    return target, ""


def may_act_on_user_turn(text: str, lane_id: Optional[str], *, lobe: str = "R") -> bool:
    """Can THIS lobe treat this turn's user text as something to do?

    L: always (it is the conversational lobe and the router).
    R: only when the turn carries a steering line addressed to its own lane id.
    A plain message — no ln= marker at all — returns False for R: that is the
    regression the user reported (two R lanes obeying one unaddressed message).
    """
    if is_conversation_lobe(lobe):
        return True
    target = normalize_lane(lane_id)
    if not target:
        return False
    return bool(addressed_steer(text, target)[1])


__all__ = ["router_block", "lane_addressing_rules", "needs_block", "lane_prompt",
           "split_steering", "addressed_steer", "may_act_on_user_turn",
           "is_conversation_lobe", "normalize_lane"]
