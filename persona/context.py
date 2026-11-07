"""The context stream - how one continuous line of thought is assembled.

    spine          the persona definition (never changes unless the user edits it)
    self-narrative A's own first-person account, rewritten only at sleep
    --------------------------------------------- stable above, live below
    time           measured clocks
    adjutant brief B's conscience note, REQUIRED corrections, due/open loops, integrity record
    observer view  C's longitudinal read of the person (hedged)
    own mood       A's current chord
    recall slice   stitched 3-layer memories, labeled and dated (appended last)
    verbatim tail  the last exchanges, from the server-side journal (any device)
    now            the person's message

One system message (stable part first, live part last) so it works with chat
templates that only accept a leading system turn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import emotion
from .brains.prompts import A_BASE
from .profiles import persona_frame
from .store import jload
from .timesense import Clock, fmt_duration, local_dt


@dataclass
class Scratch:
    """Ephemeral per-persona state. Never written to the database by design:
    the conscience must not become a memory that cites itself."""

    conscience: str = ""
    correction: str = ""
    sudden: str = ""          # B found it after A had already answered ("oh wait...")
    psyche_brief: str = ""      # the psyche lobe's (B) brief -> read by the verifier (C)
    dissent: str = ""         # the psyche lobe reads the person differently from the verifier
    meter_carryforward: str = ""  # one line of reviewer feedback for next turn; classes + fixes, never the score
    lane_questions: str = ""  # pending questions from Maya's background working lane
    dropped: list = field(default_factory=list)  # things they said that A did not answer
    last_consults: list = field(default_factory=list)
    aging: dict = field(default_factory=dict)  # last weight update C's verdict asked for


def integrity_block(integ: dict, calibration: str = "") -> str:
    if not integ.get("checked"):
        return ""
    lines = [f"Your verification record: {integ['green_streak']} clean answers in a row "
             f"({int((integ['clean_rate'] or 0) * 100)}% clean over {integ['checked']} checked)."]
    for v in integ.get("recent", []):
        if v["level"] != "GREEN":
            lines.append(f"  - recently {v['level']}: {v['rationale'][:200]}")
    if calibration:
        lines.append("Where you tend to be wrong (be careful here): " + calibration)
    lines.append("Every answer is checked after you give it. Care about being right; never argue with the record.")
    return "\n".join(lines)


_ASKS_ABOUT_LOOPS = re.compile(
    r"\b(?:tasks?|to-?dos?|reminders?|remind(?:ed)?|deadlines?|schedule|agenda|pending|"
    r"follow[- ]?ups?|promised?|forget|forgot|outstanding|left to do|on my plate|open loops?)\b", re.I)
_LOOP_FILLER = frozenset(
    "about after again also been before being could does doing from going have here into just know "
    "like lets make maybe more need really should some something talk tell than that their them then "
    "there these they thing things think this today tomorrow want what when where which will with "
    "would your speak plan".split())
# A reminder this close to now is news on any message.
LOOP_DUE_WINDOW_S = 3600.0


def _loop_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(w) >= 4 and w not in _LOOP_FILLER}


def relevant_loops(loops: list[dict], user_text: str, at: float) -> list[dict]:
    """The loops that bear on this message. A greeting gets none."""
    if not loops or _ASKS_ABOUT_LOOPS.search(user_text or ""):
        return list(loops or [])
    said = _loop_words(user_text)
    return [lp for lp in loops
            if (lp.get("due_at") and abs(lp["due_at"] - at) <= LOOP_DUE_WINDOW_S)
            or (said & _loop_words(lp.get("text", "")))]


def loops_block(loops: list[dict], tz: str, at: float) -> str:
    if not loops:
        return ""
    lines = ["Open loops your adjutant is tracking that bear on this message (use them only as far as "
             "they help with it):"]
    for lp in loops[:12]:
        due = ""
        if lp["due_at"]:
            when = local_dt(lp["due_at"], tz).strftime("%a %d %b %H:%M")
            due = f" - due {when}" + (" (OVERDUE)" if lp["due_at"] < at else
                                      f" (in {fmt_duration(lp['due_at'] - at)})")
        lines.append(f"  #{lp['id']} [{lp['kind']}] {lp['text']}{due}")
    return "\n".join(lines)


def build_messages(*, persona: dict, clock: Clock, scratch: Scratch, integrity: dict, loops: list[dict],
                   recall_text: str, tail: list[dict], user_text: str, at: float,
                   on_mind: list[str] | None = None, rules: list[dict] | None = None,
                   knowledge: str = "", passages: str = "", picture: str = "") -> list[dict]:
    # What she is given to remember with, logged at the one place it is all
    # together. The answer to "how did she know that instantly" is here: which
    # blocks were in front of her, how big each was, and how many past turns
    # were sitting in the open as plain messages.
    from . import trace as _trace
    _trace.event("context_parts", picture_chars=len(picture), picture_lines=picture.count("\n") + 1 if picture else 0,
                 recall_chars=len(recall_text), passages_chars=len(passages),
                 knowledge_chars=len(knowledge), tail_turns=len(tail),
                 tail_user_chars=sum(len(str(t.get("user_text") or "")) for t in tail),
                 tail_reply_chars=sum(len(str(t.get("reply_text") or "")) for t in tail),
                 on_mind=len(on_mind or []), rules=len(rules or []),
                 loops=len(loops or []), correction=bool(scratch.correction),
                 narrative_chars=len(persona.get("narrative") or ""),
                 psych_chars=len(persona.get("psych_profile") or ""))
    tz = persona.get("timezone") or "UTC"
    stable = [A_BASE, "", persona_frame(persona.get("definition", ""), persona.get("name", ""))]
    if persona.get("user_name"):
        stable.append(f"The person you are talking with is {persona['user_name']}.")
    # The picture goes in the CACHED prefix on purpose: it is the one piece of
    # memory that is always worth its tokens, because it is what makes "hi"
    # land like someone who knows you. It changes rarely, so it costs nothing
    # per message, and nothing on the turn has to wait for an embedding.
    if picture:
        stable += ["", "=== the picture of them (yours to keep current; every line came from what they said) ===",
                   picture]
    if persona.get("narrative"):
        stable += ["", "Your own account of yourself, written by you at your last sleep (continue this story):",
                   persona["narrative"]]
    if knowledge:
        stable += ["", knowledge]
    live = ["", "=== now ===", clock.block()]
    if passages:
        live += ["", passages]
    brief = []
    if scratch.correction:
        brief.append(f"CORRECTION REQUIRED (from your adjutant, with evidence): {scratch.correction}")
    if scratch.conscience:
        brief.append(f"Private note from your adjutant: {scratch.conscience}")
    if scratch.meter_carryforward:
        brief.append(scratch.meter_carryforward)
    if scratch.lane_questions:
        brief.append(scratch.lane_questions)
    if rules:
        brief.append("Their standing requests (keep them every time):\n" + "\n".join(f"  - {r['text']}" for r in rules))
    if scratch.dropped:
        brief.append("Last time they also said this and you never answered it - pick it up if it still matters:\n"
                     + "\n".join(f'  - "{d}"' for d in scratch.dropped))
    lb = loops_block(relevant_loops(loops, user_text, at), tz, at)
    if lb:
        brief.append(lb)
    ib = integrity_block(integrity, persona.get("calibration", ""))
    if ib:
        brief.append(ib)
    if brief:
        live += ["", "=== your adjutant (B), privately ==="] + brief
    if persona.get("psych_profile"):
        live += ["", "=== your adjutant's (B) read of them - hunches, never labels ===", persona["psych_profile"]]
    if scratch.dissent:
        live += ["", "=== your two inner minds disagree ===",
                 f"Your psyche (B) reads this differently from your verifier (C). B: {scratch.dissent}",
                 "If it matters, you may say you are of two minds - honestly, in your own words."]
    if persona.get("habits"):
        live += ["", "=== habits your adjutant (B) sees in you that you cannot see yourself ===", persona["habits"]]
    mood = jload(persona.get("mood"), {})
    if mood:
        live += ["", f"How you have been feeling: {emotion.render(mood)}."]
    # The EXACT chord you ended the last turn on. This is a short-term
    # carrier (one turn), distinct from .mood (long-term EMA). It is NOT
    # an instruction to perform the feeling — it is background data about
    # where you left off, same as the user's recent mood below. The next
    # beat can continue from there, or shift naturally based on what the
    # person just said. Loaded on top of mood so you have both the slow
    # baseline and the immediate prior tone.
    last_a_chord = jload(persona.get("last_a_chord"), {})
    if last_a_chord:
        live += [f"The chord you ended your last turn on: {emotion.render(last_a_chord)}."]
    user_mood = jload(persona.get("user_mood"), {})
    if user_mood:
        live += [f"How they have seemed lately: {emotion.render(user_mood)}."]
    # The last chord B read on the user (one turn ago). Same shape as above
    # for the user side — short-term carrier so B's next read doesn't start
    # from zero. (Set by life/subconscious.ingest on the previous turn.)
    last_chord = jload(persona.get("last_chord"), {})
    if last_chord:
        live += [f"How they seemed at the close of the previous turn: {emotion.render(last_chord)}."]
    if scratch.sudden:
        live += ["", "=== came back to you right after you last answered ===",
                 "Your adjutant found this after you had finished talking. If you have not mentioned it yet and "
                 "it fits, you may; otherwise let it go.", scratch.sudden]
    if on_mind:
        live += ["", "=== on your mind (what your other minds kept thinking about after you last talked) ===",
                 "Bring one up only at a natural moment, never forced:", *[f"• {t}" for t in on_mind]]
    if recall_text:
        live += ["", "=== memories ===", recall_text]
    messages = [{"role": "system", "content": "\n".join(stable + live)}]
    for t in tail:
        if t["kind"] == "outreach":
            if t["reply_text"]:
                messages.append({"role": "assistant", "content": t["reply_text"]})
            continue
        stamp = local_dt(t["started_at"], tz).strftime("%a %H:%M")
        via = f", via {t['channel']}" if t["channel"] not in ("app", "") else ""
        messages.append({"role": "user", "content": f"[{stamp}{via}] {t['user_text']}"})
        if t["reply_text"]:
            messages.append({"role": "assistant", "content": t["reply_text"]})
    messages.append({"role": "user", "content": user_text})
    return messages
