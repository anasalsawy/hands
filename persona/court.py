"""Court — the headless deck of concurrent desktop hands (owner, 2026-10-24).

    "the app is cards, each card is a live desktop, and many run at once"

This module is the **UI-agnostic element layer** for that product. It knows
nothing about React, Tauri or any pixel; it only exposes a small controller
(``HandsDeck``) plus a typed event stream, so *any* front end binds to the same
contract:

    create court            deck.create(persona_id, title=..., kind=...)
    spawn desktop instance  deck.spawn(persona_id, court_id=...)        # engine / stream
    list courts             deck.list(persona_id)
    get one                 deck.get(persona_id, court_id)
    attach a lane           deck.attach_lane(persona_id, court_id, lane)
    send a goal             deck.send(persona_id, court_id, goal)
    cancel                  deck.cancel(persona_id, court_id)
    voice hooks             deck.voice(persona_id, court_id, ...)
    event stream            deck.subscribe(persona_id) / deck.events_since(...)

A **Court** is one card: a *desktop instance* the mind can drive. It fuses the
three real elements that already exist in the repo:

    - the **Link**   (persona/link.py)      — how work reaches an agent/engine
    - a **lane**     (persona/mind.py lanes) — the R-line of work it runs
    - a **surface**  (persona/gateway/computer) — the live JPEG stream the card shows

Nothing here re-implements those; it orchestrates them and gives them one name.

Wire surface (mounted by persona/api/app.py and stitch.py):

    GET    /v1/courts                       deck.list      (all cards)
    POST   /v1/courts                       deck.create
    GET    /v1/courts/{court_id}            deck.get
    DELETE /v1/courts/{court_id}            deck.cancel + forget
    POST   /v1/courts/{court_id}/spawn      spawn desktop instance (engine lane)
    POST   /v1/courts/{court_id}/goal       send a goal to the court's lane
    POST   /v1/courts/{court_id}/cancel     stop the work, keep the card
    POST   /v1/courts/{court_id}/voice      voice hook (speak / listen toggle)
    GET    /v1/courts/{court_id}/events     SSE of this deck's typed events
    WS     /v1/court-events                 one socket, every card's events

Event kinds (typed; a client switches on ``kind``):

    court.created    court.updated    court.removed
    court.spawned    court.attached   court.goal
    court.cancelled  court.status     court.voice
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import base64
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

# Court kinds — what the card *is*. All are desktop instances; the kind only
# tells the UI which surface to prefer first.
KINDS = ("desktop", "window", "region", "browser")

STATUSES = ("empty", "spawning", "running", "waiting", "done", "failed", "stopped")

# The motor a card gets when it has not been given a named Link. Concurrency is
# the mind's decision, so a goal alone must be enough to start one.
DEFAULT_LINK = "hands-engine"

# How long a spawn gets to prove the engine is on /v1/hands before we call it.
SPAWN_ACK_TIMEOUT = 75.0

# The mind may open this many engines for one goal. Not a user setting: an
# operator guardrail, so a runaway plan cannot fork the machine.
MAX_SPLIT_PARTS = max(1, int(os.environ.get("SAWII_MAX_SPLIT_PARTS", "4")))

# How much work one R line may take on its own card before it must report.
MAX_LANE_STEPS = max(1, int(os.environ.get("SAWII_MAX_LANE_STEPS", "8")))
GOAL_STEP_TIMEOUT = float(os.environ.get("SAWII_GOAL_STEP_TIMEOUT", "90"))
L_VISION_TIMEOUT = float(os.environ.get("SAWII_L_VISION_TIMEOUT", "12"))
L_VISION_PROMPT = (
    "You are Persona's L side looking at one R lane's current screen. You do not act. "
    "In at most 4 short lines say what is actually visible that matters for the goal: "
    "page/app state, key controls or text, errors, and whether the last action seems to "
    "have worked. Say 'unclear' instead of guessing.")
# A tool with no answer in this long means the hand went quiet: report it.
GOAL_TOOL_TIMEOUT = float(os.environ.get("SAWII_GOAL_TOOL_TIMEOUT", "300"))


def _split_by_shape(goal: str) -> list[tuple[str, str]]:
    """Offline split: the goal's own punctuation, when there is no planner model.

    Deliberately conservative — a single sentence stays one line of work.
    """
    import re as _re

    parts = [p.strip(" .\t") for p in _re.split(r"\n+|;|\band\b|\bthen\b", goal) if p.strip(" .\t")]
    parts = [p for p in parts if len(p) > 3][:MAX_SPLIT_PARTS]
    if len(parts) < 2:
        return []
    return [(p[:80], p) for p in parts]


def _now() -> float:
    return time.time()


@dataclass
class Court:
    """One card: a desktop instance the mind can watch and drive."""

    id: str
    persona_id: int
    title: str = ""
    kind: str = "desktop"

    # The live surface this card shows (persona/gateway/computer target).
    target: dict = field(default_factory=dict)

    # The lane of work currently attached (a row id from mind.lanes).
    lane: str = ""
    link: str = ""                      # registered Link name that owns the work
    goal: str = ""                      # the last goal sent to this court

    status: str = "empty"
    error: str = ""

    # The engine process/handle, if this deck spawned one (kept opaque here).
    worker: Any = None

    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    # Anything a surface wants to remember but the deck should not interpret.
    meta: dict = field(default_factory=dict)

    def view(self) -> dict:
        """The serialisable card a front end binds to (no live handles)."""
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "title": self.title,
            "kind": self.kind,
            "target": dict(self.target),
            "lane": self.lane,
            "link": self.link,
            "goal": self.goal,
            "status": self.status,
            "error": self.error,
            "running": self.status in ("spawning", "running", "waiting"),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "meta": dict(self.meta),
        }


class HandsDeck:
    """The deck of courts for every persona. One instance lives on the mind.

    It is deliberately dependency-light: it holds ``mind`` but only calls it
    through the small, already-public surfaces (``mind.lanes``, ``mind.links``,
    ``mind.hands``, ``mind.bus``). A different front end, or a test, can drive
    a deck over a stand-in for each of those.
    """

    def __init__(self, mind, *, max_courts: int = 24):
        self.mind = mind
        self.max_courts = max_courts
        self._courts: dict[str, Court] = {}
        # Strong references to the R loops driving each card's engine. The loop
        # holds only a weak reference to a bare create_task, and a goal silently
        # vanishing mid-flight is the exact bug this deck must not repeat.
        self._work_tasks: set[asyncio.Task] = set()
        self._send_locks: dict[str, asyncio.Lock] = {}
        # Per-persona live event fan-out (each subscriber gets its own queue).
        self._subs: dict[int, list[asyncio.Queue]] = {}

    # ───────────────────────────────────────────────────────── event stream

    def subscribe(self, persona_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs.setdefault(persona_id, []).append(q)
        return q

    def unsubscribe(self, persona_id: int, q: asyncio.Queue) -> None:
        subs = self._subs.get(persona_id)
        if not subs:
            return
        try:
            subs.remove(q)
        except ValueError:
            pass
        if not subs:
            self._subs.pop(persona_id, None)

    def emit(self, court: Court, kind: str, **extra) -> dict:
        """Publish one typed event to every subscriber *and* the mind bus.

        Returning the event lets a caller also stream it back on its own HTTP
        response. Publishing on ``mind.bus`` means SSE clients already watching
        a persona see deck changes without opening a second connection.
        """
        ev = {"kind": kind, "court": court.view(), "at": _now(), **extra}
        for q in list(self._subs.get(court.persona_id, [])):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass
        bus = getattr(self.mind, "bus", None)
        if bus is not None:
            try:
                bus.publish(court.persona_id, {"type": "court", **ev})
            except Exception:  # noqa: BLE001 - a deck must not die on a bus hiccup
                log.debug("court bus publish failed", exc_info=True)
        return ev

    # ───────────────────────────────────────────────────────────── courts

    def list(self, persona_id: int) -> list[dict]:
        mine = [c for c in self._courts.values() if c.persona_id == persona_id]
        for c in mine:
            self._reconcile(c)
        rows = [c.view() for c in mine]
        rows.sort(key=lambda c: c["created_at"])
        return rows

    def get(self, persona_id: int, court_id: str) -> Court | None:
        c = self._courts.get(court_id)
        c = c if c and c.persona_id == persona_id else None
        if c is not None:
            self._reconcile(c)
        return c

    def _reconcile(self, c: Court) -> None:
        """A card never reports a motor the hub cannot see.

        Called on every read. ``running`` means "a runner for my lane is
        connected right now"; anything else is reported as it truly is, with
        the reason attached. The lie the product must never tell is a card that
        looks busy with nothing driving it.
        """
        if c.status not in ("running", "waiting", "spawning"):
            return
        if not c.lane:
            c.status = "failed"
            c.error = c.error or "court has no lane: nothing was started for this goal"
            c.updated_at = _now()
            return
        hub = getattr(self.mind, "hands", None)
        if hub is None:
            return  # a deck with no hub has nothing to reconcile against
        pid = int(getattr(c, "persona_id", 0) or 0)
        try:
            seen = hub.runners.get(hub._key(pid, self._place(c), c.lane)) is not None
        except Exception:  # noqa: BLE001 - a broken probe must not invent failure
            log.debug("court %s hub probe failed", c.id, exc_info=True)
            return
        c.meta["connected"] = seen
        if seen:
            if c.status != "running":
                self._touch(c, status="running")
                self.emit(c, "court.status", status="running", lane=c.lane, motor="live")
            return
        # The lane exists but its hand is gone.
        link = None
        links = getattr(self.mind, "links", None)
        if links is not None and hasattr(links, "get") and c.link:
            link = links.get(c.link)
        why = ""
        if link is not None and hasattr(link, "lane"):
            linked = link.lane(c.lane)
            why = str(getattr(linked, "error", "") or "") if linked is not None else ""
            if linked is not None and getattr(linked, "status", "") == "done":
                # Engine finished its work and left: that is done, not failed.
                self._touch(c, status="done")
                self.emit(c, "court.status", status="done", lane=c.lane)
                return
        # A spawn still in flight is not a failure — it is on its way. This is
        # checked against the Link's own lane record, not the card's status, so
        # a read that lands between "status=running" and "engine connected"
        # cannot stamp a healthy card as failed.
        if self._lane_starting(c):
            return
        if c.status == "spawning" and _now() - c.updated_at < SPAWN_ACK_TIMEOUT:
            return  # still inside the window a spawn is allowed to take
        self._touch(c, status="failed",
                    error=(why or "the engine for this lane is not connected").strip()[:300])
        self.emit(c, "court.status", status=c.status, error=c.error, lane=c.lane, motor="gone")

    async def reconcile(self, persona_id: int) -> None:
        """Refresh every card of one persona against the live hub."""
        links = getattr(self.mind, "links", None)
        if links is not None and hasattr(links, "observe_hub"):
            try:
                await links.observe_hub()
            except Exception:  # noqa: BLE001
                log.debug("deck link hub observe failed", exc_info=True)
        for c in list(self._courts.values()):
            if c.persona_id == persona_id:
                self._reconcile(c)

    def create(
        self,
        persona_id: int,
        *,
        title: str = "",
        kind: str = "desktop",
        target: dict | None = None,
        meta: dict | None = None,
    ) -> Court:
        kind = kind if kind in KINDS else "desktop"
        mine = [c for c in self._courts.values() if c.persona_id == persona_id]
        if len(mine) >= self.max_courts:
            raise ValueError(f"too many courts ({self.max_courts}); close some first")
        cid = "court_" + uuid.uuid4().hex[:10]
        court = Court(
            id=cid,
            persona_id=persona_id,
            title=title or f"Court {len(mine) + 1}",
            kind=kind,
            target=dict(target or self._default_target(kind)),
            meta=dict(meta or {}),
        )
        self._courts[cid] = court
        self.emit(court, "court.created")
        return court

    def remove(self, persona_id: int, court_id: str) -> bool:
        c = self.get(persona_id, court_id)
        if c is None:
            return False
        # Stop the worker if we own one, but never block the caller on it.
        if c.worker is not None:
            self._schedule(c, self._stop_worker(c))
        browser = getattr(self.mind, "browser_hub", None)
        if c.kind == "browser" and c.lane and hasattr(browser, "close_lane"):
            self._schedule(c, browser.close_lane(persona_id, c.lane))
        self._courts.pop(court_id, None)
        self._send_locks.pop(court_id, None)
        self.emit(c, "court.removed")
        return True

    @staticmethod
    def _default_target(kind: str) -> dict:
        if kind == "browser":
            return {"kind": "browser"}
        if kind == "window":
            return {"kind": "window"}
        if kind == "region":
            return {"kind": "region"}
        return {"kind": "desktop"}

    def retarget(self, persona_id: int, court_id: str, target: dict) -> Court | None:
        c = self.get(persona_id, court_id)
        if c is None:
            return None
        c.target = dict(target or {})
        c.updated_at = _now()
        self.emit(c, "court.updated", changed="target")
        return c

    def _touch(self, c: Court, status: str | None = None, error: str | None = None) -> None:
        if status is not None:
            c.status = status
        if error is not None:
            c.error = error
        c.updated_at = _now()

    def _schedule(self, c: Court, coro) -> None:
        """Run a coroutine off the request path, if there is a loop to do it on."""
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            log.debug("no running loop to schedule %s for court %s", coro, c.id)

    # ─────────────────────────────────────────────── spawn a desktop instance

    async def spawn(self, persona_id: int, court_id: str, *, link_name: str = "hands-engine",
                    goal: str = "", lane: str = "") -> Court:
        """Give this court a real desktop instance: a lane plus its engine.

        This is the *concurrency* mechanism the product needs. Each call starts
        its own R-lane (via ``persona/link.py``) so several courts run at once,
        each driving a separate ``sawii_hands.py``. If no Link is registered
        (a deck used in a test, or Link disabled) the court still becomes a
        live *surface* card — it just has no motor behind it yet.
        """
        c = self.get(persona_id, court_id)
        if c is None:
            raise KeyError("unknown court")
        self._touch(c, status="spawning", error="")

        links = getattr(self.mind, "links", None)
        link = links.get(link_name) if links and hasattr(links, "get") else None
        if link is None:
            # No motor at all. The card is still a live surface, but it must
            # never claim to be running: nothing could possibly run.
            self._touch(c, status="failed",
                        error=f"no Link registered under {link_name!r}: nothing to run this goal")
            c.meta["motor"] = "none"
            if goal:
                c.goal = goal
            self.emit(c, "court.spawned", motor="none", status=c.status, error=c.error)
            return c

        # A lane the mind already opened keeps its own id: the lane registry, the
        # card and the engine must all speak the same name or the R line and the
        # motor never meet. Otherwise mint a fresh one.
        lane_id = str(lane or "") or "ct" + uuid.uuid4().hex[:10]
        c.lane = lane_id
        c.link = link_name
        c.meta["motor"] = link.agent.strategy if getattr(link, "agent", None) else "link"
        if goal:
            c.goal = goal

        # Spawn the engine AS THIS PERSONA. The hub keys runners by
        # (persona, place, lane); a hand that checks in under another persona is
        # unreachable by this lane's tool calls, and the card would run forever
        # with no motor on the other end.
        persona_key = ""
        if hasattr(links, "persona_key"):
            persona_key = links.persona_key(persona_id)
        try:
            linked = await asyncio.wait_for(
                link.delegate(lane_id, goal or f"court {court_id}: {c.title}",
                              persona_id=persona_id, persona_key=persona_key),
                timeout=SPAWN_ACK_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            c.lane = ""
            self._touch(c, status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
            self.emit(c, "court.status", status=c.status, error=c.error)
            return c
        c.worker = linked

        # ``delegate`` for the spawn strategy only returns once the engine is
        # registered on /v1/hands (or it failed). Verify rather than trust: a
        # lane the hub cannot see is a failure, not a running card.
        try:
            lane_status = await link.status(lane_id)
        except Exception:  # noqa: BLE001
            lane_status = "failed"
        if lane_status == "failed":
            err = getattr(linked, "error", "") or "engine failed to start"
            self._touch(c, status="failed", error=err[:300])
            self.emit(c, "court.status", status=c.status, error=c.error, lane=lane_id)
            return c

        # Watch the motor: if it dies mid-goal, the card says so instead of
        # hanging at "running" with nothing driving it.
        linked.on_exit = self._motor_died(persona_id, c.id)
        if not self._motor_live(c):
            self._touch(c, status="waiting")
            log.info("court %s: lane %s engine started, not yet on /v1/hands", c.id, lane_id)
        else:
            c.meta["connected"] = True
            self._touch(c, status="running")
        self.emit(c, "court.spawned", motor=c.meta.get("motor", "link"), lane=lane_id,
                  connected=self._motor_live(c))
        return c

    async def divide(self, persona_id: int, c: Court) -> list[str]:
        """Split one goal into R-lines and give every line its own engine.

        This is the mind deciding concurrency, which is the whole design rule:
        nobody presses a spawn button. The person states a goal; the planner
        finds the parts that can truly run at once; each part becomes a real
        R-lane on ``mind.lanes``, and ``stitch.py`` puts one engine process
        behind each of those lanes. Returns the extra lane ids it opened.

        The card's own lane keeps part 1 and is driven with ONLY that part: if
        it were handed the whole goal, the parent and the new lane would both do
        every part and the same file would be written twice.
        """
        if not c.goal or not hasattr(self.mind, "lanes"):
            return []
        parts = await self._plan_parts(c.goal)
        if c.status == "stopped" or self._courts.get(c.id) is not c:
            return []
        if len(parts) < 2:
            return []
        # Part 1 belongs to this card; remember it so the parent does that piece
        # and nothing else.
        c.meta["my_part"] = parts[0][1]
        self.emit(c, "court.updated", changed="my_part")
        # The card's own lane already carries part 1; open a lane per extra part.
        opened: list[str] = []
        for topic, instruction in parts[1:]:
            if c.status == "stopped":
                break
            try:
                line = await self.mind.lanes.start(persona_id, topic, instruction)
            except Exception:  # noqa: BLE001 - the first lane is still real work
                log.exception("court %s: could not open a lane for %r", c.id, topic[:40])
                continue
            opened.append(line.id)
            card = self.create(persona_id, title=topic[:60], kind=c.kind)
            # Opening the line already put an engine behind it: ``stitch.py``
            # hooks ``mind.lanes.start`` and spawns there. Spawning a second one
            # for the same lane id leaves two processes on one lane, and stopping
            # the lane frees only the one the Link holds — the other sits on
            # /v1/hands forever. So wait for the motor that is already coming,
            # and start one only if it genuinely never arrives.
            await self._await_hub_lane(persona_id, line.id, self._place(c))
            await self.attach_lane(persona_id, card.id, line.id,
                                   link_name=c.link or DEFAULT_LINK, goal=instruction)
            if card.status == "failed":
                await self.spawn(persona_id, card.id,
                                 link_name=c.link or DEFAULT_LINK,
                                 goal=instruction, lane=line.id)
            if card.status == "failed":
                # The mind opened the lane but no engine came up. Say so on the
                # parent card rather than leaving a card that lies about work.
                self.emit(c, "court.status", status=c.status, error=card.error,
                          split=line.id, part=topic[:60])
                continue
            self._work(persona_id, card.id, line.id, instruction)
            # The deck shows the human label the mind gave this line (R2, R3...);
            # the raw lane id means nothing on a card.
            card.meta["lane_label"] = getattr(line, "label", "")
            self.emit(card, "court.updated", changed="lane_label")
            self.emit(c, "court.status", status=c.status, lane=c.lane,
                      split=line.id, part=topic[:60])
        if opened:
            log.info("court %s: goal split into %d extra R-lanes by the mind: %s",
                     c.id, len(opened), ", ".join(opened))
        return opened

    async def _plan_parts(self, goal: str) -> list[tuple[str, str]]:
        """Ask the mind for independent parts; fall back to the goal's own shape.

        Never invents work that was not asked for, and never returns more than
        MAX_SPLIT_PARTS. One part means "this is one line of work" — honest
        serial work beats a fake fan-out.
        """
        goal = str(goal or "").strip()
        if not goal:
            return []
        planner = getattr(self.mind, "a", None)
        if planner is not None and hasattr(planner, "chat"):
            try:
                reply = await asyncio.wait_for(planner.chat([
                    {"role": "system", "content": (
                        "Split this goal into parts that can genuinely run at the same "
                        "time on one computer. Return only JSON: "
                        '{"parts":[{"topic":"short label","instruction":"complete, '
                        'self-contained instruction"}]}. One part if the work is really '
                        "sequential or one action. Never add work that was not requested."
                    )},
                    {"role": "user", "content": goal[:2000]},
                ], max_tokens=700, temperature=0.1, json_mode=True), timeout=20)
                import json as _json
                obj = _json.loads(str(getattr(reply, "text", reply) or "").strip() or "{}")
                raw = obj.get("parts") if isinstance(obj, dict) else None
                parts = []
                for row in (raw if isinstance(raw, list) else []):
                    if not isinstance(row, dict):
                        continue
                    ins = str(row.get("instruction") or "").strip()[:2000]
                    if not ins:
                        continue
                    parts.append((str(row.get("topic") or ins)[:80], ins))
                    if len(parts) >= MAX_SPLIT_PARTS:
                        break
                if parts:
                    return parts
            except Exception:  # noqa: BLE001 - a mind with no model still splits
                log.debug("goal planner unavailable; using the goal's own shape",
                          exc_info=True)
        return _split_by_shape(goal)

    async def _await_hub_lane(self, persona_id: int, lane: str, place: str,
                              timeout: float = 0.0) -> bool:
        """Wait for a lane's runner to show up on the hub.

        The engine for a lane the mind just opened is started by the
        ``lanes.start`` hook in ``stitch.py``, in its own task. Attaching a card
        before that task lands reads "no hand" as "no hand will ever come" and
        stamps an honest-looking failure on a lane that is about to work. The
        hub is the only real proof, so that is what this polls.
        """
        hub = getattr(self.mind, "hands", None)
        if hub is None:
            return False
        deadline = _now() + (timeout or SPAWN_ACK_TIMEOUT)
        while _now() < deadline:
            try:
                if hub.runners.get(hub._key(persona_id, place, lane)) is not None:
                    return True
            except Exception:  # noqa: BLE001 - a hub we cannot read proves nothing
                return False
            await asyncio.sleep(0.2)
        return False

    async def _await_motor(self, c: Court, timeout: float = 0.0) -> bool:
        """Wait a moment for a motor that is already on its way.

        ``stitch.py`` spawns an engine the instant a lane opens, so a card that
        just took that lane needs only to wait — not start a rival process.
        Returns True the moment the hub can see the hand.
        """
        deadline = _now() + (timeout or SPAWN_ACK_TIMEOUT)
        while _now() < deadline:
            self._reconcile(c)
            if c.status == "running" or c.status == "failed":
                return c.status == "running"
            await asyncio.sleep(0.25)
        self._reconcile(c)
        return c.status == "running"

    def _lane_starting(self, c: Court) -> bool:
        """Is a motor for this card's lane still coming up?

        Checked against the Link's own lane record when there is one, and
        against the card's age otherwise: ``stitch.py`` spawns a lane's engine
        in a background task, so for a moment after a lane is attached there is
        no Link record yet. Failing a card inside that window would just be a
        different kind of lie.
        """
        links = getattr(self.mind, "links", None)
        if links is not None and hasattr(links, "get") and c.link and c.lane:
            link = links.get(c.link)
            linked = link.lane(c.lane) if hasattr(link, "lane") else None
            if linked is not None:
                return (getattr(linked, "status", "") in ("starting", "active")
                        and not getattr(linked, "connected", False))
        return _now() - c.updated_at < SPAWN_ACK_TIMEOUT

    def _motor_died(self, persona_id: int, court_id: str):
        """Callback the Link fires when this card's engine process exits."""
        def _died(rc, error: str = "") -> None:
            c = self.get(persona_id, court_id)
            if c is None:
                return
            c.meta["connected"] = False
            if c.status in ("running", "waiting", "spawning"):
                why = str(error or "").strip() or f"engine exited (code {rc})"
                self._touch(c, status="failed", error=why[:300])
                self.emit(c, "court.status", status=c.status, error=c.error,
                          lane=c.lane, motor="dead")
                log.warning("court %s lost its engine on lane %s: %s", c.id, c.lane, why[:160])
        return _died

    async def _stop_worker(self, c: Court) -> None:
        own = [task for task in self._work_tasks
               if task.get_name() == f"court-work-{c.id}" and task is not asyncio.current_task()]
        for task in own:
            task.cancel()
        if own:
            await asyncio.gather(*own, return_exceptions=True)
        links = getattr(self.mind, "links", None)
        link = links.get(c.link) if links and hasattr(links, "get") and c.link else None
        if link is not None and c.lane:
            try:
                await link.stop(c.lane)
            except Exception as exc:  # noqa: BLE001
                self._touch(c, status="failed", error=f"could not stop engine: {exc}"[:300])
                self.emit(c, "court.status", status=c.status, error=c.error)
                log.debug("court %s worker stop failed", c.id, exc_info=True)
        c.worker = None

    # ─────────────────────────────────────────── attach a lane (already running)

    async def attach_lane(self, persona_id: int, court_id: str, lane: str,
                          *, link_name: str = "", goal: str = "") -> Court | None:
        """Bind an existing lane of work to a card (e.g. a lane the mind opened)."""
        c = self.get(persona_id, court_id)
        if c is None:
            return None
        c.lane = lane
        if link_name:
            c.link = link_name
        elif not c.link:
            c.link = DEFAULT_LINK
        if goal:
            c.goal = goal
        # Claim "running" only if the lane really has a hand on it. A lane the
        # mind opened but never gave a motor is not a running card.
        live = bool(lane) and self._motor_live(c)
        c.meta["connected"] = live
        if not lane:
            self._touch(c, status="empty")
        elif live:
            self._touch(c, status="running", error="")
        elif self._spawn_in_flight(c):
            self._touch(c, status="spawning")
        else:
            self._touch(c, status="failed",
                        error=f"lane {lane} has no engine connected to /v1/hands")
        self.emit(c, "court.attached", lane=lane, connected=live, status=c.status)
        return c

    def _spawn_in_flight(self, c: Court) -> bool:
        """Is a motor for this card still coming up?"""
        links = getattr(self.mind, "links", None)
        if links is None or not hasattr(links, "get") or not c.link or not c.lane:
            return False
        link = links.get(c.link)
        linked = link.lane(c.lane) if hasattr(link, "lane") else None
        return getattr(linked, "status", "") in ("starting", "active")

    # ─────────────────────────────────────────────────────── goals & cancel

    def _link_for(self, c: Court):
        """The Link this card runs on: the one it already uses, else the default.

        A fresh card has no link name on it yet — that is not "no motor", it is
        "not decided yet", and deciding is the mind's job, not the caller's. So
        a goal on a bare card resolves to the registered engine instead of
        silently degrading the card into a picture with no hands.
        """
        links = getattr(self.mind, "links", None)
        if links is None or not hasattr(links, "get"):
            return None, (c.link or DEFAULT_LINK)
        name = c.link or DEFAULT_LINK
        link = links.get(name)
        if link is None:
            # The recorded name is stale (Link re-registered, or a test deck):
            # fall back to the default once rather than lose the motor.
            link = links.get(DEFAULT_LINK)
            name = DEFAULT_LINK if link is not None else name
        return link, name

    def _place(self, c: Court) -> str:
        """Which machine this card's motor runs on (the hub keys by place+lane)."""
        link, _ = self._link_for(c)
        place = getattr(getattr(link, "agent", None), "place", "") or ""
        return str(c.meta.get("place") or place or "computer")

    def _motor_live(self, c: Court) -> bool:
        """Does this card actually have a motor on the hub right now?"""
        if not c.lane:
            return False
        hub = getattr(self.mind, "hands", None)
        pid = int(getattr(c, "persona_id", 0) or 0)
        if hub is None or not pid:
            return True  # nothing to check against (a deck with no hub at all)
        try:
            return hub.runners.get(hub._key(pid, self._place(c), c.lane)) is not None
        except Exception:  # noqa: BLE001
            log.debug("court %s motor probe failed", c.id, exc_info=True)
            return True

    async def send(self, persona_id: int, court_id: str, goal: str) -> Court:
        if self.get(persona_id, court_id) is None:
            raise KeyError("unknown court")
        async with self._send_locks.setdefault(court_id, asyncio.Lock()):
            return await self._send(persona_id, court_id, goal)

    async def _send(self, persona_id: int, court_id: str, goal: str) -> Court:
        """Send a goal to a court.

        The design rule this honours: *the user states a goal; the mind decides
        the concurrency.* A goal is therefore enough on its own — if the card
        has no lane, or its engine has died, this attaches a lane AND a live
        engine. Only a card that already has a working motor is steered.
        """
        c = self.get(persona_id, court_id)
        if c is None:
            raise KeyError("unknown court")
        c.goal = goal
        c.updated_at = _now()
        link, link_name = self._link_for(c)

        if link is not None and c.lane and self._motor_live(c):
            working = any(task.get_name() == f"court-work-{c.id}" and not task.done()
                          for task in self._work_tasks)
            if working:
                try:
                    await link.steer(c.lane, goal)
                except Exception:  # noqa: BLE001
                    log.debug("court %s steer failed", c.id, exc_info=True)
                self.emit(c, "court.goal", goal=goal, lane=c.lane, motor="steer")
            else:
                # A connected motor can just be listening (Start with no goal).
                c.meta.pop("my_part", None)
                self._touch(c, status="running", error="")
                await self.divide(persona_id, c)
                if c.status in ("running", "waiting"):
                    own = str(c.meta.get("my_part") or "").strip() or goal
                    self._work(persona_id, c.id, c.lane, own)
                self.emit(c, "court.goal", goal=goal, lane=c.lane, motor="reuse")
            return c

        if link is not None:
            # No lane, or the motor is gone: a goal means "put hands on this".
            # The mind allocates the concurrency here; the client never asks.
            self.emit(c, "court.goal", goal=goal, motor="spawn")
            c = await self.spawn(persona_id, court_id, link_name=link_name, goal=goal)
            # Then let the mind fan the rest of the goal out on its own. This is
            # the whole product rule: the person states an intent, the mind
            # splits it into R-lines, and every R-line gets its own engine.
            await self.divide(persona_id, c)
            if c.lane and c.status in ("running", "waiting"):
                # After a split the parent owes only its own part; driving the
                # whole goal here would duplicate every child lane's work.
                own = str(c.meta.get("my_part") or "").strip() or c.goal
                self._work(persona_id, c.id, c.lane, own)
            return c

        # No Link registry at all (a bare deck in a test): record the goal, but
        # never claim a motor we do not have. "running" with nothing behind it
        # is a lie, and a lie is worse than an honest failure.
        c.meta["motor"] = "none"
        self._touch(c, status="failed",
                    error="no Link registered: this deck has no engine to run the goal")
        self.emit(c, "court.goal", goal=goal, motor="none", status=c.status)
        return c

    async def cancel(self, persona_id: int, court_id: str, *, forget: bool = False) -> Court | None:
        """Stop this court's work. Keeps the card (and its live surface) unless forgotten."""
        c = self.get(persona_id, court_id)
        if c is None:
            return None
        self._touch(c, status="stopped", error="")
        await self._stop_worker(c)
        lanes = getattr(self.mind, "lanes", None)
        if lanes is not None and c.lane and hasattr(lanes, "stop"):
            try:
                maybe = lanes.stop(c.lane)
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception:  # noqa: BLE001
                log.debug("court %s lane stop failed", c.id, exc_info=True)
        self.emit(c, "court.cancelled")
        if forget:
            return c if self.remove(persona_id, court_id) else c
        return c

    # ─────────────────────────────────────────────── driving the lane's engine

    def _work(self, persona_id: int, court_id: str, lane: str, goal: str) -> None:
        """Run one lane's goal through R, off the request path.

        A motor with nobody thinking through it is just a listening process.
        R is the thinker: it sees this lane's own tool set, decides the calls,
        and every call is addressed to this lane's runner. Held in a task set
        for the same reason the spawn is: a bare ``create_task`` can be garbage
        collected mid-flight and the goal would die without a word.
        """
        task = asyncio.create_task(self._drive(persona_id, court_id, lane, goal),
                                   name=f"court-work-{court_id}")
        self._work_tasks.add(task)
        task.add_done_callback(self._work_tasks.discard)
        task.add_done_callback(lambda t: self._work_crashed(persona_id, court_id, t))

    async def _retire(self, persona_id: int, c: Court, lane: str) -> None:
        """Put the lane's engine away once its work is genuinely finished.

        A runner that stays connected after its goal is done keeps
        ``/v1/hands`` advertising a hand with nothing behind the intent, and
        keeps a python process alive on the person's computer for no reason.
        Only the spawn strategy is retired — a hand the person started by hand
        in a terminal is theirs to keep, and we never kill what we did not start.
        """
        links = getattr(self.mind, "links", None)
        if links is None or not c.link:
            return
        link = links.get(c.link)
        if link is None or not hasattr(link, "stop"):
            return
        linked = link.lane(lane) if hasattr(link, "lane") else None
        if linked is None or getattr(linked, "process", None) is None:
            return  # not a process we spawned: leave the person's own hand alone
        try:
            await link.stop(lane)
            log.info("court %s: lane %s engine retired after finishing", c.id, lane)
        except Exception:  # noqa: BLE001 - a stuck engine must not undo a real result
            log.debug("court %s: could not retire lane %s", c.id, lane, exc_info=True)

    def _work_crashed(self, persona_id: int, court_id: str, task: asyncio.Task) -> None:
        """A R loop that died by exception must leave a true card behind."""
        exc = None
        try:
            exc = task.exception()
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            return
        if exc is None:
            return
        log.exception("court %s: the R line for this lane crashed", court_id, exc_info=exc)
        c = self._courts.get(court_id)
        if c is not None and c.status in ("running", "waiting", "spawning"):
            self._touch(c, status="failed",
                        error=f"the R line for this lane stopped unexpectedly "
                              f"({type(exc).__name__}: {exc})"[:300])
            self.emit(c, "court.status", status=c.status, error=c.error, lane=c.lane)

    async def _drive(self, persona_id: int, court_id: str, lane: str, goal: str) -> None:
        """The R loop for one lane: think, call this lane's hand, read, repeat."""
        hub = getattr(self.mind, "hands", None)
        lobe = getattr(self.mind, "a", None)
        c = self.get(persona_id, court_id)
        if c is None:
            return
        if hub is None or lobe is None or not hasattr(lobe, "chat"):
            self._touch(c, status="failed",
                        error="no inference lobe available: the engine has nobody to think for it")
            self.emit(c, "court.status", status=c.status, error=c.error, lane=lane)
            return
        place = self._place(c)
        try:
            runner = hub.runners.get(hub._key(persona_id, place, lane))
        except Exception:  # noqa: BLE001
            runner = None
        browser = getattr(self.mind, "browser_hub", None) if c.kind == "browser" else None
        if runner is None and browser is None:
            self._touch(c, status="failed", error="the engine for this lane never connected")
            self.emit(c, "court.status", status=c.status, error=c.error, lane=lane)
            return
        from .tars_loop import PLAN_NAME, PLAN_TOOL, run_tars_loop

        if browser is not None:
            # A browser card drives its own lane browser (the one its surface
            # streams), not a shell or a separate headless Chrome.
            from .gateway.browser.tools import BrowserTools
            schemas = [*BrowserTools(None).schemas(), PLAN_TOOL]
        else:
            schemas = [*list(getattr(runner, "tools", []) or []), PLAN_TOOL]

        async def browser_frame() -> dict:
            from .gateway.browser.session import read_page
            from .gateway.browser.tools import _render_snapshot
            one = await browser._get((persona_id, lane))
            snap = await read_page(one.session, screenshot=True, ground=False)
            frame = {"source": "browser", "lane": lane, "text": _render_snapshot(snap)[:4000]}
            if snap.screenshot_png:
                frame["image_url"] = "data:image/png;base64," + base64.b64encode(snap.screenshot_png).decode()
            return frame
        instructions = (
            f"You are Sawii's R line {lane}, working on one computer with the tools "
            "in front of you. Do the task with real tool calls, then say in one short "
            "line what you actually did. Never claim a tool ran unless it did. "
            f"When the next step depends on what a hand call returns, put {PLAN_NAME} "
            "first in the same batch as that call: give your confidence and the likely "
            "outcomes (ok/error or exact status) mapped to the next action, plus an "
            "ordered batch with each step's _expected_outcome. A step whose outcome the "
            "real receipt matches fires at once when you are confident; verify it after."
        )
        receipts: list[str] = []
        # The last tool-level failure this lane saw, and whether anything
        # succeeded after it. A goal whose hand died mid-flight must end
        # ``failed`` even if R politely explains that nothing happened: a card
        # reading "done" over work that never ran is the exact lie this deck
        # exists to prevent. A recovery (error, then a good call) clears it.
        last_tool_error = ""
        names = {tool["function"]["name"] for tool in schemas}
        steps = 0
        initial_goal = c.goal

        def active() -> Court:
            c = self.get(persona_id, court_id)
            if c is None or c.status == "stopped":
                raise asyncio.CancelledError
            return c

        async def infer(messages: list[dict]) -> dict:
            nonlocal steps
            c = active()
            steps += 1
            c.meta["steps"] = steps
            if steps > MAX_LANE_STEPS:
                raise RuntimeError("lane step limit reached")
            if c.goal and c.goal != initial_goal:
                messages = [*messages, {"role": "user", "content":
                            "Updated user guidance for this lane: " + c.goal[:4000]}]
            reply = await asyncio.wait_for(
                lobe.chat(messages, tools=schemas or None, max_tokens=700, temperature=0.2),
                timeout=GOAL_STEP_TIMEOUT)
            calls = []
            for tc in getattr(reply, "tool_calls", None) or []:
                args = _json.loads(tc.arguments or "{}")
                if tc.name not in names or not isinstance(args, dict):
                    raise RuntimeError("R requested an unadvertised tool or invalid arguments")
                calls.append({"id": tc.id, "name": tc.name, "arguments": args})
            return {"text": getattr(reply, "text", "") or "", "tool_calls": calls,
                    "finish_reason": getattr(reply, "finish_reason", "stop")}

        async def execute(name: str, args: dict) -> str:
            nonlocal last_tool_error
            c = active()
            timeline = getattr(self.mind, "shared_timeline", None)
            call_id = f"{lane}:{len(receipts) + 1}"
            if timeline is not None:
                await timeline.append(persona_id, producer=c.meta.get("lane_label") or lane,
                                      kind="tool_call", task_id=lane,
                                      payload={"tool": name, "args": str(args)[:1500],
                                               "call_id": call_id, "part": goal[:500]})
            out = await asyncio.wait_for(
                browser.run(persona_id, name, args, lane=lane) if browser is not None
                else hub.call(persona_id, name, args, lane=lane),
                                         timeout=GOAL_TOOL_TIMEOUT)
            text = str(out)
            receipts.append(f"{name}: {text[:200]}")
            try:
                parsed = _json.loads(text)
                bad = isinstance(parsed, dict) and bool(parsed.get("error") or parsed.get("isError"))
                last_tool_error = str(parsed.get("error") or "tool failed")[:300] if bad else ""
            except ValueError:
                last_tool_error = ""
            c.meta["last_tools"] = receipts[-3:]
            if timeline is not None:
                await timeline.append(persona_id, producer=c.meta.get("lane_label") or lane,
                                      kind="tool_result", task_id=lane,
                                      payload={"tool": name, "call_id": call_id, "part": goal[:500],
                                               "result": {"text": text[:2000]}})
            v_prefetch()
            return text[:8000]

        v_reads: dict[str, str] = {}
        v_next: asyncio.Task | None = None

        async def v_look() -> dict | None:
            # V: pull the lane's screen and have L read it (same Persona model,
            # read-only, no tools). An unchanged screen reuses L's last read.
            c = active()
            frame = await asyncio.wait_for(
                browser_frame() if browser is not None else hub.observe(persona_id, place, lane=lane),
                timeout=8 if browser is not None else 3)
            image = frame.get("image_url") if isinstance(frame, dict) else None
            if not image:
                return frame
            key = hashlib.sha1(str(image).encode()).hexdigest()
            read = v_reads.get(key, "")
            started = time.monotonic()
            if read:
                c.meta["l_vision_reused"] = int(c.meta.get("l_vision_reused", 0)) + 1
            else:
                try:
                    seen = await asyncio.wait_for(lobe.chat([
                        {"role": "system", "content": L_VISION_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"Goal: {goal[:600]}\n{str(frame.get('text') or '')[:600]}"},
                            {"type": "image_url", "image_url": {"url": image}}]},
                    ], tools=None, max_tokens=160, temperature=0.0), timeout=L_VISION_TIMEOUT)
                    read = str(getattr(seen, "text", "") or "").strip()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - R still gets the raw frame
                    c.meta["l_vision_error"] = f"{type(exc).__name__}: {exc}"[:200]
                c.meta["l_vision_ms"] = int((time.monotonic() - started) * 1000)
                if not read:
                    return frame
                v_reads.clear()
                v_reads[key] = read
                timeline = getattr(self.mind, "shared_timeline", None)
                if timeline is not None:
                    await timeline.append(persona_id, producer="L", kind="l_vision", task_id=lane,
                                          payload={"text": read[:1500], "lane": lane})
            c.meta["l_vision"] = read[:500]
            return {**frame, "text": f"L sees: {read[:1500]}\n{str(frame.get('text') or '')}".strip()}

        def v_prefetch() -> None:
            nonlocal v_next
            if v_next is not None:
                v_next.cancel()
            v_next = asyncio.create_task(v_look(), name=f"court-v-{c.id}")
            v_next.add_done_callback(lambda t: t.cancelled() or t.exception())

        async def observe() -> dict | None:
            nonlocal v_next
            task, v_next = v_next, None
            if task is not None:
                return await task
            return await v_look()

        c.meta["agent_loop"] = "omni-tars"
        try:
            try:
                said = await run_tars_loop(
                    goal=goal[:4000], instructions=instructions, tools=schemas,
                    infer=infer, execute=execute, observe=observe,
                    max_steps=MAX_LANE_STEPS,
                    timeout=max(GOAL_STEP_TIMEOUT, GOAL_TOOL_TIMEOUT) + 5,
                    autofire=os.getenv("SAWII_HANDS_AUTOFIRE", "1") != "0",
                    stats=active().meta.setdefault("spec", {}))
            finally:
                if v_next is not None:
                    v_next.cancel()
        except Exception as exc:  # noqa: BLE001 - a dead loop must be reported
            c = self.get(persona_id, court_id)
            if c is not None and c.status != "stopped":
                self._touch(c, status="failed", error=f"R loop failed ({type(exc).__name__}: {exc})"[:300])
                self.emit(c, "court.status", status=c.status, error=c.error, lane=lane)
                await self._retire(persona_id, c, lane)
            return
        c = self.get(persona_id, court_id)
        if c is not None and c.status in ("running", "waiting"):
            c.meta["result"] = said[:500]
            self._touch(c, status="failed" if last_tool_error else "done", error=last_tool_error)
            self.emit(c, "court.status", status=c.status, error=c.error, lane=lane, result=said[:300])
            await self._retire(persona_id, c, lane)

    # ───────────────────────────────────────────────────────────── voice hooks

    def voice(self, persona_id: int, court_id: str, *, action: str = "status",
              text: str = "", lane: str = "L") -> dict:
        """Voice hooks for a card. ``action`` is one of speak | listen | stop | status.

        The deck never owns the voice stack (that is ``persona/voice`` +
        ``voice_sidecar.py``); it only gives a card a *named handle* onto it, so
        a front end can wire a per-card mic/orb without knowing the internals.
        """
        c = self.get(persona_id, court_id)
        if c is None:
            raise KeyError("unknown court")
        action = action if action in ("speak", "listen", "stop", "status") else "status"
        c.meta["voice"] = {
            "action": action,
            "lane": lane,
            "text": text[:2000],
            "at": _now(),
            "websocket": f"/v1/voice?lane={lane}",
        }
        c.updated_at = _now()
        self.emit(c, "court.voice", action=action, lane=lane, text=text[:2000])
        return dict(c.meta["voice"])

    # ─────────────────────────────────────────────────────────────── helpers

    def snapshot(self, persona_id: int) -> dict:
        """Everything a front end needs to render the deck in one call."""
        cards = self.list(persona_id)
        return {
            "courts": cards,
            "count": len(cards),
            "running": sum(1 for c in cards if c["running"]),
            "kinds": list(KINDS),
            "statuses": list(STATUSES),
        }


def register_court_routes(app) -> None:
    """Mount the deck's HTTP + SSE + WebSocket surface on a Starlette app.

    Kept self-contained so both ``persona/api/app.py`` and ``stitch.py`` can
    attach it without knowing the deck's internals.
    """
    import json as _json

    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route, WebSocketRoute

    def deck_of(request_or_ws) -> HandsDeck | None:
        runtime = getattr(request_or_ws.app.state, "runtime", None)
        if runtime is None:
            return None
        deck = getattr(runtime.mind, "deck", None)
        return deck if isinstance(deck, HandsDeck) else None

    async def _persona(request_or_ws):
        runtime = getattr(request_or_ws.app.state, "runtime", None)
        if runtime is None:
            return None
        headers = request_or_ws.headers
        auth = headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else request_or_ws.query_params.get("key", "")
        if not key:
            return None
        from .identity import persona_by_key
        persona = await persona_by_key(runtime.mind.store, key)
        if persona is None:
            return None
        # Remember which key this persona authenticates with, so an engine we
        # start for one of its lanes can register on /v1/hands as THAT persona.
        # Only held in memory, never persisted — and the store keeps just a hash.
        links = getattr(runtime.mind, "links", None)
        if links is not None and hasattr(links, "remember_persona_key"):
            links.remember_persona_key(persona["id"], key)
        return persona

    def _err(status: int, msg: str) -> JSONResponse:
        return JSONResponse({"error": {"message": msg}}, status_code=status)

    async def _body(request: Request) -> dict:
        raw = await request.body()
        if not raw:
            return {}
        data = _json.loads(raw)
        return data if isinstance(data, dict) else {}

    async def courts_index(request: Request):
        deck = deck_of(request)
        if deck is None:
            return _err(503, "the deck is not mounted")
        persona = await _persona(request)
        if persona is None:
            return _err(401, "invalid persona key")
        if request.method == "GET":
            return JSONResponse(deck.snapshot(persona["id"]))
        try:
            data = await _body(request)
        except (ValueError, _json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        try:
            court = deck.create(
                persona["id"],
                title=str(data.get("title") or "")[:80],
                kind=str(data.get("kind") or "desktop"),
                target=data.get("target") if isinstance(data.get("target"), dict) else None,
                meta=data.get("meta") if isinstance(data.get("meta"), dict) else None,
            )
        except ValueError as exc:
            return _err(409, str(exc))
        auto = bool(data.get("spawn"))
        if auto:
            await deck.spawn(persona["id"], court.id, goal=str(data.get("goal") or ""))
        return JSONResponse(court.view(), status_code=201)

    async def court_one(request: Request):
        deck = deck_of(request)
        if deck is None:
            return _err(503, "the deck is not mounted")
        persona = await _persona(request)
        if persona is None:
            return _err(401, "invalid persona key")
        cid = request.path_params["court_id"]
        if request.method == "DELETE":
            ok = deck.remove(persona["id"], cid)
            return JSONResponse({"ok": ok}, status_code=200 if ok else 404)
        court = deck.get(persona["id"], cid)
        if court is None:
            return _err(404, "unknown court")
        return JSONResponse(court.view())

    async def court_spawn(request: Request):
        deck = deck_of(request)
        persona = await _persona(request)
        if deck is None or persona is None:
            return _err(503 if deck is None else 401, "deck not mounted" if deck is None else "invalid persona key")
        try:
            data = await _body(request)
        except (ValueError, _json.JSONDecodeError):
            data = {}
        try:
            court = await deck.spawn(
                persona["id"], request.path_params["court_id"],
                link_name=str(data.get("link") or "hands-engine"),
                goal=str(data.get("goal") or ""),
            )
        except KeyError:
            return _err(404, "unknown court")
        return JSONResponse(court.view(), status_code=202)

    async def court_goal(request: Request):
        deck = deck_of(request)
        persona = await _persona(request)
        if deck is None or persona is None:
            return _err(503 if deck is None else 401, "deck not mounted" if deck is None else "invalid persona key")
        try:
            data = await _body(request)
        except (ValueError, _json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        goal = str(data.get("goal") or data.get("instruction") or "").strip()
        if not goal:
            return _err(400, "goal is required")
        try:
            court = await deck.send(persona["id"], request.path_params["court_id"], goal)
        except KeyError:
            return _err(404, "unknown court")
        return JSONResponse(court.view(), status_code=202)

    async def court_cancel(request: Request):
        deck = deck_of(request)
        persona = await _persona(request)
        if deck is None or persona is None:
            return _err(503 if deck is None else 401, "deck not mounted" if deck is None else "invalid persona key")
        try:
            data = await _body(request)
        except (ValueError, _json.JSONDecodeError):
            data = {}
        court = await deck.cancel(persona["id"], request.path_params["court_id"],
                                  forget=bool(data.get("forget")))
        if court is None:
            return _err(404, "unknown court")
        return JSONResponse(court.view())

    async def court_voice(request: Request):
        deck = deck_of(request)
        persona = await _persona(request)
        if deck is None or persona is None:
            return _err(503 if deck is None else 401, "deck not mounted" if deck is None else "invalid persona key")
        try:
            data = await _body(request)
        except (ValueError, _json.JSONDecodeError):
            data = {}
        try:
            out = deck.voice(
                persona["id"], request.path_params["court_id"],
                action=str(data.get("action") or "status"),
                text=str(data.get("text") or ""),
                lane=str(data.get("lane") or "L"),
            )
        except KeyError:
            return _err(404, "unknown court")
        return JSONResponse(out)

    async def court_events(request: Request):
        """SSE of this persona's deck events (typed)."""
        deck = deck_of(request)
        persona = await _persona(request)
        if deck is None or persona is None:
            return _err(503 if deck is None else 401, "deck not mounted" if deck is None else "invalid persona key")
        pid = persona["id"]
        q = deck.subscribe(pid)

        async def gen():
            try:
                yield "retry: 3000\n\n"
                # Prime with the current deck so a late client is never blank.
                yield "event: snapshot\ndata: " + _json.dumps(deck.snapshot(pid)) + "\n\n"
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: {ev.get('kind', 'court')}\ndata: {_json.dumps(ev)}\n\n"
            finally:
                deck.unsubscribe(pid, q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def court_events_ws(websocket):
        deck = deck_of(websocket)
        if deck is None:
            await websocket.close(code=1013)
            return
        await websocket.accept()
        persona = await _persona(websocket)
        if persona is None:
            await websocket.send_json({"type": "error", "detail": "invalid persona key"})
            await websocket.close(code=4401)
            return
        pid = persona["id"]
        q = deck.subscribe(pid)
        try:
            await websocket.send_json({"kind": "snapshot", **deck.snapshot(pid)})
            while True:
                ev = await q.get()
                await websocket.send_json(ev)
        except Exception:  # noqa: BLE001 - disconnect or bad frame, either way done
            pass
        finally:
            deck.unsubscribe(pid, q)

    app.router.routes.extend([
        Route("/v1/courts", courts_index, methods=["GET", "POST"]),
        Route("/v1/courts/{court_id}", court_one, methods=["GET", "DELETE"]),
        Route("/v1/courts/{court_id}/spawn", court_spawn, methods=["POST"]),
        Route("/v1/courts/{court_id}/goal", court_goal, methods=["POST"]),
        Route("/v1/courts/{court_id}/cancel", court_cancel, methods=["POST"]),
        Route("/v1/courts/{court_id}/voice", court_voice, methods=["POST"]),
        Route("/v1/courts/{court_id}/events", court_events, methods=["GET"]),
        WebSocketRoute("/v1/court-events", court_events_ws),
    ])
