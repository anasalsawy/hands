"""Durable Execution Board: tool outcomes and live project state shared by L/R.

The response reads cached state only. Only evidence-bearing tool results write factual
status; user sketch notes are explicitly unverified until evidence arrives.
There is one store and one dual-kernel runtime, no extra agent or mode.
"""
from __future__ import annotations

import asyncio
import json
import hashlib
from typing import Any

from .picture import TaskPicture, PersonPicture, _one_line, _FAILED
from .sketch_graph import SketchGraph

# The person picture is the slowest-changing block in the cached prompt prefix, so it
# goes first: the KV cache breaks at the first byte that changes, and a biography line
# that rewrites rarely must not sit behind a graph that rewrites every tool step.
# It is still a budget - the whole picture is read every turn, so it stays small.
PERSON_PROMPT_MAX = 2500


def _cap_rows(text: str, limit: int) -> str:
    """Cut the picture to a size budget without ever splitting a row in half.

    Rows are the unit of the picture (one fact per line), so a partial row is a
    fact misquoted. Whole rows are dropped from the end instead.
    """
    out: list[str] = []
    used = 0
    for row in str(text or "").splitlines():
        if used + len(row) + 1 > limit:
            break
        out.append(row)
        used += len(row) + 1
    return "\n".join(out)


class PictureBoard:
    def __init__(self, timeline, bus=None):
        self.timeline = timeline
        self.bus = bus
        self._locks: dict[int, asyncio.Lock] = {}
        self._cached: dict[int, dict] = {}
        self._views: dict[int, dict] = {}
        self._prompt: dict[int, str] = {}

    def _install(self, pid: int, data: dict, revision: int) -> dict:
        view = self._view(data, revision)
        self._cached[pid] = data
        self._views[pid] = view
        self._prompt[pid] = self._format_prompt(view)
        return view

    async def warm(self) -> None:
        await self.timeline.initialize()
        rows = await self.timeline.store.all(
            "SELECT t.persona_id,t.seq,t.payload FROM shared_timeline t "
            "INNER JOIN (SELECT persona_id,MAX(seq) AS seq FROM shared_timeline "
            "WHERE kind='picture_state' GROUP BY persona_id) latest "
            "ON latest.persona_id=t.persona_id AND latest.seq=t.seq")
        for row in rows:
            try:
                self._install(int(row['persona_id']), json.loads(row['payload']), int(row['seq']))
            except (KeyError, TypeError, ValueError):
                continue

    def peek(self, pid: int) -> dict:
        return self._views.get(pid) or self._view({'task': {}, 'notes': [], 'person': {}}, 0)

    def for_prompt(self, pid: int) -> str:
        return self._prompt.get(pid, '')

    @staticmethod
    def _format_prompt(view: dict) -> str:
        # Stable biography FIRST, then the fast-changing board. Both live in the same
        # cached prefix; putting the slow block ahead keeps the prefix reusable when
        # only a graph row moves. The person picture arrives here because it is the
        # one thing about the user that must be in front of the voice every turn
        # without costing a memory call (ARCHITECTURE.md sections 4 and 5).
        lines: list[str] = []
        person_picture = _cap_rows(view.get('person_picture') or '', PERSON_PROMPT_MAX)
        if person_picture:
            lines.append('PERSON PICTURE (stable facts about them):')
            lines.append(person_picture)
        lines.append('EXECUTION BOARD: actual work and verified tool outcomes, '
                     'separate from the person picture above.')
        lines.append('A successful tool step does not prove completion of the whole task.')
        if view['task_picture']:
            lines.append(view['task_picture'])
        for n in view['graph']['nodes'][-65:]:
            lines.append(f"[{n.get('id')}] {n.get('label')} | status={n.get('status')} "
                         + str(n.get('result') or '')[:220])
        for e in view['graph']['edges'][-100:]:
            lines.append(f"{e.get('from')} -> {e.get('to')}: {e.get('label')}")
        if view['notes']:
            lines.append('Unverified working notes:')
            lines.extend(str(n)[:400] for n in view['notes'][-20:])
        return '\n'.join(lines)[:14000]

    async def _load(self, pid: int) -> tuple[dict[str, Any], int]:
        await self.timeline.initialize()
        row = await self.timeline.store.one(
            "SELECT seq,payload FROM shared_timeline "
            "WHERE persona_id=? AND kind='picture_state' ORDER BY seq DESC LIMIT 1", pid)
        if not row:
            return {"task": {}, "person": {}, "notes": []}, 0
        try:
            data = json.loads(row["payload"])
            if isinstance(data, dict):
                return data, int(row["seq"])
        except (TypeError, ValueError):
            pass
        return {"task": {}, "person": {}, "notes": []}, 0

    @staticmethod
    def _objects(data: dict) -> tuple[TaskPicture, PersonPicture, list[str]]:
        task = TaskPicture()
        for field in ("goal", "constraints", "working", "done", "broken",
                      "open", "known", "whole"):
            value = (data.get("task") or {}).get(field)
            if field == "goal":
                task.goal = _one_line(value, 240) if isinstance(value, str) else ""
            elif isinstance(value, list):
                setattr(task, field, [str(v)[:160] for v in value if isinstance(v, str)][:24])
        person = PersonPicture.parse(data.get("person") or {})
        notes = [str(v)[:400] for v in data.get("notes", [])
                 if isinstance(v, str) and v.strip()][:40]
        return task, person, notes

    @staticmethod
    def _data(task: TaskPicture, person: PersonPicture, notes: list[str]) -> dict:
        return {
            "task": {field: getattr(task, field) for field in (
                "goal", "constraints", "working", "done", "broken",
                "open", "known", "whole")},
            "person": json.loads(person.to_json()),
            "notes": notes[-40:],
        }

    @staticmethod
    def _view(data: dict, revision: int) -> dict:
        task, person, notes = PictureBoard._objects(data)
        return {
            "revision": revision,
            "task": PictureBoard._data(task, person, notes)["task"],
            "person": json.loads(person.to_json()),
            "notes": notes,
            "task_picture": task.render(),
            "person_picture": person.render(),
            "diagram": task.diagram(),
            "graph": SketchGraph(data.get("graph")).to_dict(),
        }

    async def get(self, pid: int) -> dict:
        return self.peek(pid)

    async def update(self, pid: int, edit: dict, *, expected_revision: int | None = None) -> dict:
        if not isinstance(edit, dict):
            raise ValueError("picture change must be an object")
        async with self._locks.setdefault(pid, asyncio.Lock()):
            data, revision = await self._load(pid)
            if expected_revision is not None and revision != expected_revision:
                raise RuntimeError("Picture changed; reload before editing")
            task, person, notes = self._objects(data)
            graph = SketchGraph(data.get("graph"))
            action = str(edit.get("action") or "")
            value = str(edit.get("text") or "").strip()
            if action == "goal":
                if not value: raise ValueError("goal cannot be empty")
                task.set_goal(value[:240])
                graph.ensure_goal(task.goal)
            elif action == "open":
                if not value: raise ValueError("open item cannot be empty")
                task.raise_open(value)
                node_id = "open_" + hashlib.sha256(value.encode()).hexdigest()[:12]
                if not graph._node(node_id):
                    graph.add(node_id, value, kind="question", status="unverified")
            elif action == "close":
                if not value: raise ValueError("open item required")
                task.close_open(value)
                for node in list(graph.nodes):
                    if node.get("kind") == "question" and node.get("label") == value:
                        graph.edit("remove_node", {"node_id": node["id"]})
            elif action == "note":
                if not value: raise ValueError("note cannot be empty")
                if value[:400] not in notes: notes.append(value[:400])
            elif action == "remove_note":
                notes = [note for note in notes if note != value]
            elif action == "areas":
                paths = edit.get("paths")
                if not isinstance(paths, list) or len(paths) > 24 or any(
                    not isinstance(v, str) for v in paths):
                    raise ValueError("paths must be a list of up to 24 strings")
                task.set_areas(paths)
            elif action == "person_patch":
                patch = edit.get("patch")
                if not isinstance(patch, dict) or any(k not in
                        ("who", "with", "weight", "doing", "since", "tend", "asked")
                        for k in patch):
                    raise ValueError("invalid person picture fields")
                person.merge(patch)
            elif action in ("add_node", "rename_node", "move_node", "remove_node", "connect_nodes"):
                graph.edit(action, edit)
            elif action == "tool_result":
                # Internal only. A user's typed note cannot mark work verified.
                raise ValueError("tool evidence cannot be manually asserted")
            else:
                raise ValueError("unknown picture action")
            payload = {**self._data(task, person, notes), "graph": graph.to_dict()}
            saved = await self.timeline.append(
                pid, producer="picture", kind="picture_state",
                payload=payload, supersedes_seq=revision or None)
            self._install(pid, payload, saved["seq"])
            if self.bus:
                self.bus.publish(pid, {"type": "execution_board_updated", "revision": saved["seq"]})
            return self._install(pid, payload, saved["seq"])

    async def record_tool(self, pid: int, *, name: str, arguments, output: str,
                          succeeded: bool, call_id: str = "") -> None:
        async with self._locks.setdefault(pid, asyncio.Lock()):
            data, revision = await self._load(pid)
            task, person, notes = self._objects(data)
            graph = SketchGraph(data.get("graph"))
            # Only actual returned tool data can create a DONE node.
            effective_success = bool(succeeded) and not bool(_FAILED.search(str(output)[:400]))
            try:
                result_data = json.loads(str(output))
                pending = (isinstance(result_data, dict) and
                           str(result_data.get("status", "")).lower() in
                           ("started", "queued", "running", "accepted", "pending"))
            except (ValueError, TypeError):
                pending = False
            changed = graph.tool_finished(
                call_id, name, arguments, output, effective_success, pending=pending)
            if not changed:
                return
            task.note_result(name, arguments, output)
            if pending:
                task.start("R", name)
            elif not effective_success:
                task.fail(f"{name}: {str(output)[:110]}")
            # Graph nodes record steps, not an assertion that the full goal was met.
            if effective_success and not pending:
                task.finish("R", f"tool {name}", f"returned: {str(output)[:80]}")
            payload = {**self._data(task, person, notes), "graph": graph.to_dict()}
            saved = await self.timeline.append(
                pid, producer="R", kind="picture_state",
                payload=payload, supersedes_seq=revision or None)
            self._install(pid, payload, saved["seq"])
            if self.bus:
                self.bus.publish(pid, {"type": "execution_board_updated", "revision": saved["seq"]})

    async def tool_started(self, pid: int, *, call_id: str, name: str, arguments) -> None:
        if not call_id:
            return
        async with self._locks.setdefault(pid, asyncio.Lock()):
            data, revision = await self._load(pid)
            task, person, notes = self._objects(data)
            graph = SketchGraph(data.get("graph"))
            before = len(graph.nodes)
            graph.tool_started(call_id, name, arguments)
            if len(graph.nodes) == before:
                return
            task.start("R", name)
            payload = {**self._data(task, person, notes), "graph": graph.to_dict()}
            saved = await self.timeline.append(
                pid, producer="R", kind="picture_state", payload=payload,
                supersedes_seq=revision or None)
            self._install(pid, payload, saved["seq"])
            if self.bus:
                self.bus.publish(pid, {"type": "execution_board_updated", "revision": saved["seq"]})

    async def update_from_b(self, pid: int, user_text: str, psyche: dict) -> None:
        """B silently maintains the grounded Person Picture after the reply is out.

        Grounding rule, unchanged from the rule the owner set: a fact enters the
        picture ONLY if the quote B cites appears verbatim in the user's own words
        for this turn. What gets stored is that raw user quote, never B's
        paraphrase. Hypotheses, inferred emotions and unsupported readings are not
        promoted - they stay in the hypothesis/memory layers where they expire.

        Costs no model call: B's psyche output for the turn already exists.
        """
        user_text = str(user_text or "")
        patches: dict[str, list[str]] = {"who": [], "with": [], "weight": [], "doing": [], "tend": []}
        for item in (psyche.get("notes") or []) if isinstance(psyche, dict) else []:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            if not quote or quote not in user_text:
                continue
            kind = str(item.get("kind") or "")
            field = ("tend" if kind == "preference" else
                     "weight" if kind == "feeling" else
                     "doing" if kind in ("plan", "promise") else
                     "who" if kind in ("detail", "event") else None)
            if field:
                # Keep the verbatim source rather than trusting B's paraphrase.
                patches[field].append(quote[:180])
        for ent in (psyche.get("entities") or []) if isinstance(psyche, dict) else []:
            if not isinstance(ent, dict):
                continue
            quote = str(ent.get("quote") or "").strip()
            if not quote or quote not in user_text:
                continue
            name = str(ent.get("name") or "").strip()[:70]
            if not name:
                continue
            field = "with" if ent.get("kind") in ("person", "pet") else "doing"
            patches[field].append(f"{name}: {quote[:120]}")
        if not any(patches.values()):
            return
        async with self._locks.setdefault(pid, asyncio.Lock()):
            data, revision = await self._load(pid)
            task, person, notes = self._objects(data)
            person.merge(patches)
            # _data() carries no graph, so spread the loaded row first to keep it.
            payload = {**data, **self._data(task, person, notes)}
            saved = await self.timeline.append(
                pid, producer="B", kind="picture_state", payload=payload,
                supersedes_seq=revision or None)
            # Re-install the cache: the picture is read from memory next turn, and a
            # write that missed this step left the durable row and the prompt prefix
            # disagreeing (the scope bug this must never reintroduce).
            self._install(pid, payload, saved["seq"])
            if self.bus:
                self.bus.publish(pid, {"type": "execution_board_updated", "revision": saved["seq"]})

    async def render_for_prompt(self, pid: int) -> str:
        return self.for_prompt(pid)
