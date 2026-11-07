"""Deterministic, editable project sketch graph.

Boxes and arrows are data, never AI-generated images. Real tool events update
status automatically; manual boxes and annotations remain explicitly unverified.
"""
from __future__ import annotations

import hashlib
import json
import re

MAX_NODES = 90
MAX_EDGES = 180


def clean(value, limit=120):
    return " ".join(str(value or "").split())[:limit]


def position(value, ceiling):
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError("invalid graph coordinate")
    if not -200 <= value <= ceiling:
        raise ValueError("graph coordinate out of bounds")
    return round(float(value), 1)


class SketchGraph:
    def __init__(self, value=None):
        d = value if isinstance(value, dict) else {}
        self.nodes = [n for n in (d.get("nodes") or []) if isinstance(n, dict)][:MAX_NODES]
        self.edges = [e for e in (d.get("edges") or []) if isinstance(e, dict)][:MAX_EDGES]
        self.receipts = [r for r in (d.get("receipts") or []) if isinstance(r, str)][-200:]

    def to_dict(self):
        return {"nodes": self.nodes, "edges": self.edges, "receipts": self.receipts}

    def _node(self, node_id):
        return next((n for n in self.nodes if n["id"] == node_id), None)

    def ensure_goal(self, goal: str):
        if not goal: return
        node = self._node("goal")
        if node:
            node["label"] = clean(goal)
        else:
            self.nodes.insert(0, {"id": "goal", "label": clean(goal),
                "kind": "goal", "status": "active", "x": 20, "y": 160})

    def _edge(self, source, target, label=""):
        if len(self.edges) >= MAX_EDGES:
            return
        if not any(e.get("from") == source and e.get("to") == target for e in self.edges):
            self.edges.append({"from": source, "to": target, "label": clean(label, 60)})

    def add(self, node_id: str, label: str, *, kind="note", status="unverified"):
        if self._node(node_id):
            raise ValueError("node already exists")
        if len(self.nodes) >= MAX_NODES:
            raise ValueError("sketch graph is full")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", node_id):
            raise ValueError("invalid node ID")
        if not label.strip():
            raise ValueError("node label required")
        i = len(self.nodes)
        self.nodes.append({"id": node_id, "label": clean(label), "kind": kind,
            "status": status, "x": 220 + i % 3 * 175,
            "y": 25 + i // 3 * 90})
        if self._node("goal") and node_id != "goal":
            self._edge("goal", node_id, "contains")

    def edit(self, action: str, data: dict):
        node_id = str(data.get("node_id") or "")
        if action == "add_node":
            self.add(node_id, str(data.get("label") or ""))
            return
        node = self._node(node_id)
        if action == "connect_nodes":
            target = str(data.get("target_id") or "")
            if not node or not self._node(target) or node_id == target:
                raise ValueError("both different nodes must exist")
            self._edge(node_id, target, str(data.get("label") or "depends on"))
            return
        if not node or node_id == "goal" and action == "remove_node":
            raise ValueError("unknown or protected graph node")
        if action == "rename_node":
            label = clean(data.get("label"))
            if not label:
                raise ValueError("node label required")
            node["label"] = label
        elif action == "move_node":
            node["x"] = position(data.get("x"), 1800)
            node["y"] = position(data.get("y"), 3000)
        elif action == "remove_node":
            self.nodes = [n for n in self.nodes if n["id"] != node_id]
            self.edges = [e for e in self.edges if node_id not in (e["from"], e["to"])]
        else:
            raise ValueError("unknown sketch graph action")

    def _room_for_tool(self) -> bool:
        if len(self.nodes) < MAX_NODES:
            return True
        # Never evict manually sketched boxes or an active tool. Retire the
        # oldest completed tool evidence (its receipt and timeline remain).
        old = next((n for n in self.nodes if n.get("kind") == "tool"
                    and n.get("status") in ("done", "failed")), None)
        if old is None:
            return False
        self.nodes = [n for n in self.nodes if n["id"] != old["id"]]
        self.edges = [e for e in self.edges
                      if old["id"] not in (e.get("from"), e.get("to"))]
        return True

    def tool_started(self, call_id: str, name: str, arguments):
        token = clean(call_id, 100)
        if not token:
            return
        node_id = "tool_" + hashlib.sha256(token.encode()).hexdigest()[:16]
        if self._node(node_id):
            return
        args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        if self._room_for_tool():
            self.add(node_id, f"{name} {clean(args, 78)}", kind="tool", status="working")

    def tool_finished(self, call_id: str, name: str, arguments, result: str, success: bool, *, pending: bool = False):
        token = clean(call_id or f"{name}:{arguments}:{result}", 400)
        digest = hashlib.sha256(token.encode()).hexdigest()[:16]
        key = "tool_" + digest
        if key in self.receipts:
            return False
        self.receipts.append(key)
        self.receipts = self.receipts[-200:]
        node = self._node(key)
        if not node:
            args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
            if not self._room_for_tool():
                return False
            self.add(key, f"{name} {clean(args, 78)}", kind="tool", status="working")
            node = self._node(key)
        node["status"] = "working" if pending else "done" if success else "failed"
        node["result"] = clean(result, 220)
        return True
