"""Link — Persona's universal adapter for connecting to any external agent.

Two strategies, chosen automatically from the agent's own capabilities:

1. **Hijack** — the agent already supports delegation or sub-agent creation
   (e.g. Hermes delegate_task, Cursor Task/spawn_subagent, Claude Code Agent).
   Persona discovers that pathway at registration time and uses it to inject
   R-lane work into the agent's own concurrency model. The agent thinks it
   spawned a sub-agent; in reality Persona is steering the work.

2. **Spawn** — the agent has no native delegation. Persona launches a new
   process of ``sawii_hands.py`` pointed at the lane's WebSocket endpoint,
   giving it the agent's engine binary/URL and allowed folders. Each R-lane
   gets its own process. This is what the README calls "creating multiple
   instances of the engine."

The Link does not modify the agent's own code. It is an adapter between
Persona's LaneBook and whatever surface the agent exposes.

Registration:
    link = Link.detect(agent_config)
    # agent_config is a dict with at least:
    #   name, url (or binary), tools (list of OpenAI function schemas)
    # Optional: key, roots, approve, place

Usage:
    await link.delegate(lane, instruction)   # start work on this lane
    await link.steer(lane, note)             # mid-work guidance
    await link.stop(lane)                    # cancel
    await link.status(lane)                  # is it still running?
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .host_delegation import is_host_background_tool

log = logging.getLogger(__name__)

# How long to wait for a spawned process to connect back via WebSocket.
SPAWN_CONNECT_TIMEOUT = 60.0
# How long to wait for a hijacked delegation call to be acknowledged.
HIJACK_ACK_TIMEOUT = 30.0


@dataclass
class AgentProfile:
    """What we know about an external agent after probing it."""
    name: str
    # The pathway we'll use to connect.
    strategy: str                  # "hijack" or "spawn"
    # For hijack: which tool and how to call it.
    delegation_tool: str = ""      # e.g. "delegate_task", "Task", "spawn_subagent"
    delegation_schema: dict = field(default_factory=dict)
    supports_steer: bool = False
    supports_stop: bool = False
    supports_list: bool = False
    # For spawn: what binary/script to run.
    binary: str = ""               # path to the engine executable
    url: str = ""                  # Persona URL the spawned process connects to
    key: str = ""                  # Persona key for auth
    roots: list[str] = field(default_factory=list)
    approve: str = "auto"          # spawned workers are headless
    place: str = "cloud"
    extra_args: list[str] = field(default_factory=list)
    # All tools the agent advertised (for reference).
    tools: list[dict] = field(default_factory=list)


@dataclass
class LinkedLane:
    """A running connection between a Persona R-lane and an external agent."""
    lane_id: str
    agent: AgentProfile
    # For hijack: the tool_call_id and any agent-side sub-agent id.
    host_call_id: str = ""
    host_agent_id: str = ""
    # For spawn: the subprocess handle.
    process: Any = None
    # Status.
    status: str = "starting"       # starting | active | done | failed | stopped


class Link:
    """One Link instance per registered external agent."""

    def __init__(self, agent: AgentProfile):
        self.agent = agent
        self._lanes: dict[str, LinkedLane] = {}

    # ------------------------------------------------------------------ detect

    @classmethod
    def detect(cls, config: dict) -> "Link":
        """Probe an agent config and return a Link with the right strategy.

        config keys:
            name (str): human label
            tools (list[dict]): OpenAI function schemas the agent advertises
            url (str, optional): for hijack — the agent's API endpoint
            binary (str, optional): for spawn — path to engine script
            persona_url (str): Persona's own URL (for spawn callback)
            persona_key (str): Persona key (for spawn auth)
            roots (list[str], optional): folders the spawned engine may access
            approve (str, optional): "auto"|"writes"|"ask"
            place (str, optional): "computer"|"cloud"
        """
        name = str(config.get("name") or "unknown")
        tools = config.get("tools") or []

        # 1. Check for native delegation support.
        delegation_tool = _find_delegation_tool(tools)
        if delegation_tool:
            tool_name, schema, caps = delegation_tool
            log.info("Link: %s supports native delegation via %s (steer=%s, stop=%s)",
                     name, tool_name, caps["steer"], caps["stop"])
            profile = AgentProfile(
                name=name,
                strategy="hijack",
                delegation_tool=tool_name,
                delegation_schema=schema,
                supports_steer=caps["steer"],
                supports_stop=caps["stop"],
                supports_list=caps["list"],
                url=str(config.get("url") or ""),
                tools=tools,
            )
            return cls(profile)

        # 2. Fall back to spawn if we have a binary.
        binary = str(config.get("binary") or "")
        if not binary:
            # Default: use sawii_hands.py from this repo.
            binary = str(Path(__file__).parent.parent / "sawii_hands.py")

        log.info("Link: %s has no delegation — will spawn engine instances", name)
        profile = AgentProfile(
            name=name,
            strategy="spawn",
            binary=binary,
            url=str(config.get("persona_url") or ""),
            key=str(config.get("persona_key") or ""),
            roots=list(config.get("roots") or []),
            approve=str(config.get("approve") or "auto"),
            place=str(config.get("place") or "cloud"),
            extra_args=list(config.get("extra_args") or []),
            tools=tools,
        )
        return cls(profile)

    # ------------------------------------------------------------------ delegate

    async def delegate(self, lane_id: str, instruction: str, **kwargs) -> LinkedLane:
        """Start work on an R-lane through this agent."""
        if lane_id in self._lanes and self._lanes[lane_id].status == "active":
            raise RuntimeError(f"lane {lane_id} is already active on {self.agent.name}")

        linked = LinkedLane(lane_id=lane_id, agent=self.agent)
        self._lanes[lane_id] = linked

        if self.agent.strategy == "hijack":
            await self._hijack_delegate(linked, instruction, **kwargs)
        else:
            await self._spawn_delegate(linked, instruction, **kwargs)

        return linked

    async def steer(self, lane_id: str, note: str) -> bool:
        """Send mid-work guidance to a running lane."""
        linked = self._lanes.get(lane_id)
        if not linked or linked.status != "active":
            return False

        if self.agent.strategy == "hijack" and self.agent.supports_steer:
            return await self._hijack_steer(linked, note)

        # Spawn strategy: guidance goes through Persona's own lane notes,
        # which the spawned worker reads via the WebSocket.
        return True  # The caller writes to the LaneBook; the worker reads it.

    async def stop(self, lane_id: str) -> bool:
        """Stop work on a lane."""
        linked = self._lanes.get(lane_id)
        if not linked or linked.status not in ("active", "starting"):
            return False

        if self.agent.strategy == "hijack" and self.agent.supports_stop:
            await self._hijack_stop(linked)
        elif self.agent.strategy == "spawn" and linked.process:
            try:
                linked.process.terminate()
                await asyncio.sleep(0.5)
                if linked.process.returncode is None:
                    linked.process.kill()
            except (OSError, ProcessLookupError):
                pass

        linked.status = "stopped"
        return True

    async def status(self, lane_id: str) -> str:
        """Check the status of a lane."""
        linked = self._lanes.get(lane_id)
        if not linked:
            return "unknown"

        if self.agent.strategy == "spawn" and linked.process:
            if linked.process.returncode is not None:
                linked.status = "done" if linked.process.returncode == 0 else "failed"

        return linked.status

    def active_lanes(self) -> list[str]:
        return [lid for lid, ln in self._lanes.items() if ln.status == "active"]

    # ------------------------------------------------------------------ hijack internals

    async def _hijack_delegate(self, linked: LinkedLane, instruction: str, **kwargs):
        """Use the agent's own delegation tool to start work."""
        tool_name = self.agent.delegation_tool
        schema = self.agent.delegation_schema
        props = (schema.get("parameters") or {}).get("properties") or {}

        # Build the arguments based on which delegation tool we detected.
        if tool_name == "delegate_task":
            # Hermes-style: tasks array with goals.
            args = {
                "action": "spawn",
                "tasks": [{"goal": instruction}],
            }
        elif tool_name in ("Task", "task", "spawn_subagent", "launch_subagent"):
            # Cursor/Claude Code style: prompt + optional background flag.
            args = {"prompt": instruction}
            if "run_in_background" in props:
                args["run_in_background"] = True
            elif "is_background" in props:
                args["is_background"] = True
            if "description" in props:
                args["description"] = instruction[:100]
        else:
            args = {"instruction": instruction}

        # The actual call is issued by the inference loop when it sees the
        # tool_call in the model's output. We prepare the call structure here
        # and the lane book records it as delegated.
        call_id = f"call_{linked.lane_id}_{secrets.token_hex(3)}"
        linked.host_call_id = call_id
        linked.status = "active"

        # Store the prepared delegation call for the inference loop to pick up.
        linked._prepared_call = {
            "id": call_id,
            "name": tool_name,
            "arguments": json.dumps(args, ensure_ascii=False),
        }

    async def _hijack_steer(self, linked: LinkedLane, note: str) -> bool:
        """Steer a running delegated task."""
        if not linked.host_agent_id:
            return False

        tool_name = self.agent.delegation_tool
        if tool_name == "delegate_task":
            linked._prepared_call = {
                "id": f"call_{linked.lane_id}_{secrets.token_hex(3)}",
                "name": tool_name,
                "arguments": json.dumps({
                    "action": "steer",
                    "subagent_id": linked.host_agent_id,
                    "message": note,
                }, ensure_ascii=False),
            }
            return True
        return False

    async def _hijack_stop(self, linked: LinkedLane):
        """Stop a delegated task via the agent's own stop mechanism."""
        tool_name = self.agent.delegation_tool
        if tool_name == "delegate_task" and linked.host_agent_id:
            linked._prepared_call = {
                "id": f"call_{linked.lane_id}_{secrets.token_hex(3)}",
                "name": tool_name,
                "arguments": json.dumps({
                    "action": "stop",
                    "subagent_id": linked.host_agent_id,
                }, ensure_ascii=False),
            }

    # ------------------------------------------------------------------ spawn internals

    async def _spawn_delegate(self, linked: LinkedLane, instruction: str, **kwargs):
        """Launch a new sawii_hands.py process for this lane."""
        binary = self.agent.binary
        if not binary or not os.path.isfile(binary):
            linked.status = "failed"
            log.error("Link: spawn binary not found: %s", binary)
            return

        cmd = [
            sys.executable, binary,
            "--url", self.agent.url,
            "--key", self.agent.key,
            "--approve", self.agent.approve,
            "--place", self.agent.place,
            "--lane", linked.lane_id,
        ]
        for root in self.agent.roots:
            cmd += ["--root", root]
        cmd += self.agent.extra_args

        env = {**os.environ}
        # Pass the lane instruction as an env var so the worker can read it.
        env["SAWII_LANE_INSTRUCTION"] = instruction[:4000]

        try:
            linked.process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            linked.status = "active"
            log.info("Link: spawned %s (pid %d) for lane %s",
                     binary, linked.process.pid, linked.lane_id)
        except Exception as exc:
            linked.status = "failed"
            log.exception("Link: failed to spawn %s: %s", binary, exc)

    # ------------------------------------------------------------------ cleanup

    async def shutdown(self):
        """Stop all active lanes."""
        for lane_id in list(self._lanes):
            await self.stop(lane_id)


# -------------------------------------------------------------------- detection

def _find_delegation_tool(tools: list[dict]) -> tuple[str, dict, dict] | None:
    """Probe a tool list for any known delegation/sub-agent pattern.

    Returns (tool_name, full_schema, capabilities_dict) or None.
    """
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or {}
        props = schema.get("properties") or {}

        # Hermes: delegate_task with tasks array.
        if name == "delegate_task":
            tasks = props.get("tasks")
            if isinstance(tasks, dict) and tasks.get("type") == "array":
                actions = (props.get("action") or {}).get("enum") or []
                return (name, fn, {
                    "steer": "steer" in actions,
                    "stop": "stop" in actions,
                    "list": "list" in actions,
                })

        # Cursor/Claude Code: Task, task, spawn_subagent, launch_subagent.
        if name in ("Task", "task", "spawn_subagent", "launch_subagent"):
            if is_host_background_tool(tool):
                return (name, fn, {
                    "steer": False,  # These typically don't support mid-task steering.
                    "stop": "cancel" in props or "stop" in props,
                    "list": False,
                })

    return None


class LinkRegistry:
    """All registered agent Links for a persona, keyed by agent name."""

    def __init__(self):
        self._links: dict[str, Link] = {}

    def register(self, config: dict) -> Link:
        """Detect and register an agent. Returns the Link."""
        link = Link.detect(config)
        self._links[link.agent.name] = link
        return link

    def get(self, name: str) -> Link | None:
        return self._links.get(name)

    def all(self) -> dict[str, Link]:
        return dict(self._links)

    def compatible(self, name: str) -> bool:
        """Can we connect to this agent at all?"""
        link = self._links.get(name)
        if link is None:
            return False
        return True  # If detect() succeeded, we have a strategy.

    def strategies(self) -> dict[str, str]:
        """Agent name -> strategy for all registered agents."""
        return {name: link.agent.strategy for name, link in self._links.items()}

    async def shutdown_all(self):
        for link in self._links.values():
            await link.shutdown()
