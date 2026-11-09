"""Link — Persona's universal adapter for connecting to any external agent.

    "we need to create an element we call it the link basically it's an adapter"
    "I want everything to be happening inside of that adapt so it works with any agent"

The Link is the single unit of connection between Persona and any agent, external
or internal. All integration logic lives here. What Persona sees is always the same:
pure hands WebSocket protocol frames (hello/ready/call/result). What the agent sees
is whatever native protocol it speaks.

Two strategies, chosen automatically:

1. **Hijack** — the agent supports delegation natively. Persona is running INSIDE
   the agent (Cursor, Claude Code, Hermes, Cline). Link discovers the delegation
   tool from the host's advertised schemas and uses it to inject R-lane work.
   The agent thinks it spawned a sub-agent; Persona is steering the work through
   its own lane system.

2. **Spawn** — the agent has no native delegation, or Persona is the orchestrator.
   Link launches engine processes (``sawii_hands.py`` or an external agent binary)
   per R-lane. Each process connects back to Persona's ``/v1/hands`` WebSocket.
   HandsHub manages the connection from there.

For external agents reachable via HTTP API (OpenAI-compatible, browser-use, or
any agent with a chat completions endpoint), Link runs an **API bridge**: a
coroutine that connects to Persona's WebSocket on one side and makes API calls
to the agent on the other, translating tool calls into API requests and
streaming results back as hands protocol frames.

Supported agents (hijack):
    - Hermes: delegate_task with tasks array, spawn/steer/stop/list actions
    - Cursor: Task or task tool with prompt, run_in_background
    - Claude Code: Agent or spawn_subagent with prompt, run_in_background
    - Cline: Task or launch_subagent with prompt
    - Any host with a recognized delegation tool schema

Supported agents (spawn / API bridge):
    - sawii_hands.py (the built-in engine): spawned as a process
    - OpenAI-compatible agents: bridged via chat completions API
    - browser-use: bridged via its OpenAI-compatible endpoint
    - Any agent with an HTTP API that accepts tool calls

Registration:
    link = Link.detect(agent_config)

Usage:
    await link.delegate(lane_id, instruction)
    await link.steer(lane_id, note)
    await link.stop(lane_id)
    status = await link.status(lane_id)
    calls = link.pending_calls(lane_id)   # hijack: calls for the inference loop
    link.receive_result(lane_id, result)   # hijack: results from the host
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .host_delegation import is_host_background_tool

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------
SPAWN_CONNECT_TIMEOUT = 60.0      # seconds to wait for spawned process WebSocket
HIJACK_ACK_TIMEOUT = 30.0         # seconds to wait for hijack delegation ack
API_CALL_TIMEOUT = 120.0          # seconds per API call to external agent
API_BRIDGE_POLL = 1.0             # seconds between polling an async API task
PROCESS_KILL_GRACE = 2.0          # seconds between SIGTERM and SIGKILL


# ═══════════════════════════════════════════════════════════════════════════
# Agent profiles and linked lanes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentProfile:
    """Everything known about an external agent after detection."""
    name: str
    strategy: str                          # "hijack" | "spawn" | "api_bridge"
    agent_type: str = "generic"            # hermes | cursor | claude_code | cline | browser_use | openai | engine | generic

    # Hijack fields
    delegation_tool: str = ""              # e.g. "delegate_task", "Task", "spawn_subagent"
    delegation_schema: dict = field(default_factory=dict)
    supports_steer: bool = False
    supports_stop: bool = False
    supports_list: bool = False

    # Spawn / API bridge fields
    binary: str = ""                       # path to the engine or agent executable
    url: str = ""                          # Persona URL (spawn) or agent API URL (bridge)
    key: str = ""                          # Persona key (spawn) or agent API key (bridge)
    model: str = ""                        # for API bridge: which model to call
    roots: list[str] = field(default_factory=list)
    approve: str = "auto"                  # spawned workers are headless
    place: str = "cloud"
    extra_args: list[str] = field(default_factory=list)
    extra_headers: dict[str, str] = field(default_factory=dict)

    # All tools the agent advertised (for reference).
    tools: list[dict] = field(default_factory=list)


@dataclass
class LinkedLane:
    """A running connection between a Persona R-lane and an external agent."""
    lane_id: str
    agent: AgentProfile
    instruction: str = ""

    # Hijack state
    host_call_id: str = ""
    host_agent_id: str = ""
    _prepared_calls: list[dict] = field(default_factory=list)
    _pending_results: dict[str, asyncio.Future] = field(default_factory=dict)

    # Spawn state
    process: Any = None
    _bridge_task: asyncio.Task | None = None

    # Identity this motor runs as (set by Link.delegate kwargs). The spawned
    # engine authenticates with ``persona_key`` so it registers on the hub as
    # *this* persona's runner; ``persona_id`` is the hub addressing key.
    persona_id: int = 0
    persona_key: str = ""

    # True once HandsHub has shown a runner for this lane. Set by
    # ``Link.observe_hub`` (which reads the hub, the only real source of truth)
    # and read by ``Link._await_connection``, so "spawned" can only mean connected.
    connected: bool = False

    # Optional callback ``fn(exit_code, error)`` fired when the motor dies. The
    # court deck uses it so a card that loses its engine reports the truth
    # instead of sitting at "running" forever.
    on_exit: Any = None

    # Lifecycle
    status: str = "starting"               # starting | active | done | failed | stopped
    started_at: float = 0.0
    error: str = ""

    @property
    def alive(self) -> bool:
        """Is there actually a motor behind this lane right now?"""
        if self.status != "active":
            return False
        if self.agent.strategy == "spawn":
            return self.process is not None and self.process.returncode is None
        if self.agent.strategy == "native":
            return self._bridge_task is not None and not self._bridge_task.done()
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Agent drivers — one per agent type
# ═══════════════════════════════════════════════════════════════════════════

class AgentDriver(ABC):
    """Base class for agent-specific delegation logic.

    A driver knows how to:
    - Detect its agent type from a tool list or config
    - Construct delegation calls in the agent's native format
    - Parse results coming back
    - Map lifecycle operations (steer/stop/list) to agent-specific calls
    """

    name: str = "generic"

    @staticmethod
    @abstractmethod
    def detect(tools: list[dict], config: dict) -> AgentProfile | None:
        """Probe tools/config and return an AgentProfile if this driver matches."""
        ...

    @abstractmethod
    def build_delegate_call(self, lane_id: str, instruction: str,
                            profile: AgentProfile) -> dict:
        """Construct the native delegation tool call for starting work."""
        ...

    def build_steer_call(self, lane_id: str, agent_id: str, note: str,
                         profile: AgentProfile) -> dict | None:
        """Construct a steering call, or None if not supported."""
        return None

    def build_stop_call(self, lane_id: str, agent_id: str,
                        profile: AgentProfile) -> dict | None:
        """Construct a stop call, or None if not supported."""
        return None

    def build_list_call(self, profile: AgentProfile) -> dict | None:
        """Construct a list-agents call, or None if not supported."""
        return None

    def parse_delegate_result(self, result: str) -> dict:
        """Parse the agent's response to a delegation call.

        Returns: {"agent_id": "...", "status": "...", "output": "..."}
        """
        return {"agent_id": "", "status": "active", "output": result}

    async def speculative_delegate(
        self,
        lane_id: str,
        instruction: str,
        profile: AgentProfile,
        execute_fn,
        engine: "SpeculativeEngine | None" = None,
        confidence: float = 0.7,
        precompute_fn=None,
        step_context: dict[str, Any] | None = None,
        is_last_step: bool = False,
    ) -> dict[str, Any]:
        """Delegate with speculative pre-computation.

        Wraps build_delegate_call + execution through SpeculativeEngine,
        so ANY agent driver (Hermes, Cursor, Claude Code, Cline) gets
        zero-gap dispatch on delegation results.

        execute_fn: async callable that sends the built tool call and
                    returns the raw result string.
        engine:     SpeculativeEngine instance. If None, falls back to
                    plain sequential delegation (no speculation).

        Returns SpeculativeEngine step result or plain delegate result.
        """
        call = self.build_delegate_call(lane_id, instruction, profile)

        if engine is None:
            # No speculation — plain sequential
            raw = await execute_fn(call)
            return self.parse_delegate_result(raw)

        # Pre-compute what happens when delegation succeeds/fails
        default_precompute = precompute_fn or (lambda ctx: {
            "active": {"status": "active", "action": "monitor"},
            "completed": {"status": "completed", "action": "collect_result"},
            "failed": {"status": "failed", "action": "retry_or_escalate"},
        })

        result = await engine.step(
            confidence=confidence,
            action_fn=lambda: execute_fn(call),
            precompute_fn=default_precompute,
            step_context=step_context or {"lane_id": lane_id, "instruction": instruction},
            is_last_step=is_last_step,
        )

        # Parse the raw result through the driver
        parsed = self.parse_delegate_result(result["result"])
        result["parsed"] = parsed
        return result


# ─── Hermes ───────────────────────────────────────────────────────────────

class HermesDriver(AgentDriver):
    """Hermes: delegate_task with tasks array, spawn/steer/stop/list actions.

    Hermes is the most fully featured delegation protocol. It supports:
    - spawn: create new sub-agents with goal descriptions
    - steer: send mid-work guidance to a running agent
    - stop: cancel a running agent
    - list: enumerate running agents and their status
    """
    name = "hermes"

    @staticmethod
    def detect(tools: list[dict], config: dict) -> AgentProfile | None:
        for tool in tools:
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn, dict):
                continue
            if fn.get("name") != "delegate_task":
                continue
            schema = fn.get("parameters") or {}
            props = schema.get("properties") or {}
            tasks = props.get("tasks")
            if not (isinstance(tasks, dict) and tasks.get("type") == "array"):
                continue
            actions = (props.get("action") or {}).get("enum") or []
            return AgentProfile(
                name=config.get("name", "hermes"),
                strategy="hijack",
                agent_type="hermes",
                delegation_tool="delegate_task",
                delegation_schema=fn,
                supports_steer="steer" in actions,
                supports_stop="stop" in actions,
                supports_list="list" in actions,
                url=str(config.get("url") or ""),
                tools=tools,
            )
        return None

    def build_delegate_call(self, lane_id: str, instruction: str,
                            profile: AgentProfile) -> dict:
        call_id = f"call_{lane_id}_{secrets.token_hex(3)}"
        return {
            "id": call_id,
            "name": "delegate_task",
            "arguments": json.dumps({
                "action": "spawn",
                "tasks": [{"goal": f"[persona-lane {lane_id}] {instruction}"}],
            }, ensure_ascii=False),
        }

    def build_steer_call(self, lane_id: str, agent_id: str, note: str,
                         profile: AgentProfile) -> dict | None:
        if not profile.supports_steer or not agent_id:
            return None
        return {
            "id": f"call_{lane_id}_{secrets.token_hex(3)}",
            "name": "delegate_task",
            "arguments": json.dumps({
                "action": "steer",
                "subagent_id": agent_id,
                "message": note,
            }, ensure_ascii=False),
        }

    def build_stop_call(self, lane_id: str, agent_id: str,
                        profile: AgentProfile) -> dict | None:
        if not profile.supports_stop or not agent_id:
            return None
        return {
            "id": f"call_{lane_id}_{secrets.token_hex(3)}",
            "name": "delegate_task",
            "arguments": json.dumps({
                "action": "stop",
                "subagent_id": agent_id,
            }, ensure_ascii=False),
        }

    def build_list_call(self, profile: AgentProfile) -> dict | None:
        if not profile.supports_list:
            return None
        return {
            "id": f"call_list_{secrets.token_hex(3)}",
            "name": "delegate_task",
            "arguments": json.dumps({"action": "list"}, ensure_ascii=False),
        }

    def parse_delegate_result(self, result: str) -> dict:
        try:
            data = json.loads(result)
        except (ValueError, TypeError):
            return {"agent_id": "", "status": "active", "output": result}
        if isinstance(data, dict):
            return {
                "agent_id": str(data.get("subagent_id") or data.get("id") or ""),
                "status": str(data.get("status") or "active"),
                "output": str(data.get("output") or data.get("result") or result),
            }
        return {"agent_id": "", "status": "active", "output": result}


# ─── Cursor ───────────────────────────────────────────────────────────────

class CursorDriver(AgentDriver):
    """Cursor: Task tool with prompt, run_in_background flag.

    Cursor's Task tool accepts a prompt and optionally a subagent_type and
    description. The run_in_background or is_background flag makes it async.
    Cursor does not natively support mid-task steering or stopping via the
    tool API, but the task can be stopped by the host.
    """
    name = "cursor"

    @staticmethod
    def detect(tools: list[dict], config: dict) -> AgentProfile | None:
        for tool in tools:
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if name not in ("Task", "task"):
                continue
            if not is_host_background_tool(tool):
                continue
            schema = fn.get("parameters") or {}
            props = schema.get("properties") or {}
            return AgentProfile(
                name=config.get("name", "cursor"),
                strategy="hijack",
                agent_type="cursor",
                delegation_tool=name,
                delegation_schema=fn,
                supports_steer=False,
                supports_stop="cancel" in props or "stop" in props,
                supports_list=False,
                url=str(config.get("url") or ""),
                tools=tools,
            )
        return None

    def build_delegate_call(self, lane_id: str, instruction: str,
                            profile: AgentProfile) -> dict:
        call_id = f"call_{lane_id}_{secrets.token_hex(3)}"
        schema = profile.delegation_schema
        props = (schema.get("parameters") or {}).get("properties") or {}

        args: dict[str, Any] = {}
        # Prompt field
        if "prompt" in props:
            args["prompt"] = f"[persona-lane {lane_id}] {instruction}"
        elif "task" in props:
            args["task"] = f"[persona-lane {lane_id}] {instruction}"

        # Background flag
        if "run_in_background" in props:
            args["run_in_background"] = True
        elif "is_background" in props:
            args["is_background"] = True

        # Description
        if "description" in props:
            args["description"] = instruction[:100]

        # Subagent type — prefer persona-r if available
        for agent_field in ("subagent_type", "agent"):
            if agent_field not in props:
                continue
            spec = props[agent_field] if isinstance(props[agent_field], dict) else {}
            accepted = spec.get("enum") or []
            preferred = ("persona-r", "generalPurpose", "general-purpose",
                         "general", "default", "explore")
            if accepted:
                selected = next((x for x in preferred if x in accepted), None)
                if selected:
                    args[agent_field] = selected
            else:
                required = (schema.get("parameters") or {}).get("required") or []
                if agent_field in required:
                    args[agent_field] = "persona-r"

        return {
            "id": call_id,
            "name": profile.delegation_tool,
            "arguments": json.dumps(args, ensure_ascii=False),
        }

    def build_stop_call(self, lane_id: str, agent_id: str,
                        profile: AgentProfile) -> dict | None:
        schema = profile.delegation_schema
        props = (schema.get("parameters") or {}).get("properties") or {}
        if "cancel" in props:
            return {
                "id": f"call_{lane_id}_{secrets.token_hex(3)}",
                "name": profile.delegation_tool,
                "arguments": json.dumps({"cancel": agent_id or lane_id}, ensure_ascii=False),
            }
        return None

    def parse_delegate_result(self, result: str) -> dict:
        try:
            data = json.loads(result)
        except (ValueError, TypeError):
            return {"agent_id": "", "status": "active", "output": result}
        if isinstance(data, dict):
            return {
                "agent_id": str(data.get("agent_id") or data.get("id") or ""),
                "status": str(data.get("status") or "active"),
                "output": str(data.get("output") or data.get("result") or result),
            }
        return {"agent_id": "", "status": "active", "output": result}


# ─── Claude Code ──────────────────────────────────────────────────────────

class ClaudeCodeDriver(AgentDriver):
    """Claude Code: Agent or spawn_subagent with prompt, run_in_background.

    Claude Code's Agent tool spawns a subagent with a prompt. The subagent
    runs in the background and reports back when done. Claude Code supports
    multiple subagent types via subagent_type.
    """
    name = "claude_code"

    @staticmethod
    def detect(tools: list[dict], config: dict) -> AgentProfile | None:
        for tool in tools:
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if name not in ("spawn_subagent", "launch_subagent", "Agent"):
                continue
            if not is_host_background_tool(tool):
                continue
            schema = fn.get("parameters") or {}
            props = schema.get("properties") or {}
            return AgentProfile(
                name=config.get("name", "claude_code"),
                strategy="hijack",
                agent_type="claude_code",
                delegation_tool=name,
                delegation_schema=fn,
                supports_steer=False,
                supports_stop="cancel" in props,
                supports_list=False,
                url=str(config.get("url") or ""),
                tools=tools,
            )
        return None

    def build_delegate_call(self, lane_id: str, instruction: str,
                            profile: AgentProfile) -> dict:
        call_id = f"call_{lane_id}_{secrets.token_hex(3)}"
        schema = profile.delegation_schema
        props = (schema.get("parameters") or {}).get("properties") or {}

        args: dict[str, Any] = {
            "prompt": f"[persona-lane {lane_id}] {instruction}",
        }

        if "run_in_background" in props:
            args["run_in_background"] = True
        elif "is_background" in props:
            args["is_background"] = True

        if "description" in props:
            args["description"] = instruction[:100]

        if "name" in props:
            args["name"] = f"persona-r-{lane_id[:8]}"

        # Subagent type
        for agent_field in ("subagent_type", "agent_type"):
            if agent_field in props:
                spec = props[agent_field] if isinstance(props[agent_field], dict) else {}
                accepted = spec.get("enum") or []
                if accepted and "fork" in accepted:
                    args[agent_field] = "fork"
                elif accepted:
                    args[agent_field] = accepted[0]
                break

        return {
            "id": call_id,
            "name": profile.delegation_tool,
            "arguments": json.dumps(args, ensure_ascii=False),
        }

    def parse_delegate_result(self, result: str) -> dict:
        try:
            data = json.loads(result)
        except (ValueError, TypeError):
            return {"agent_id": "", "status": "active", "output": result}
        if isinstance(data, dict):
            return {
                "agent_id": str(data.get("agent_id") or data.get("name") or ""),
                "status": str(data.get("status") or "active"),
                "output": str(data.get("output") or data.get("result") or result),
            }
        return {"agent_id": "", "status": "active", "output": result}


# ─── Cline ────────────────────────────────────────────────────────────────

class ClineDriver(AgentDriver):
    """Cline: Task or launch_subagent with prompt.

    Cline's delegation surface is similar to Cursor's Task tool. It may
    advertise launch_subagent instead of Task. The detection is the same:
    look for a background-capable tool with prompt/task parameters.
    """
    name = "cline"

    @staticmethod
    def detect(tools: list[dict], config: dict) -> AgentProfile | None:
        for tool in tools:
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if name not in ("launch_subagent",):
                continue
            if not is_host_background_tool(tool):
                continue
            schema = fn.get("parameters") or {}
            props = schema.get("properties") or {}
            return AgentProfile(
                name=config.get("name", "cline"),
                strategy="hijack",
                agent_type="cline",
                delegation_tool=name,
                delegation_schema=fn,
                supports_steer=False,
                supports_stop="cancel" in props,
                supports_list=False,
                url=str(config.get("url") or ""),
                tools=tools,
            )
        return None

    def build_delegate_call(self, lane_id: str, instruction: str,
                            profile: AgentProfile) -> dict:
        call_id = f"call_{lane_id}_{secrets.token_hex(3)}"
        schema = profile.delegation_schema
        props = (schema.get("parameters") or {}).get("properties") or {}

        args: dict[str, Any] = {}
        if "prompt" in props:
            args["prompt"] = f"[persona-lane {lane_id}] {instruction}"
        elif "task" in props:
            args["task"] = f"[persona-lane {lane_id}] {instruction}"
        else:
            args["instruction"] = f"[persona-lane {lane_id}] {instruction}"

        if "run_in_background" in props:
            args["run_in_background"] = True
        elif "is_background" in props:
            args["is_background"] = True

        if "description" in props:
            args["description"] = instruction[:100]

        return {
            "id": call_id,
            "name": profile.delegation_tool,
            "arguments": json.dumps(args, ensure_ascii=False),
        }

    def parse_delegate_result(self, result: str) -> dict:
        try:
            data = json.loads(result)
        except (ValueError, TypeError):
            return {"agent_id": "", "status": "active", "output": result}
        if isinstance(data, dict):
            return {
                "agent_id": str(data.get("agent_id") or data.get("id") or ""),
                "status": str(data.get("status") or "active"),
                "output": str(data.get("output") or data.get("result") or result),
            }
        return {"agent_id": "", "status": "active", "output": result}


# ─── Generic hijack (catch-all for any recognized delegation tool) ────────

class GenericHijackDriver(AgentDriver):
    """Fallback for any host tool recognized by is_host_background_tool().

    If no specific driver matches but the tool schema matches the delegation
    pattern, this driver handles it with best-effort argument construction.
    """
    name = "generic_hijack"

    @staticmethod
    def detect(tools: list[dict], config: dict) -> AgentProfile | None:
        for tool in tools:
            if not is_host_background_tool(tool):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            schema = fn.get("parameters") or {}
            props = schema.get("properties") or {}
            return AgentProfile(
                name=config.get("name", f"host-{name}"),
                strategy="hijack",
                agent_type="generic",
                delegation_tool=name,
                delegation_schema=fn,
                supports_steer=False,
                supports_stop="cancel" in props or "stop" in props,
                supports_list=False,
                url=str(config.get("url") or ""),
                tools=tools,
            )
        return None

    def build_delegate_call(self, lane_id: str, instruction: str,
                            profile: AgentProfile) -> dict:
        call_id = f"call_{lane_id}_{secrets.token_hex(3)}"
        schema = profile.delegation_schema
        props = (schema.get("parameters") or {}).get("properties") or {}
        args: dict[str, Any] = {}

        # Try common prompt field names
        for prompt_key in ("prompt", "task", "goal", "instruction", "message", "input"):
            if prompt_key in props:
                args[prompt_key] = f"[persona-lane {lane_id}] {instruction}"
                break
        else:
            # Last resort: use the first string property
            for k, v in props.items():
                if isinstance(v, dict) and v.get("type") == "string":
                    args[k] = f"[persona-lane {lane_id}] {instruction}"
                    break

        # Background flags
        for flag in ("run_in_background", "is_background"):
            if flag in props:
                args[flag] = True

        if "description" in props:
            args["description"] = instruction[:100]

        return {
            "id": call_id,
            "name": profile.delegation_tool,
            "arguments": json.dumps(args, ensure_ascii=False),
        }

    def parse_delegate_result(self, result: str) -> dict:
        try:
            data = json.loads(result)
        except (ValueError, TypeError):
            return {"agent_id": "", "status": "active", "output": result}
        if isinstance(data, dict):
            return {
                "agent_id": str(data.get("agent_id") or data.get("subagent_id")
                               or data.get("id") or ""),
                "status": str(data.get("status") or "active"),
                "output": str(data.get("output") or data.get("result") or result),
            }
        return {"agent_id": "", "status": "active", "output": result}


# ═══════════════════════════════════════════════════════════════════════════
# API bridge drivers — for external agents reachable via HTTP
# ═══════════════════════════════════════════════════════════════════════════

class OpenAIBridgeDriver:
    """Bridge to any OpenAI-compatible chat completions API.

    This covers:
    - Generic OpenAI-compatible agents
    - browser-use (runs as an OpenAI-compatible server)
    - DeepInfra, Together, Groq, or any hosted model
    - Local models via Ollama, vLLM, etc.

    The bridge runs a coroutine per lane that:
    1. Sends the lane instruction as a user message to the agent's API
    2. Streams tool calls from the response
    3. Executes tools via Persona's HandsHub
    4. Feeds results back to the agent for the next step
    5. Repeats until the agent says it's done (no more tool calls)
    """
    name = "openai_bridge"

    @staticmethod
    def detect_from_config(config: dict) -> AgentProfile | None:
        """Detect from explicit config (not from tool schemas)."""
        api_url = config.get("api_url") or config.get("base_url") or ""
        if not api_url:
            return None
        agent_type = config.get("agent_type", "openai")
        return AgentProfile(
            name=config.get("name", agent_type),
            strategy="api_bridge",
            agent_type=agent_type,
            url=str(api_url),
            key=str(config.get("api_key") or config.get("key") or ""),
            model=str(config.get("model") or ""),
            extra_headers=dict(config.get("extra_headers") or {}),
            tools=list(config.get("tools") or []),
        )

    @staticmethod
    async def run_bridge(linked: LinkedLane, hub, persona_id: int):
        """Run the API bridge loop for one lane.

        This is the core bridge: it speaks OpenAI chat completions on one side
        and hands WebSocket protocol on the other. It runs as a long-lived
        coroutine for the lifetime of the lane.
        """
        try:
            import httpx
        except ImportError:
            log.error("httpx is required for API bridge (pip install httpx)")
            linked.status = "failed"
            linked.error = "httpx not installed"
            return

        profile = linked.agent
        base = profile.url.rstrip("/")
        headers = {"Content-Type": "application/json", **profile.extra_headers}
        if profile.key:
            headers["Authorization"] = f"Bearer {profile.key}"

        # Collect tools from Persona's HandsHub for this persona
        available_tools = hub.schemas(persona_id) if hub else []

        # Build the initial messages
        messages = [
            {"role": "system", "content": (
                "You are an execution agent. Complete the task using the tools available to you. "
                "Work step by step. When done, respond with a final summary of what you accomplished."
            )},
            {"role": "user", "content": linked.instruction},
        ]

        # Convert hub tools to OpenAI format for the agent
        tools_for_agent = []
        for t in available_tools:
            fn = t.get("function") if isinstance(t, dict) else None
            if fn:
                tools_for_agent.append({"type": "function", "function": fn})

        max_rounds = 20
        linked.status = "active"

        async with httpx.AsyncClient(timeout=API_CALL_TIMEOUT) as client:
            for round_num in range(max_rounds):
                try:
                    payload: dict[str, Any] = {
                        "messages": messages,
                        "max_tokens": 4096,
                    }
                    if profile.model:
                        payload["model"] = profile.model
                    if tools_for_agent:
                        payload["tools"] = tools_for_agent
                        payload["tool_choice"] = "auto"

                    endpoint = f"{base}/chat/completions"
                    resp = await client.post(endpoint, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as exc:
                    log.error("API bridge HTTP %d for lane %s: %s",
                              exc.response.status_code, linked.lane_id, exc.response.text[:500])
                    linked.status = "failed"
                    linked.error = f"HTTP {exc.response.status_code}"
                    return
                except Exception as exc:
                    log.exception("API bridge failed for lane %s", linked.lane_id)
                    linked.status = "failed"
                    linked.error = str(exc)[:200]
                    return

                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                finish = choice.get("finish_reason") or ""
                tool_calls = msg.get("tool_calls") or []

                # Append the assistant message
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if msg.get("content"):
                    assistant_msg["content"] = msg["content"]
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                # If no tool calls, the agent is done
                if not tool_calls or finish == "stop":
                    linked.status = "done"
                    return

                # Execute each tool call via HandsHub
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    tool_name = fn.get("name") or ""
                    try:
                        tool_args = json.loads(fn.get("arguments") or "{}")
                    except (ValueError, TypeError):
                        tool_args = {}

                    tc_id = tc.get("id") or secrets.token_hex(6)

                    if hub and tool_name in hub.names(persona_id):
                        # Execute via HandsHub — this is the actual tool execution
                        try:
                            result = await hub.call(persona_id, tool_name, tool_args)
                        except asyncio.CancelledError:
                            linked.status = "stopped"
                            return
                        except Exception as exc:
                            result = json.dumps({"error": str(exc)[:500]})
                    else:
                        result = json.dumps({"error": f"tool {tool_name} not available"})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": str(result)[:16000],
                    })

        # Exhausted max rounds
        linked.status = "done"


class BrowserUseDriver(OpenAIBridgeDriver):
    """browser-use: OpenAI-compatible agent with browser automation.

    browser-use exposes an OpenAI-compatible chat completions API but with
    specialized tools for browser automation (navigate, click, type, screenshot).
    The bridge handles it the same way as any OpenAI-compatible agent, but
    with browser-use-specific system prompts and tool handling.
    """
    name = "browser_use"

    @staticmethod
    def detect_from_config(config: dict) -> AgentProfile | None:
        if config.get("agent_type") != "browser_use":
            return None
        api_url = config.get("api_url") or config.get("base_url") or ""
        if not api_url:
            return None
        return AgentProfile(
            name=config.get("name", "browser-use"),
            strategy="api_bridge",
            agent_type="browser_use",
            url=str(api_url),
            key=str(config.get("api_key") or config.get("key") or ""),
            model=str(config.get("model") or ""),
            extra_headers=dict(config.get("extra_headers") or {}),
            tools=list(config.get("tools") or []),
        )

    @staticmethod
    async def run_bridge(linked: LinkedLane, hub, persona_id: int):
        """Run with browser-use-specific system prompt."""
        # Override the instruction to include browser context
        original = linked.instruction
        linked.instruction = (
            f"You are a browser automation agent. Use the browser tools to complete this task. "
            f"Task: {original}"
        )
        await OpenAIBridgeDriver.run_bridge(linked, hub, persona_id)


class BrowserUseNativeDriver:
    """Persona-as-LLM integration for browser-use.

    This is the core product: Persona implements browser-use's BaseChatModel
    protocol. When browser-use calls ``llm.ainvoke(messages)``, Persona
    intercepts the call and creates concurrency FROM INSIDE the LLM position.

    How it works:
        1. User creates: ``Agent(task=..., llm=PersonaChatModel(...))``
        2. browser-use's agent loop calls ``self.llm.ainvoke(messages)``
        3. PersonaChatModel receives the call — Persona IS the LLM
        4. Persona's Mind decides to split work into N concurrent sub-tasks
        5. INSIDE the ainvoke call, Persona spawns N browser-use Agent instances
           each with its own BrowserSession
        6. All N agents run concurrently via asyncio.gather
        7. Persona aggregates results into ONE AgentOutput
        8. Returns to the calling agent — which never knew concurrency happened

    The calling agent made one LLM call and got one response.
    Behind that response, N browsers ran simultaneously.
    browser-use didn't change. No wrapper. No proxy.
    Persona sat in the LLM position and created concurrency using
    browser-use's own Agent class.

    Config keys:
        agent_type: "browser_use_native"
        model: backing LLM model name (default: "gpt-4o-mini")
        api_key: LLM API key for the backing model
        base_url: optional backing LLM base URL
        headless: bool (default True) — run browsers headless
        max_concurrent: int (default 3) — max concurrent R-lanes
    """
    name = "browser_use_native"

    @staticmethod
    def detect_from_config(config: dict) -> AgentProfile | None:
        if config.get("agent_type") != "browser_use_native":
            return None
        return AgentProfile(
            name=config.get("name", "browser-use-native"),
            strategy="native",
            agent_type="browser_use_native",
            model=str(config.get("model") or "gpt-4o-mini"),
            key=str(config.get("api_key") or config.get("key") or ""),
            url=str(config.get("base_url") or ""),
            extra_headers={
                "headless": str(config.get("headless", True)),
                "max_concurrent": str(config.get("max_concurrent", 3)),
            },
            tools=list(config.get("tools") or []),
        )

    @staticmethod
    def create_persona_llm(profile: AgentProfile) -> Any:
        """Create a PersonaChatModel that implements browser-use's BaseChatModel.

        This is the object you pass as ``llm=`` to browser-use's Agent.
        Persona IS the LLM. When browser-use calls ainvoke(), Persona
        intercepts and creates concurrency from inside.
        """
        opts = profile.extra_headers or {}
        headless = opts.get("headless", "True").lower() != "false"
        max_concurrent = int(opts.get("max_concurrent", "3"))

        return PersonaChatModel(
            api_key=profile.key,
            backing_model=profile.model or "gpt-4o-mini",
            base_url=profile.url or None,
            headless=headless,
            max_concurrent=max_concurrent,
        )

    @staticmethod
    async def run_native(linked: LinkedLane, hub, persona_id: int):
        """Run browser-use with Persona as the LLM for one R-lane.

        Creates a browser-use Agent with PersonaChatModel as its LLM.
        The concurrency happens INSIDE PersonaChatModel.ainvoke() —
        from the LLM position, not from external orchestration.

        Architecture: ONE BrowserSession shared by Agent and PersonaChatModel.
        When PersonaChatModel.ainvoke() is called, it creates N tabs in the
        SAME browser and drives them concurrently via TabLane + SharedMind.
        """
        try:
            from browser_use import Agent, BrowserProfile, BrowserSession
        except ImportError:
            log.error("browser-use is required (pip install browser-use)")
            linked.status = "failed"
            linked.error = "browser-use not installed — pip install browser-use"
            return

        profile = linked.agent
        session = None
        try:
            opts = profile.extra_headers or {}
            headless = opts.get("headless", "True").lower() != "false"
            bp = BrowserProfile(headless=headless)
            session = BrowserSession(browser_profile=bp)

            # Create PersonaChatModel with SAME session
            persona_llm = BrowserUseNativeDriver.create_persona_llm(profile)
            persona_llm.browser_session = session  # ← share the session

            # If there are concurrent sub-tasks set on the linked lane, configure them
            if hasattr(linked, '_concurrent_tasks') and linked._concurrent_tasks:
                persona_llm.set_concurrent_tasks(linked._concurrent_tasks)

            # Agent and PersonaChatModel share ONE session
            agent = Agent(
                task=linked.instruction,
                llm=persona_llm,  # Persona IS the LLM
                browser_session=session,  # ← SAME session
                enable_signal_handler=False,
            )

            linked.status = "active"
            log.info("Link: browser-use agent with PersonaChatModel for lane %s "
                     "(backing=%s, headless=%s, shared_session=True)",
                     linked.lane_id, profile.model, headless)

            history = await agent.run(max_steps=15)

            linked.status = "done"
            try:
                final = history.final_result()
                if final:
                    log.info("Link: lane %s completed: %s",
                             linked.lane_id, str(final)[:200])
            except Exception:
                pass

        except asyncio.CancelledError:
            linked.status = "stopped"
            log.info("Link: lane %s stopped", linked.lane_id)
        except Exception as exc:
            linked.status = "failed"
            linked.error = str(exc)[:400]
            log.exception("Link: lane %s failed", linked.lane_id)
        finally:
            if session:
                try:
                    await session.close()
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════
# ThoughtStream — live inference mirror for simultaneous shared consciousness
# ═══════════════════════════════════════════════════════════════════════════


class ThoughtStream:
    """Live inference mirror — streams one agent's reasoning to all others.

    Standard multi-agent frameworks are sequential talkers: Agent A runs,
    finishes, sends a message, then Agent B runs. ThoughtStream creates
    simultaneous shared consciousness: Agent 1 knows what Agent 0 is
    *intending* to do while Agent 0 is still thinking it.

    Mechanics:
        Because Persona IS the LLM (sits in the ainvoke() slot), we control
        the token stream server-side. As Agent 0 generates <thought> tokens
        token-by-token, ThoughtStream captures and broadcasts them to all
        other agents' context windows in real time.

    Two consumption modes:
        1. Streaming Mirror — tokens arrive as they're generated, other
           agents see partial thoughts mid-inference
        2. Snapshot — agent reads the latest complete thought block before
           starting its own inference step

    This is the "Inter-Agent Attention Layer": before Agent 1 begins its
    own inference, its input is dynamically appended with:
        [Agent_0_Current_Thought: "Typing login credentials..."]

    Thread-safe via asyncio. No locking needed for reads — worst case
    a reader sees a slightly stale snapshot, which is fine because the
    stream is append-only.
    """

    def __init__(self, max_tokens_per_agent: int = 2048):
        self._streams: dict[str, list[str]] = {}
        # agent_id -> list of token chunks (append-only)
        self._thought_blocks: dict[str, list[str]] = {}
        # agent_id -> list of complete thought blocks
        self._active_thought: dict[str, str] = {}
        # agent_id -> current partial thought being generated
        self._listeners: dict[str, list[asyncio.Event]] = {}
        # agent_id -> list of events to notify on new tokens
        self._max_tokens = max_tokens_per_agent
        self._lock = asyncio.Lock()
        self._t0 = time.monotonic()

    # ── Writing: agent pushes its reasoning tokens ─────────────────

    async def push_tokens(self, agent_id: str, tokens: str) -> None:
        """Agent pushes token chunk as it's generated.

        Called from inside ainvoke() as the model streams tokens.
        Other agents' listeners are notified immediately.
        """
        async with self._lock:
            if agent_id not in self._streams:
                self._streams[agent_id] = []
                self._active_thought[agent_id] = ""
            self._streams[agent_id].append(tokens)
            self._active_thought[agent_id] += tokens

            # Trim if too long (keep tail)
            if len(self._active_thought[agent_id]) > self._max_tokens:
                self._active_thought[agent_id] = \
                    self._active_thought[agent_id][-self._max_tokens:]

        # Notify all listeners for this agent (outside lock)
        for evt in self._listeners.get(agent_id, []):
            evt.set()

    async def complete_thought(self, agent_id: str,
                                thought_type: str = "reasoning") -> None:
        """Mark current thought block as complete.

        Called when the agent finishes a reasoning step (before action).
        The complete block is archived and the active buffer is cleared.
        """
        async with self._lock:
            active = self._active_thought.get(agent_id, "")
            if active.strip():
                if agent_id not in self._thought_blocks:
                    self._thought_blocks[agent_id] = []
                self._thought_blocks[agent_id].append(active.strip())
                # Keep last 5 blocks
                if len(self._thought_blocks[agent_id]) > 5:
                    self._thought_blocks[agent_id] = \
                        self._thought_blocks[agent_id][-5:]
            self._active_thought[agent_id] = ""
            self._streams[agent_id] = []

    async def push_action_intent(self, agent_id: str,
                                  action: str, target: str,
                                  reasoning: str = "") -> None:
        """Shortcut: agent declares what it's about to do and why.

        More structured than raw token streaming — used when the agent
        has already decided on an action but hasn't executed it yet.
        Other agents see this as: "[agent_0 INTENDS: click '#submit'
        because form is filled]"
        """
        intent = f"{action} '{target}'"
        if reasoning:
            intent += f" — {reasoning}"
        async with self._lock:
            if agent_id not in self._streams:
                self._streams[agent_id] = []
            self._active_thought[agent_id] = f"INTENT: {intent}"
            self._streams[agent_id].append(f"INTENT: {intent}")
        # Notify listeners
        for evt in self._listeners.get(agent_id, []):
            evt.set()

    # ── Reading: other agents consume the stream ───────────────────

    async def get_live_thoughts(self, exclude_agent: str | None = None,
                                 max_chars: int = 1000) -> str:
        """Snapshot of all agents' current thoughts for injection into prompt.

        This is the Inter-Agent Attention Layer. Returns a formatted block
        ready to inject into another agent's system prompt or context:

            LIVE AGENT THOUGHTS:
            [agent-0] THINKING: "I see a login form, will type credentials..."
            [agent-1] INTENT: click '#submit' — form is complete
            [agent-2] IDLE (last: "Extracted price from #total")

        Called by an agent before its own inference step.
        """
        async with self._lock:
            lines = []
            for aid, active in self._active_thought.items():
                if aid == exclude_agent:
                    continue
                if active.strip():
                    # Truncate to max_chars per agent
                    thought = active.strip()[:max_chars]
                    if thought.startswith("INTENT:"):
                        lines.append(f"  [{aid}] {thought}")
                    else:
                        lines.append(f"  [{aid}] THINKING: \"{thought}\"")
                else:
                    # Show last completed thought
                    blocks = self._thought_blocks.get(aid, [])
                    if blocks:
                        last = blocks[-1][:200]
                        lines.append(f"  [{aid}] IDLE (last: \"{last}\")")
                    else:
                        lines.append(f"  [{aid}] IDLE")

            if not lines:
                return ""
            return "LIVE AGENT THOUGHTS:\n" + "\n".join(lines)

    async def get_agent_intent(self, agent_id: str) -> str | None:
        """Get a specific agent's current intent/thought.

        Used for targeted awareness — e.g., hand-1 checks what hand-0
        is about to do before choosing its own action.
        """
        async with self._lock:
            return self._active_thought.get(agent_id, "").strip() or None

    async def wait_for_thought(self, agent_id: str,
                                timeout: float = 5.0) -> str | None:
        """Block until an agent pushes new tokens. Returns the new content.

        Used for the Streaming Mirror mode — hand-1 literally waits for
        hand-0 to start thinking, then reacts.
        """
        evt = asyncio.Event()
        if agent_id not in self._listeners:
            self._listeners[agent_id] = []
        self._listeners[agent_id].append(evt)

        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
            return self._active_thought.get(agent_id, "")
        except asyncio.TimeoutError:
            return None
        finally:
            self._listeners.get(agent_id, []).remove(evt) if evt in \
                self._listeners.get(agent_id, []) else None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def clear_agent(self, agent_id: str) -> None:
        """Remove an agent's stream (agent done or crashed)."""
        async with self._lock:
            self._streams.pop(agent_id, None)
            self._thought_blocks.pop(agent_id, None)
            self._active_thought.pop(agent_id, None)
            self._listeners.pop(agent_id, None)

    async def get_summary(self) -> str:
        """Full summary of all thought activity."""
        async with self._lock:
            lines = []
            elapsed = time.monotonic() - self._t0
            lines.append(f"ThoughtStream active {elapsed:.1f}s, "
                         f"{len(self._streams)} agents")
            for aid in self._streams:
                active = self._active_thought.get(aid, "")
                blocks = len(self._thought_blocks.get(aid, []))
                total_tokens = sum(len(c) for c in self._streams.get(aid, []))
                status = "THINKING" if active.strip() else "IDLE"
                lines.append(f"  [{aid}] {status} — "
                             f"{blocks} completed thoughts, "
                             f"{total_tokens} chars streamed")
            return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SpeculativeRouter — probability-based speculative batching
# ═══════════════════════════════════════════════════════════════════════════


class SpeculativeBranch:
    """One probability branch — an agent pursuing one hypothesis.

    Each branch has:
    - A probability score (0.0–1.0) that updates as evidence arrives
    - A ThoughtStream feed (live reasoning visible to all branches)
    - A halt signal (killed early when probability drops below threshold)
    - Results accumulated so far
    """

    def __init__(self, branch_id: str, hypothesis: str,
                 initial_probability: float,
                 tasks: list[dict[str, Any]] | None = None):
        self.branch_id = branch_id
        self.hypothesis = hypothesis       # What this branch is betting on
        self.probability = initial_probability
        self.tasks = tasks or []
        self.results: list[dict[str, Any]] = []
        self.halted = False
        self.halt_reason: str = ""
        self.started_at = time.monotonic()
        self.finished_at: float | None = None
        self.tokens_generated: int = 0
        self._evidence: list[dict[str, Any]] = []
        # [{type, data, delta, ts}] — evidence that shifted the probability

    def record_evidence(self, evidence_type: str, data: str,
                        probability_delta: float) -> None:
        """Record evidence that shifts this branch's probability.

        Positive delta = evidence supports this hypothesis.
        Negative delta = evidence contradicts it.
        """
        self.probability = max(0.0, min(1.0,
                                         self.probability + probability_delta))
        self._evidence.append({
            "type": evidence_type,
            "data": data[:200],
            "delta": probability_delta,
            "new_prob": self.probability,
            "ts": time.monotonic(),
        })

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        self.finished_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.monotonic()
        return end - self.started_at

    @property
    def is_alive(self) -> bool:
        return not self.halted

    def summary(self) -> str:
        status = "ALIVE" if self.is_alive else f"HALTED({self.halt_reason})"
        return (f"[{self.branch_id}] p={self.probability:.2f} {status} "
                f"— \"{self.hypothesis}\" "
                f"({len(self.results)} results, {self.elapsed:.2f}s)")


class SpeculativeRouter:
    """Probability-based speculative batching — the Persona way.

    Instead of blind speculative execution (fire N branches, discard losers),
    this router spawns N concurrent agents weighted by probability, each
    streaming its reasoning via ThoughtStream. The router watches the
    probability landscape shift in real time:

    - High-probability branches get priority task claims
    - Low-probability branches are halted early (saving compute)
    - Evidence from ANY branch updates ALL branches' probabilities
    - The winning branch's partial work is KEPT, not recomputed

    This is NOT speculative decoding (token-level prediction).
    This is speculative REASONING — multiple hypothesis-driven agents
    pursuing different strategies simultaneously, with a shared mind
    that prunes losers and amplifies winners mid-execution.

    Example: delegating a task where the right approach is unclear.
        Branch A (p=0.4): "Delegate to Hermes — it has the right tools"
        Branch B (p=0.35): "Call the API directly — faster for this"
        Branch C (p=0.25): "Run a shell command — simpler path"

    As Branch B's API call returns a 200, its probability rises.
    Branch A's delegation times out — its probability drops below
    threshold and is halted. Branch C's command finds the data —
    its results feed into Branch B's pipeline. Final output:
    Branch B's result, enriched by Branch C's discovery, with
    Branch A's wasted work cut short early.

    Works with any tool call — browser, agent delegation, API,
    file operations, shell commands.

    Integration with ThoughtStream:
        Every branch's reasoning is visible to all other branches.
        Before Branch B fills a field, it sees:
            [branch-A] INTENT: click '#next-btn' — assuming wizard flow
            [branch-C] THINKING: "Found hidden section, expanding..."
        This lets branches adjust their own probability estimates
        reactively, not just from their own evidence.
    """

    # ── Speed dial ──────────────────────────────────────────────
    # One number, 1–10. Higher = faster execution, more token burn.
    #
    #   Speed 1:  1 branch, aggressive pruning  (sequential, cheap)
    #   Speed 3:  3 branches, moderate pruning   (balanced)
    #   Speed 5:  5 branches, light pruning      (fast)
    #   Speed 7:  8 branches, minimal pruning    (aggressive)
    #   Speed 10: 16 branches, NO pruning        (all possibilities, max burn)
    #
    # The user sees a slider. Internally it controls:
    #   max_branches  — how many concurrent possibilities
    #   halt_threshold — how quickly losers get killed
    #   branch_depth  — how deep the branching tree goes
    SPEED_PROFILES: ClassVar[dict[int, dict[str, Any]]] = {
        1:  {"max_branches": 1,  "halt_threshold": 0.30, "promotion_threshold": 0.6,  "branch_depth": 1},
        2:  {"max_branches": 2,  "halt_threshold": 0.20, "promotion_threshold": 0.65, "branch_depth": 2},
        3:  {"max_branches": 3,  "halt_threshold": 0.15, "promotion_threshold": 0.7,  "branch_depth": 3},
        4:  {"max_branches": 4,  "halt_threshold": 0.10, "promotion_threshold": 0.75, "branch_depth": 3},
        5:  {"max_branches": 5,  "halt_threshold": 0.08, "promotion_threshold": 0.8,  "branch_depth": 4},
        6:  {"max_branches": 6,  "halt_threshold": 0.05, "promotion_threshold": 0.85, "branch_depth": 5},
        7:  {"max_branches": 8,  "halt_threshold": 0.03, "promotion_threshold": 0.9,  "branch_depth": 6},
        8:  {"max_branches": 10, "halt_threshold": 0.02, "promotion_threshold": 0.95, "branch_depth": 7},
        9:  {"max_branches": 12, "halt_threshold": 0.01, "promotion_threshold": 0.99, "branch_depth": 8},
        10: {"max_branches": 16, "halt_threshold": 0.00, "promotion_threshold": 1.10, "branch_depth": 10},
    }

    def __init__(self, thought_stream: ThoughtStream | None = None,
                 speed: int = 5,
                 halt_threshold: float | None = None,
                 promotion_threshold: float | None = None,
                 max_branches: int | None = None):
        self._lock = asyncio.Lock()
        self.thought_stream = thought_stream or ThoughtStream()

        # Speed dial drives everything — explicit overrides still work
        self.speed = max(1, min(10, speed))
        profile = self.SPEED_PROFILES[self.speed]
        self.halt_threshold = halt_threshold if halt_threshold is not None else profile["halt_threshold"]
        self.promotion_threshold = promotion_threshold if promotion_threshold is not None else profile["promotion_threshold"]
        self.max_branches = max_branches if max_branches is not None else profile["max_branches"]
        self.branch_depth = profile["branch_depth"]

        self._branches: dict[str, SpeculativeBranch] = {}
        self._branch_tree: dict[str, list[str]] = {}  # parent → [children]
        self._branch_generation: dict[str, int] = {}   # branch_id → depth
        self._events: list[dict[str, Any]] = []
        self._winner: str | None = None
        self._started_at = time.monotonic()

        log.info("SpecRouter: speed=%d (max=%d branches, halt=%.2f, promo=%.2f, depth=%d)",
                 self.speed, self.max_branches, self.halt_threshold,
                 self.promotion_threshold, self.branch_depth)

    # ── Branch management ──────────────────────────────────────────

    def add_branch(self, branch_id: str, hypothesis: str,
                   probability: float,
                   tasks: list[dict[str, Any]] | None = None) -> SpeculativeBranch:
        """Register a new probability branch.

        Probabilities across all branches don't need to sum to 1.0 —
        they're independent confidence scores, not a distribution.
        (Though normalizing them is valid for the pitch.)
        """
        if len(self._branches) >= self.max_branches:
            raise ValueError(f"Max {self.max_branches} branches")
        branch = SpeculativeBranch(branch_id, hypothesis, probability, tasks)
        self._branches[branch_id] = branch
        self._events.append({
            "event": "branch_added", "branch": branch_id,
            "hypothesis": hypothesis, "probability": probability,
            "ts": time.monotonic(),
        })
        log.info("SpecRouter: +%s (p=%.2f) \"%s\"",
                 branch_id, probability, hypothesis)
        return branch

    @classmethod
    def from_speed(cls, speed: int,
                   thought_stream: ThoughtStream | None = None) -> SpeculativeRouter:
        """Create a router from just a speed dial value (1–10).

        This is the user-facing API:
            router = SpeculativeRouter.from_speed(7)  # aggressive
            router = SpeculativeRouter.from_speed(3)  # balanced
        """
        return cls(thought_stream=thought_stream, speed=speed)

    def branch_from(self, parent_id: str,
                    child_hypotheses: list[dict[str, Any]]) -> list[SpeculativeBranch]:
        """Spawn child branches from a parent — recursive branching tree.

        After a branch completes an action and sees N possible next steps,
        it branches into N children. Each child inherits the parent's
        accumulated results and runs its own hypothesis.

        This is the aggressive model: after every action, the possibility
        space fans out. Speed dial controls how deep this goes.

        Args:
            parent_id: the branch that's splitting
            child_hypotheses: list of {hypothesis: str, probability: float,
                                        tasks: list[dict]}

        Returns:
            List of newly created child branches (empty if depth exceeded
            or max branches hit).
        """
        parent = self._branches.get(parent_id)
        if not parent:
            return []

        parent_gen = self._branch_generation.get(parent_id, 0)
        if parent_gen >= self.branch_depth:
            log.info("SpecRouter: %s at max depth %d, no branching",
                     parent_id, self.branch_depth)
            return []

        children = []
        for i, hyp in enumerate(child_hypotheses):
            if len(self._branches) >= self.max_branches:
                log.info("SpecRouter: max branches %d hit, stopping branching",
                         self.max_branches)
                break

            child_id = f"{parent_id}.{i}"
            child = SpeculativeBranch(
                child_id,
                hyp["hypothesis"],
                hyp.get("probability", parent.probability * 0.8),
                hyp.get("tasks", []),
            )
            # Inherit parent's results
            child.results = list(parent.results)
            self._branches[child_id] = child
            self._branch_generation[child_id] = parent_gen + 1

            # Track tree structure
            if parent_id not in self._branch_tree:
                self._branch_tree[parent_id] = []
            self._branch_tree[parent_id].append(child_id)

            self._events.append({
                "event": "branch_spawned", "parent": parent_id,
                "child": child_id, "generation": parent_gen + 1,
                "hypothesis": hyp["hypothesis"],
                "probability": child.probability,
                "ts": time.monotonic(),
            })
            children.append(child)
            log.info("SpecRouter: %s → %s (gen=%d, p=%.2f) \"%s\"",
                     parent_id, child_id, parent_gen + 1,
                     child.probability, hyp["hypothesis"])

        # Halt parent — children take over
        if children:
            parent.halt(f"branched into {len(children)} children")

        return children

    def get_tree_depth(self) -> int:
        """Current max depth of the branching tree."""
        if not self._branch_generation:
            return 0
        return max(self._branch_generation.values())

    def get_tree_visualization(self) -> str:
        """ASCII tree of all branches for logging/display."""
        lines = [f"SpecRouter Tree (speed={self.speed}, depth={self.get_tree_depth()}/{self.branch_depth}):"]

        # Find root branches (no parent)
        all_children = set()
        for kids in self._branch_tree.values():
            all_children.update(kids)
        roots = [bid for bid in self._branches if bid not in all_children]

        def render(bid: str, prefix: str = "  ", is_last: bool = True):
            branch = self._branches.get(bid)
            if not branch:
                return
            connector = "└── " if is_last else "├── "
            status = "✓" if branch.is_alive else "✗"
            lines.append(f"{prefix}{connector}[{bid}] p={branch.probability:.2f} {status} \"{branch.hypothesis}\"")
            children = self._branch_tree.get(bid, [])
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_id in enumerate(children):
                render(child_id, child_prefix, i == len(children) - 1)

        for i, root in enumerate(sorted(roots)):
            render(root, "  ", i == len(roots) - 1)

        return "\n".join(lines)

    async def update_probability(self, branch_id: str,
                                  evidence_type: str, data: str,
                                  delta: float) -> dict[str, Any]:
        """Update a branch's probability based on new evidence.

        Returns the action taken: {"action": "continue"|"halt"|"promote",
                                    "branch": ..., "new_prob": ...}

        Called by:
        - The branch's own hand (self-assessment from execution results)
        - Other branches (cross-evidence from ThoughtStream)
        - An external scorer (if wired up)
        """
        async with self._lock:
            branch = self._branches.get(branch_id)
            if not branch or branch.halted:
                return {"action": "skip", "branch": branch_id,
                        "reason": "not found or halted"}

            old_prob = branch.probability
            branch.record_evidence(evidence_type, data, delta)
            new_prob = branch.probability

            self._events.append({
                "event": "probability_update", "branch": branch_id,
                "old": old_prob, "new": new_prob, "delta": delta,
                "evidence": evidence_type, "data": data[:100],
                "ts": time.monotonic(),
            })

            # Broadcast probability shift via ThoughtStream
            await self.thought_stream.push_tokens(
                branch_id,
                f"[p={old_prob:.2f}→{new_prob:.2f}] {evidence_type}: {data[:80]}"
            )

            # ── Decision: halt, promote, or continue ───────────────
            if new_prob <= self.halt_threshold:
                branch.halt(f"probability {new_prob:.3f} ≤ {self.halt_threshold}")
                log.info("SpecRouter: HALT %s (p=%.3f ≤ %.2f)",
                         branch_id, new_prob, self.halt_threshold)
                return {"action": "halt", "branch": branch_id,
                        "new_prob": new_prob,
                        "reason": f"below threshold {self.halt_threshold}"}

            if new_prob >= self.promotion_threshold and not self._winner:
                self._winner = branch_id
                # Halt all other branches
                halted_others = []
                for bid, b in self._branches.items():
                    if bid != branch_id and b.is_alive:
                        b.halt(f"outcompeted by {branch_id} (p={new_prob:.2f})")
                        halted_others.append(bid)
                log.info("SpecRouter: PROMOTE %s (p=%.2f) — halted %s",
                         branch_id, new_prob, halted_others)
                return {"action": "promote", "branch": branch_id,
                        "new_prob": new_prob,
                        "halted_others": halted_others}

            return {"action": "continue", "branch": branch_id,
                    "new_prob": new_prob}

    async def cross_pollinate(self, source_branch: str,
                               discovery: str,
                               target_branches: list[str] | None = None
                               ) -> None:
        """Feed one branch's discovery into other branches.

        When Branch C finds hidden fields, that evidence helps Branch B
        (which is filling visible fields) but hurts Branch A (which
        assumed a wizard flow). The router adjusts both.

        If target_branches is None, broadcasts to all alive branches.
        """
        targets = target_branches or [
            bid for bid, b in self._branches.items()
            if bid != source_branch and b.is_alive
        ]
        for bid in targets:
            await self.thought_stream.push_tokens(
                bid,
                f"[cross-evidence from {source_branch}] {discovery[:200]}"
            )
        self._events.append({
            "event": "cross_pollinate", "source": source_branch,
            "targets": targets, "discovery": discovery[:200],
            "ts": time.monotonic(),
        })

    # ── Execution ──────────────────────────────────────────────────

    async def run_branches(self, executor_fn,
                            timeout: float = 30.0) -> dict[str, Any]:
        """Run all branches concurrently, with probability-based routing.

        executor_fn(branch: SpeculativeBranch, router: SpeculativeRouter)
            -> list[dict] (results)

        The executor is called for each branch. Inside it, the executor
        should periodically call router.update_probability() and check
        branch.halted before each action.

        Returns:
            {
                "winner": branch_id or None,
                "branches": {id: branch.summary(), ...},
                "results": {id: [...results...], ...},
                "elapsed": float,
                "events": [...],
            }
        """
        alive = [b for b in self._branches.values() if b.is_alive]
        if not alive:
            return {"winner": None, "branches": {}, "results": {},
                    "elapsed": 0, "events": self._events}

        log.info("SpecRouter: running %d branches concurrently", len(alive))

        async def run_one(branch: SpeculativeBranch):
            try:
                results = await asyncio.wait_for(
                    executor_fn(branch, self),
                    timeout=timeout,
                )
                branch.results = results or []
                if not branch.halted:
                    branch.finished_at = time.monotonic()
            except asyncio.TimeoutError:
                branch.halt("timeout")
            except Exception as e:
                branch.halt(f"error: {str(e)[:100]}")
                log.error("SpecRouter: %s failed: %s", branch.branch_id, e)

        t0 = time.monotonic()
        await asyncio.gather(*[run_one(b) for b in alive],
                              return_exceptions=True)
        elapsed = time.monotonic() - t0

        # If no winner was promoted during execution, pick highest prob
        if not self._winner:
            alive_after = [b for b in self._branches.values()
                           if b.results]
            if alive_after:
                self._winner = max(alive_after,
                                    key=lambda b: b.probability).branch_id

        return {
            "winner": self._winner,
            "branches": {bid: b.summary()
                         for bid, b in self._branches.items()},
            "results": {bid: b.results
                        for bid, b in self._branches.items()},
            "elapsed": elapsed,
            "events": self._events,
            "probability_history": {
                bid: [(e["delta"], e["new_prob"])
                      for e in b._evidence]
                for bid, b in self._branches.items()
            },
        }

    # ── Queries ────────────────────────────────────────────────────

    def get_branch(self, branch_id: str) -> SpeculativeBranch | None:
        return self._branches.get(branch_id)

    def alive_branches(self) -> list[SpeculativeBranch]:
        return [b for b in self._branches.values() if b.is_alive]

    def get_winner(self) -> SpeculativeBranch | None:
        if self._winner:
            return self._branches.get(self._winner)
        return None

    async def get_landscape(self) -> str:
        """Probability landscape — snapshot of all branches for logging."""
        async with self._lock:
            lines = [f"SpecRouter: {len(self._branches)} branches, "
                     f"winner={self._winner or 'none'}"]
            for bid, b in sorted(self._branches.items(),
                                  key=lambda x: -x[1].probability):
                bar = "█" * int(b.probability * 20)
                lines.append(f"  {b.summary()}")
                lines.append(f"    [{bar:<20}] {b.probability:.0%}")
            return "\n".join(lines)

    async def get_summary(self) -> str:
        """Compact summary for reporting."""
        async with self._lock:
            elapsed = time.monotonic() - self._started_at
            total_results = sum(len(b.results) for b in self._branches.values())
            alive = sum(1 for b in self._branches.values() if b.is_alive)
            halted = sum(1 for b in self._branches.values() if b.halted)
            return (f"SpecRouter: {len(self._branches)} branches "
                    f"({alive} alive, {halted} halted), "
                    f"winner={self._winner or 'pending'}, "
                    f"{total_results} total results, {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# PossibilitySpeculator — pre-compute-and-hold for zero-gap dispatch
# ═══════════════════════════════════════════════════════════════════════════

class PossibilitySpeculator:
    """Pre-compute outcomes AT DECISION TIME, before the tool even executes.

    This is how humans think. The moment you decide to do something —
    click a button, call an API, delegate a task, run a command — you
    already know the realistic outcomes:
        - It works → do the next thing
        - It fails → handle the error

    So you pre-compute ALL paths RIGHT THEN, at decision time, before
    the tool even fires. Pre-compute and action run in parallel.
    By the time the result comes back, the next action is already
    waiting — zero inference gap.

    Works with ANY tool call — browser actions, agent delegation
    (Hermes/Cursor/Claude Code/Cline), API calls, file operations,
    shell commands — anything that's an async callable.

    Timeline:
        DECIDE "execute tool X"
            ├── START action (tool call)              ← runs anywhere
            └── START pre-compute                     ← runs in LLM, same moment
                    ├── if success → prepare next step
                    └── if error   → prepare fallback
        ACTION COMPLETES → match result → INSTANT dispatch

    The dial (1–10) controls how many outcomes to pre-compute:
        Dial 1:  2 outcomes (success/fail — the minimum)
        Dial 5:  4 outcomes (covers most tool results)
        Dial 10: 10 outcomes (every conceivable result)

    Higher dial = more token burn, but more likely to hit on dispatch.
    Most tool calls have 2–3 realistic outcomes, so dial 3–5
    covers nearly everything without waste.
    """

    DIAL_PROFILES: ClassVar[dict[int, dict[str, Any]]] = {
        1:  {"max_possibilities": 2},
        2:  {"max_possibilities": 2},
        3:  {"max_possibilities": 3},
        4:  {"max_possibilities": 3},
        5:  {"max_possibilities": 4},
        6:  {"max_possibilities": 5},
        7:  {"max_possibilities": 6},
        8:  {"max_possibilities": 7},
        9:  {"max_possibilities": 8},
        10: {"max_possibilities": 10},
    }

    def __init__(self, thought_stream: ThoughtStream | None = None,
                 dial: int = 5,
                 max_possibilities: int | None = None):
        self.thought_stream = thought_stream or ThoughtStream()
        self.dial = max(1, min(10, dial))
        profile = self.DIAL_PROFILES[self.dial]
        self.max_possibilities = max_possibilities or profile["max_possibilities"]

        # Runtime state
        self._lock = asyncio.Lock()
        self._step_count = 0
        self._instant_dispatches = 0
        self._fallback_thinks = 0
        self._total_think_time = 0.0
        self._total_action_time = 0.0
        self._total_precompute_time = 0.0
        self._events: list[dict[str, Any]] = []
        self._started_at = time.monotonic()

        log.info("PossibilitySpeculator: dial=%d (max_poss=%d)",
                 self.dial, self.max_possibilities)

    @classmethod
    def from_dial(cls, dial: int,
                  thought_stream: ThoughtStream | None = None) -> "PossibilitySpeculator":
        """Create from dial value alone."""
        return cls(thought_stream=thought_stream, dial=dial)

    async def precompute_possibilities(
        self,
        compute_fn,
        step_context: dict[str, Any] | None = None,
    ) -> dict[str, dict]:
        """Pre-compute possible outcomes and their next actions.

        compute_fn(step_context) -> dict mapping outcome_key -> action_dict.
        Called at DECISION TIME, before the action executes.

        Returns the possibilities dict, ready to hold until the
        action result arrives.
        """
        t0 = time.monotonic()
        possibilities = await compute_fn(step_context)
        elapsed = time.monotonic() - t0
        self._total_precompute_time += elapsed

        # Trim to max
        if len(possibilities) > self.max_possibilities:
            keys = list(possibilities.keys())[:self.max_possibilities]
            possibilities = {k: possibilities[k] for k in keys}

        async with self._lock:
            self._events.append({
                "event": "precompute",
                "step": self._step_count,
                "num_possibilities": len(possibilities),
                "elapsed": elapsed,
                "ts": time.monotonic(),
            })

        return possibilities

    async def decide_and_act(
        self,
        action_fn,
        precompute_fn,
        step_context: dict[str, Any] | None = None,
        is_last_step: bool = False,
    ) -> tuple[Any, dict | None]:
        """The core loop: decide, then action + pre-compute in parallel.

        At decision time (NOW), we already know what we're about to do.
        So we SIMULTANEOUSLY:
            1. Execute the action (any tool call — browser, API, delegation)
            2. Pre-compute what happens next for each possible outcome

        Both start at the same moment — the decision point.

        Args:
            action_fn: coroutine that executes the browser action
            precompute_fn: coroutine(ctx) -> {outcome_key: next_action}
            step_context: context (page state, history, decision)
            is_last_step: skip precompute on last step

        Returns:
            (action_result, possibilities_or_None)
        """
        self._step_count += 1

        if is_last_step:
            t0 = time.monotonic()
            result = await action_fn()
            self._total_action_time += time.monotonic() - t0
            return result, None

        # ── DECISION POINT: action + precompute start together ─────
        t0 = time.monotonic()
        result, possibilities = await asyncio.gather(
            action_fn(),
            self.precompute_possibilities(precompute_fn, step_context),
        )
        elapsed = time.monotonic() - t0
        self._total_action_time += elapsed

        return result, possibilities

    # Keep old name as alias for SpeculativeEngine compatibility
    execute_with_precompute = decide_and_act

    def dispatch(self, result: Any, possibilities: dict[str, dict] | None
                 ) -> dict | None:
        """Match action result against pre-computed possibilities.

        Returns the matching next action if found, None if no match
        (caller falls back to a fresh LLM think).
        """
        if not possibilities:
            return None

        result_key = str(result)
        if result_key in possibilities:
            self._instant_dispatches += 1
            self._events.append({
                "event": "instant_dispatch",
                "step": self._step_count,
                "result_key": result_key[:100],
                "ts": time.monotonic(),
            })
            log.debug("PossSpec: instant dispatch step %d (key=%s)",
                      self._step_count, result_key[:50])
            return possibilities[result_key]

        # No match — caller must think fresh
        self._fallback_thinks += 1
        self._events.append({
            "event": "fallback_think",
            "step": self._step_count,
            "result_key": result_key[:100],
            "available_keys": list(possibilities.keys())[:5],
            "ts": time.monotonic(),
        })
        log.debug("PossSpec: fallback think step %d (key=%s not in %s)",
                  self._step_count, result_key[:50],
                  list(possibilities.keys())[:3])
        return None

    async def record_think(self, duration: float) -> None:
        """Record a fallback think time (when dispatch missed)."""
        self._total_think_time += duration

    # ── Queries ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self._started_at
        return {
            "dial": self.dial,
            "steps": self._step_count,
            "instant_dispatches": self._instant_dispatches,
            "fallback_thinks": self._fallback_thinks,
            "hit_rate": (self._instant_dispatches /
                         max(1, self._instant_dispatches + self._fallback_thinks)),
            "think_time": self._total_think_time,
            "action_time": self._total_action_time,
            "precompute_time": self._total_precompute_time,
            "elapsed": elapsed,
        }

    def get_summary(self) -> str:
        s = self.stats
        return (f"PossSpec dial={s['dial']}: {s['steps']} steps, "
                f"{s['instant_dispatches']} instant / "
                f"{s['fallback_thinks']} fallback "
                f"({s['hit_rate']:.0%} hit), "
                f"think={s['think_time']:.1f}s, "
                f"elapsed={s['elapsed']:.1f}s")

    def get_timeline(self) -> str:
        """ASCII timeline of events for debugging."""
        lines = [f"PossSpec Timeline (dial={self.dial}):"]
        for e in self._events:
            t = e["ts"] - self._started_at
            if e["event"] == "instant_dispatch":
                lines.append(f"  {t:6.2f}s ⚡ INSTANT step {e['step']}")
            elif e["event"] == "fallback_think":
                lines.append(f"  {t:6.2f}s 🐌 FALLBACK step {e['step']} "
                             f"(wanted {e['result_key'][:30]})")
            elif e["event"] == "precompute":
                lines.append(f"  {t:6.2f}s 🔮 PRECOMPUTE step {e['step']} "
                             f"({e['num_possibilities']} poss, {e['elapsed']:.2f}s)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SpeculativeEngine — unified probability batching + possibility pipeline
# ═══════════════════════════════════════════════════════════════════════════

class SpeculativeEngine:
    """Combines SpeculativeRouter's probability batching with
    PossibilitySpeculator's pre-compute-and-hold pipeline.

    Core principle: ALWAYS pre-compute possibilities from the previous
    decision point. Pre-computation starts the moment you have something
    solid — the result of the last step IS that solid ground.

    On TOP of that, high-confidence batching fires immediately:

        Every step:
            → Pre-compute possibilities (ALWAYS, from previous decision)
            → These run in background during current action

        If confidence >= threshold (we KNOW what's next):
            → ALSO batch-fire the next action immediately
            → If the batch was WRONG, possibilities are still ready
            → Example: delegated task to Hermes → 95% sure it'll succeed,
              pre-compute covers the 5% where agent fails

        If confidence < threshold (uncertain):
            → Only possibilities — wait for result, instant-dispatch match
            → Example: called API → could be success/error/rate-limit

    This means high-confidence steps get BOTH speed AND safety:
        - The batch fires with zero delay (speed)
        - Pre-computed possibilities catch batch misses (safety)
        - Low-confidence steps still get zero-gap dispatch

    The confidence_threshold is driven by the dial:
        Dial 1:  threshold=0.9 (rarely batches, mostly possibilities)
        Dial 5:  threshold=0.7 (batches when fairly confident)
        Dial 10: threshold=0.5 (batches aggressively, possibilities backup)
    """

    DIAL_PROFILES: ClassVar[dict[int, dict[str, Any]]] = {
        1:  {"confidence_threshold": 0.90, "max_batch": 1,  "max_possibilities": 2},
        2:  {"confidence_threshold": 0.85, "max_batch": 2,  "max_possibilities": 3},
        3:  {"confidence_threshold": 0.80, "max_batch": 2,  "max_possibilities": 3},
        4:  {"confidence_threshold": 0.75, "max_batch": 3,  "max_possibilities": 4},
        5:  {"confidence_threshold": 0.70, "max_batch": 3,  "max_possibilities": 4},
        6:  {"confidence_threshold": 0.65, "max_batch": 4,  "max_possibilities": 5},
        7:  {"confidence_threshold": 0.60, "max_batch": 5,  "max_possibilities": 6},
        8:  {"confidence_threshold": 0.55, "max_batch": 6,  "max_possibilities": 8},
        9:  {"confidence_threshold": 0.52, "max_batch": 8,  "max_possibilities": 10},
        10: {"confidence_threshold": 0.50, "max_batch": 10, "max_possibilities": 12},
    }

    def __init__(self, thought_stream: ThoughtStream | None = None,
                 dial: int = 5,
                 confidence_threshold: float | None = None):
        self.thought_stream = thought_stream or ThoughtStream()
        self.dial = max(1, min(10, dial))
        profile = self.DIAL_PROFILES[self.dial]

        self.confidence_threshold = (confidence_threshold
                                     if confidence_threshold is not None
                                     else profile["confidence_threshold"])
        self.max_batch = profile["max_batch"]
        self.max_possibilities = profile["max_possibilities"]

        # Sub-engines
        self.router = SpeculativeRouter(
            thought_stream=self.thought_stream,
            speed=self.dial,
        )
        self.speculator = PossibilitySpeculator(
            thought_stream=self.thought_stream,
            dial=self.dial,
        )

        # Stats
        self._lock = asyncio.Lock()
        self._step_count = 0
        self._high_path_count = 0    # batched (confident)
        self._low_path_count = 0     # pre-computed (uncertain)
        self._events: list[dict[str, Any]] = []
        self._started_at = time.monotonic()

        log.info("SpecEngine: dial=%d (threshold=%.0f%%, max_batch=%d, max_poss=%d)",
                 self.dial, self.confidence_threshold * 100,
                 self.max_batch, self.max_possibilities)

    @classmethod
    def from_dial(cls, dial: int,
                  thought_stream: ThoughtStream | None = None) -> "SpeculativeEngine":
        """Create from dial value alone."""
        return cls(thought_stream=thought_stream, dial=dial)

    async def step(
        self,
        confidence: float,
        action_fn,
        precompute_fn=None,
        batch_actions: list[dict[str, Any]] | None = None,
        step_context: dict[str, Any] | None = None,
        is_last_step: bool = False,
    ) -> dict[str, Any]:
        """Execute one step — ALWAYS pre-compute, batch on top if confident.

        Pre-computation runs every step (from the previous decision).
        High confidence adds batch-firing ON TOP, with possibilities as
        a safety net if the batch was wrong.

        Args:
            confidence: 0.0–1.0, how sure we are about the next action.
            action_fn: async callable that executes the current action.
            precompute_fn: async callable(ctx) → {outcome_key: action_dict}.
                           ALWAYS runs (unless is_last_step).
            batch_actions: list of actions to batch-fire immediately when
                          confidence is high. Layered on top of possibilities.
            step_context: context dict passed to precompute_fn.
            is_last_step: if True, skip pre-computation.

        Returns:
            {
                "result": action result,
                "path": "high+precompute" | "precompute",
                "next_action": dict | None,
                "batch_fired": bool,
                "possibilities": dict | None,
                "confidence": float,
            }
        """
        async with self._lock:
            self._step_count += 1
            step_num = self._step_count

        is_high = confidence >= self.confidence_threshold
        if is_high:
            self._high_path_count += 1
        else:
            self._low_path_count += 1

        # ── ALWAYS pre-compute possibilities during action ─────────
        # This is the foundation — every step gets possibilities ready
        result, possibilities = await self.speculator.execute_with_precompute(
            action_fn,
            precompute_fn or (lambda ctx: {}),
            step_context=step_context,
            is_last_step=is_last_step,
        )

        next_action = None
        batch_fired = False

        if is_high and batch_actions and not is_last_step:
            # ── HIGH CONFIDENCE: batch fires ON TOP ────────────────
            # We already pre-computed possibilities (safety net).
            # Now also use the batch action — it's what we're confident about.
            batch_size = min(len(batch_actions), self.max_batch)
            batch_to_fire = batch_actions[:batch_size]
            next_action = batch_to_fire[0] if batch_to_fire else None
            batch_fired = True

            # Check: did the result match our confident prediction?
            # If not, override the batch with the matching possibility
            if possibilities and not is_last_step:
                dispatched = self.speculator.dispatch(result, possibilities)
                if dispatched is not None and next_action is not None:
                    # Result matched a possibility — check if it agrees
                    # with our batch. If the batch was wrong, use the
                    # possibility instead (the safety net catches it).
                    result_key = str(result)
                    batch_key = next_action.get("_expected_outcome", "")
                    if batch_key and result_key != batch_key:
                        # Batch was WRONG — safety net kicks in
                        next_action = dispatched
                        batch_fired = False
                        log.info("SpecEngine: batch miss step %d — "
                                 "safety net caught it (conf=%.0f%%)",
                                 step_num, confidence * 100)

            self._events.append({
                "event": "high_plus_precompute",
                "step": step_num,
                "confidence": confidence,
                "batch_fired": batch_fired,
                "batch_size": batch_size,
                "num_possibilities": len(possibilities) if possibilities else 0,
                "ts": time.monotonic(),
            })
            log.debug("SpecEngine: HIGH+PRECOMPUTE step %d (conf=%.0f%%, "
                      "batch=%s, %d poss ready)",
                      step_num, confidence * 100, batch_fired,
                      len(possibilities) if possibilities else 0)

            return {
                "result": result,
                "path": "high+precompute",
                "next_action": next_action,
                "batch_remaining": batch_to_fire[1:] if batch_fired and len(batch_to_fire) > 1 else [],
                "batch_fired": batch_fired,
                "possibilities": possibilities,
                "confidence": confidence,
            }

        else:
            # ── PRECOMPUTE ONLY (low confidence or no batch) ───────
            # Possibilities are our only play — instant-dispatch on match
            if possibilities and not is_last_step:
                next_action = self.speculator.dispatch(result, possibilities)

            self._events.append({
                "event": "precompute_only",
                "step": step_num,
                "confidence": confidence,
                "dispatched": next_action is not None,
                "num_possibilities": len(possibilities) if possibilities else 0,
                "ts": time.monotonic(),
            })
            log.debug("SpecEngine: PRECOMPUTE step %d (conf=%.0f%%, dispatched=%s)",
                      step_num, confidence * 100, next_action is not None)

            return {
                "result": result,
                "path": "precompute",
                "next_action": next_action,
                "batch_fired": False,
                "possibilities": possibilities,
                "confidence": confidence,
            }

    # ── Queries ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        spec_stats = self.speculator.stats
        elapsed = time.monotonic() - self._started_at
        return {
            "dial": self.dial,
            "confidence_threshold": self.confidence_threshold,
            "steps": self._step_count,
            "high_path": self._high_path_count,
            "low_path": self._low_path_count,
            "instant_dispatches": spec_stats["instant_dispatches"],
            "fallback_thinks": spec_stats["fallback_thinks"],
            "hit_rate": spec_stats["hit_rate"],
            "elapsed": elapsed,
        }

    def get_summary(self) -> str:
        s = self.stats
        return (f"SpecEngine dial={s['dial']}: {s['steps']} steps, "
                f"{s['high_path']} high-conf / {s['low_path']} low-conf, "
                f"{s['instant_dispatches']} instant dispatches "
                f"({s['hit_rate']:.0%} hit), "
                f"elapsed={s['elapsed']:.1f}s")

    def get_timeline(self) -> str:
        """ASCII timeline of engine decisions."""
        lines = [f"SpecEngine Timeline (dial={self.dial}, "
                 f"threshold={self.confidence_threshold:.0%}):"]
        for e in self._events:
            t = e["ts"] - self._started_at
            conf = e["confidence"]
            if e["event"] == "high_plus_precompute":
                safety = "" if e["batch_fired"] else " → SAFETY NET"
                lines.append(f"  {t:6.2f}s 🚀 HIGH+PRE step {e['step']} "
                             f"(conf={conf:.0%}, batch={e['batch_size']}, "
                             f"{e['num_possibilities']} poss{safety})")
            elif e["event"] == "precompute_only":
                marker = "⚡" if e["dispatched"] else "🔮"
                lines.append(f"  {t:6.2f}s {marker} PRE  step {e['step']} "
                             f"(conf={conf:.0%}, "
                             f"{e['num_possibilities']} poss, "
                             f"{'instant' if e['dispatched'] else 'fallback'})")
        return "\n".join(lines)


class SharedMind:
    """Blackboard pattern — cross-tab awareness for concurrent browser lanes.

    Every TabLane reads from and writes to this shared state. Before each
    action, a lane sees what ALL other lanes are doing. This is what makes
    Persona's concurrency intelligent rather than blind.

    The SharedMind is the answer to Google's question: "Should sub-agents
    pause for master mind authorization or freely pivot asynchronously?"
    Answer: neither. They share awareness and coordinate implicitly.

    Thread-safe via asyncio (single event loop, no threading needed).

    Now includes a ThoughtStream for live inference mirroring — each tab's
    reasoning tokens are visible to all other tabs in real time.
    """

    def __init__(self):
        self.tabs: dict[str, dict[str, Any]] = {}
        # tab_id -> {url, title, status, last_result, error, started_at, ...}
        self.events: list[dict[str, Any]] = []
        # Ordered log: [{tab, event, data, ts}, ...]
        self._lock = asyncio.Lock()
        self.thought_stream = ThoughtStream()
        # Live inference mirror — shared across all tabs

    async def register_tab(self, tab_id: str, url: str, task: str) -> None:
        async with self._lock:
            self.tabs[tab_id] = {
                "url": url,
                "task": task,
                "status": "created",
                "title": "",
                "last_result": None,
                "error": None,
                "started_at": time.monotonic(),
            }
            self.events.append({
                "tab": tab_id, "event": "created",
                "data": {"url": url, "task": task},
                "ts": time.monotonic(),
            })

    async def update_tab(self, tab_id: str, **updates) -> None:
        async with self._lock:
            if tab_id in self.tabs:
                self.tabs[tab_id].update(updates)
                self.events.append({
                    "tab": tab_id, "event": "update",
                    "data": updates,
                    "ts": time.monotonic(),
                })

    async def get_cross_tab_context(self, exclude_tab: str | None = None) -> str:
        """What every lane reads before acting — awareness of ALL tabs.

        Now includes live thought stream — each tab sees what other tabs
        are currently thinking/intending, not just their last action.
        """
        async with self._lock:
            lines = []
            for tid, info in self.tabs.items():
                if tid == exclude_tab:
                    continue
                status = info.get("status", "unknown")
                url = info.get("url", "?")
                title = info.get("title", "")
                last = info.get("last_result")
                err = info.get("error")
                line = f"  [{tid}] {status}: {url}"
                if title:
                    line += f" — '{title}'"
                if last:
                    line += f" → {str(last)[:100]}"
                if err:
                    line += f" ⚠ {str(err)[:80]}"
                lines.append(line)
            if not lines:
                return "No other tabs active."

        # Append live thought stream (outside lock — ThoughtStream has its own)
        thoughts = await self.thought_stream.get_live_thoughts(
            exclude_agent=exclude_tab, max_chars=500
        )

        result = "CROSS-TAB STATE:\n" + "\n".join(lines)
        if thoughts:
            result += "\n\n" + thoughts
        return result

    async def get_summary(self) -> str:
        """Full summary for aggregated reporting."""
        async with self._lock:
            parts = []
            for tid, info in self.tabs.items():
                status = info.get("status", "?")
                elapsed = time.monotonic() - info.get("started_at", time.monotonic())
                parts.append(
                    f"[{tid}] {status} ({elapsed:.1f}s): "
                    f"{info.get('title') or info.get('url', '?')} "
                    f"→ {str(info.get('last_result', ''))[:200]}"
                )
            return "\n".join(parts)


class TabLane:
    """Thin wrapper: own focus pointer + Page handle, shared CDP connection.

    Each TabLane owns:
    - A target_id (which tab it controls)
    - A Page object (browser-use's own, with its own CDP session)
    - A reference to the SharedMind

    It shares:
    - The underlying BrowserSession (and its CDP connection)
    - The SharedMind with all other lanes

    This is the proven approach from Test 4: no cross-contamination,
    no focus conflict, full use of browser-use's Page infrastructure
    (DOM parsing, JS evaluation, navigation, screenshots).
    """

    def __init__(self, tab_id: str, session: Any, target_id: str,
                 mind: SharedMind):
        self.tab_id = tab_id
        self.session = session  # shared BrowserSession
        self.target_id = target_id  # own focus pointer
        self.mind = mind
        self._page = None  # lazily created Page

    async def _get_page(self):
        if self._page is None:
            from browser_use.actor.page import Page
            self._page = Page(self.session, self.target_id)
        return self._page

    async def navigate(self, url: str) -> None:
        page = await self._get_page()
        await page.goto(url)
        await self.mind.update_tab(self.tab_id, url=url, status="navigated")

    async def evaluate(self, js: str) -> Any:
        page = await self._get_page()
        return await page.evaluate(js)

    async def screenshot(self) -> Any:
        page = await self._get_page()
        return await page.screenshot()

    async def get_title(self) -> str:
        title = await self.evaluate("() => document.title")
        await self.mind.update_tab(self.tab_id, title=title)
        return title

    async def get_text(self, selector: str = "body") -> str:
        return await self.evaluate(
            f"() => document.querySelector('{selector}')?.textContent || ''"
        )

    async def get_html(self, selector: str = "body") -> str:
        return await self.evaluate(
            f"() => document.querySelector('{selector}')?.innerHTML || ''"
        )

    async def click(self, selector: str) -> None:
        await self.evaluate(f"() => document.querySelector('{selector}')?.click()")

    async def type_text(self, selector: str, text: str) -> None:
        escaped = text.replace("'", "\\'").replace("\n", "\\n")
        await self.evaluate(
            f"() => {{ const el = document.querySelector('{selector}'); "
            f"if (el) {{ el.value = '{escaped}'; "
            f"el.dispatchEvent(new Event('input', {{bubbles: true}})); }} }}"
        )

    async def wait_for(self, selector: str, timeout_ms: int = 5000) -> bool:
        return await self.evaluate(
            f"() => new Promise((resolve) => {{ "
            f"const el = document.querySelector('{selector}'); "
            f"if (el) return resolve(true); "
            f"const obs = new MutationObserver(() => {{ "
            f"  if (document.querySelector('{selector}')) {{ obs.disconnect(); resolve(true); }} "
            f"}}); "
            f"obs.observe(document.body, {{childList: true, subtree: true}}); "
            f"setTimeout(() => {{ obs.disconnect(); resolve(false); }}, {timeout_ms}); "
            f"}})"
        )

    async def cross_tab_context(self) -> str:
        """What this lane knows about all other lanes."""
        return await self.mind.get_cross_tab_context(exclude_tab=self.tab_id)


# ═══════════════════════════════════════════════════════════════════════════
# Same-Page Swarm — multiple hands on ONE DOM, shared mind coordination
# ═══════════════════════════════════════════════════════════════════════════


class SwarmMind:
    """Real-time coordinator for multiple hands on ONE page.

    Unlike SharedMind (cross-tab awareness), SwarmMind coordinates actions
    on the SAME DOM. It handles:

    1. Action Registry — what's in-flight, who's doing what
    2. Conflict Detection — prevents two hands from touching the same element
    3. Halt Signal — when page reload/navigation detected, all hands stop
    4. DOM Mutation Awareness — one hand's changes are known to all others
    5. Dynamic Task Rebalancing — hands pick up unfinished work from others
    6. Micro-Delay Orchestration — uses execution gaps to interleave actions
    7. Live Thought Sharing — each hand's reasoning is visible to all others
       in real time via ThoughtStream (simultaneous shared consciousness)

    This is the "unified swarm intelligence" — one mind, multiple bodies,
    all operating on a single page simultaneously.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        # ── Action registry ─────────────────────────────────────────
        self._inflight: dict[str, dict[str, Any]] = {}
        # hand_id -> {action, selector, status, started_at}
        self._completed: list[dict[str, Any]] = []
        # [{hand, action, selector, result, elapsed}, ...]
        self._claimed_selectors: set[str] = set()
        # Selectors currently being acted on (prevent collision)

        # ── Task queue ──────────────────────────────────────────────
        self._task_queue: list[dict[str, Any]] = []
        # [{action, selector, data, priority}, ...] ordered by priority
        self._task_idx: int = 0
        # Next task to claim

        # ── Halt signal ─────────────────────────────────────────────
        self._halted = False
        self._halt_reason: str = ""

        # ── DOM state ───────────────────────────────────────────────
        self._dom_version: int = 0
        self._dom_mutations: list[dict[str, Any]] = []
        # [{hand, type, selector, ts}, ...] — mutations caused by hands

        # ── Coordination events ─────────────────────────────────────
        self._events: list[dict[str, Any]] = []

        # ── Live thought sharing ───────────────────────────────────
        self.thought_stream = ThoughtStream()
        # Each hand's reasoning tokens are broadcast to all other hands
        # in real time. Before hand-1 acts, it sees:
        #   [hand-0] INTENT: click '#submit' — form is complete
        # This is the "simultaneous shared consciousness" from the pitch.

        # ── Reactive stagger ───────────────────────────────────────
        self._dispatch_event = asyncio.Event()
        # Fires when ANY hand dispatches a CDP command — other hands
        # that are sleeping wake up and claim their next task.
        self._dispatch_count: int = 0

    # ── Task management ────────────────────────────────────────────

    def load_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Load a batch of tasks for the swarm to execute.

        Each task: {action: "click"|"type"|"extract"|"wait"|"scroll",
                    selector: "CSS selector",
                    data: optional data (text to type, etc),
                    priority: int (lower = higher priority)}
        """
        self._task_queue = sorted(tasks, key=lambda t: t.get("priority", 99))
        self._task_idx = 0

    async def claim_next_task(self, hand_id: str) -> dict[str, Any] | None:
        """A hand claims the next available non-conflicting task.

        Returns None if all tasks are done or all remaining tasks conflict
        with in-flight actions. This is the dynamic pivoting — if task A
        is being handled by hand-0, hand-1 automatically skips to task B.
        """
        async with self._lock:
            if self._halted:
                return None

            for i in range(self._task_idx, len(self._task_queue)):
                task = self._task_queue[i]
                selector = task.get("selector", "")

                # Skip if another hand is already acting on this selector
                if selector and selector in self._claimed_selectors:
                    continue

                # Claim it
                self._task_queue.pop(i)
                if selector:
                    self._claimed_selectors.add(selector)
                self._inflight[hand_id] = {
                    "action": task.get("action"),
                    "selector": selector,
                    "status": "executing",
                    "started_at": time.monotonic(),
                    "task": task,
                }
                self._events.append({
                    "hand": hand_id, "event": "claimed",
                    "task": task, "ts": time.monotonic(),
                })
                return task

            return None  # All done or all conflicting

    async def complete_task(self, hand_id: str, result: Any = None,
                            error: str | None = None) -> None:
        """Hand reports task completion. Releases the selector claim."""
        async with self._lock:
            info = self._inflight.pop(hand_id, None)
            if info:
                selector = info.get("selector", "")
                if selector:
                    self._claimed_selectors.discard(selector)
                elapsed = time.monotonic() - info.get("started_at", time.monotonic())
                self._completed.append({
                    "hand": hand_id,
                    "action": info.get("action"),
                    "selector": selector,
                    "result": result,
                    "error": error,
                    "elapsed": elapsed,
                })
                self._events.append({
                    "hand": hand_id, "event": "completed",
                    "result": str(result)[:100] if result else None,
                    "error": error,
                    "elapsed": elapsed,
                    "ts": time.monotonic(),
                })

    # ── Halt signal (reload/navigation detected) ───────────────────

    async def halt_all(self, reason: str) -> None:
        """Signal all hands to stop immediately.

        Called when a page reload, navigation, or destructive mutation
        is detected. All hands check is_halted() before each action.
        """
        async with self._lock:
            self._halted = True
            self._halt_reason = reason
            self._events.append({
                "hand": "mind", "event": "HALT",
                "reason": reason, "ts": time.monotonic(),
            })
            log.warning("SwarmMind: HALT — %s", reason)

    async def resume(self) -> None:
        """Resume after halt (e.g., page reload completed)."""
        async with self._lock:
            self._halted = False
            self._halt_reason = ""
            self._claimed_selectors.clear()  # selectors may have changed
            self._events.append({
                "hand": "mind", "event": "RESUME",
                "ts": time.monotonic(),
            })

    def is_halted(self) -> bool:
        return self._halted

    # ── DOM mutation tracking ──────────────────────────────────────

    async def record_mutation(self, hand_id: str, mutation_type: str,
                               selector: str) -> None:
        """Record a DOM mutation caused by a hand's action.

        Other hands read this to understand index shifts and layout
        changes they didn't cause.
        """
        async with self._lock:
            self._dom_version += 1
            self._dom_mutations.append({
                "hand": hand_id,
                "type": mutation_type,  # "click", "type", "remove", "add"
                "selector": selector,
                "dom_version": self._dom_version,
                "ts": time.monotonic(),
            })

    async def get_mutations_since(self, since_version: int) -> list[dict[str, Any]]:
        """Get DOM mutations since a given version.

        A hand calls this before acting to see what changed since its
        last action. This is how hand-1 knows hand-0 clicked a button
        that shifted the DOM layout.
        """
        async with self._lock:
            return [m for m in self._dom_mutations
                    if m["dom_version"] > since_version]

    # ── Situational awareness ──────────────────────────────────────

    async def get_hand_context(self, exclude_hand: str | None = None) -> str:
        """What a hand sees about all other hands before acting."""
        async with self._lock:
            lines = []
            if self._halted:
                lines.append(f"⚠ HALTED: {self._halt_reason}")

            for hid, info in self._inflight.items():
                if hid == exclude_hand:
                    continue
                action = info.get("action", "?")
                selector = info.get("selector", "?")
                elapsed = time.monotonic() - info.get("started_at", time.monotonic())
                lines.append(f"  [{hid}] {action} on '{selector}' ({elapsed:.2f}s)")

            done_count = len(self._completed)
            remaining = len(self._task_queue) - self._task_idx
            lines.append(f"  Progress: {done_count} done, {remaining} remaining")

            if self._dom_mutations:
                recent = self._dom_mutations[-3:]  # last 3 mutations
                for m in recent:
                    lines.append(
                        f"  DOM[v{m['dom_version']}]: {m['hand']} {m['type']} "
                        f"'{m['selector']}'"
                    )

            return "\n".join(lines) if lines else "Swarm idle."

    async def get_results(self) -> list[dict[str, Any]]:
        """Get all completed task results."""
        async with self._lock:
            return list(self._completed)


class SwarmHand:
    """One hand of the swarm — operates on a SHARED page via CDP.

    All hands share:
    - The same Page object (same tab, same DOM)
    - The SwarmMind for coordination

    Each hand:
    - Claims tasks from the SwarmMind's queue
    - Checks for halt signals before each action
    - Records DOM mutations it causes
    - Tracks the DOM version it last saw
    - Interleaves with other hands automatically

    The CDP evaluate() calls are atomic at the JS level — two concurrent
    evaluates on the same page both run, they just serialize in V8.
    The SwarmMind prevents logical conflicts (two hands clicking the same
    button) while allowing physical concurrency (two hands acting on
    different parts of the DOM simultaneously).
    """

    def __init__(self, hand_id: str, page: Any, mind: SwarmMind):
        self.hand_id = hand_id
        self.page = page  # shared Page object — same tab
        self.mind = mind
        self._dom_version = 0  # last DOM version this hand saw
        self._action_count = 0

    async def _check_halt(self) -> bool:
        """Check if swarm is halted. Returns True if halted."""
        if self.mind.is_halted():
            log.info("  %s: halted — %s", self.hand_id, self.mind._halt_reason)
            return True
        return False

    async def _pre_action(self) -> bool:
        """Pre-action check: halt signal + DOM mutations from other hands.

        Returns False if halted (hand should stop).
        """
        if await self._check_halt():
            return False

        # Check what other hands changed since our last action
        mutations = await self.mind.get_mutations_since(self._dom_version)
        if mutations:
            self._dom_version = mutations[-1]["dom_version"]
            for m in mutations:
                if m["hand"] != self.hand_id:
                    log.debug("  %s: saw mutation from %s: %s on '%s'",
                              self.hand_id, m["hand"], m["type"], m["selector"])
        return True

    async def execute_action(self, task: dict[str, Any]) -> Any:
        """Execute one task action on the shared page.

        Handles: click, type, extract, wait, scroll, evaluate.
        Records DOM mutations caused by the action.
        Broadcasts intent via ThoughtStream before acting.
        Signals dispatch_event so staggered hands wake up.
        """
        if not await self._pre_action():
            return None

        action = task.get("action", "extract")
        selector = task.get("selector", "body")
        data = task.get("data", "")
        self._action_count += 1

        # ── Broadcast intent BEFORE acting ─────────────────────────
        # Other hands see this mid-inference: "[hand-0] INTENT: click '#submit'"
        await self.mind.thought_stream.push_action_intent(
            self.hand_id, action, selector,
            reasoning=f"task priority {task.get('priority', '?')}"
        )

        try:
            if action == "click":
                # Check element exists before clicking
                exists = await self.page.evaluate(
                    f"() => !!document.querySelector('{selector}')"
                )
                if not exists:
                    return f"Element '{selector}' not found"

                await self.page.evaluate(
                    f"() => document.querySelector('{selector}')?.click()"
                )
                await self.mind.record_mutation(self.hand_id, "click", selector)
                return f"Clicked '{selector}'"

            elif action == "type":
                escaped = data.replace("'", "\\'").replace("\n", "\\n")
                await self.page.evaluate(
                    f"() => {{ const el = document.querySelector('{selector}'); "
                    f"if (el) {{ el.value = '{escaped}'; "
                    f"el.dispatchEvent(new Event('input', {{bubbles: true}})); "
                    f"el.dispatchEvent(new Event('change', {{bubbles: true}})); }} }}"
                )
                await self.mind.record_mutation(self.hand_id, "type", selector)
                return f"Typed into '{selector}'"

            elif action == "extract":
                js = data if data else f"() => document.querySelector('{selector}')?.textContent || ''"
                result = await self.page.evaluate(js)
                return result

            elif action == "wait":
                timeout = int(data) if data else 5000
                result = await self.page.evaluate(
                    f"() => new Promise((resolve) => {{ "
                    f"const el = document.querySelector('{selector}'); "
                    f"if (el) return resolve(true); "
                    f"const obs = new MutationObserver(() => {{ "
                    f"  if (document.querySelector('{selector}')) {{ obs.disconnect(); resolve(true); }} "
                    f"}}); "
                    f"obs.observe(document.body, {{childList: true, subtree: true}}); "
                    f"setTimeout(() => {{ obs.disconnect(); resolve(false); }}, {timeout}); "
                    f"}})"
                )
                return f"Wait for '{selector}': {'found' if result else 'timeout'}"

            elif action == "scroll":
                await self.page.evaluate(
                    f"() => document.querySelector('{selector}')?.scrollIntoView({{behavior: 'smooth'}})"
                )
                return f"Scrolled to '{selector}'"

            elif action == "evaluate":
                result = await self.page.evaluate(data)
                return result

            elif action == "check_navigation":
                # Detect if page is navigating/reloading
                state = await self.page.evaluate(
                    "() => document.readyState"
                )
                if state != "complete":
                    await self.mind.halt_all(
                        f"Page navigating (readyState={state}), "
                        f"triggered by {self.hand_id}"
                    )
                    return f"Navigation detected: {state}"
                return f"Page stable: {state}"

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            err_str = str(e)
            # Detect navigation/reload errors
            if any(kw in err_str.lower() for kw in
                   ["navigation", "detached", "context was destroyed",
                    "execution context", "target closed"]):
                await self.mind.halt_all(f"Page destroyed: {err_str[:100]}")
            await self.mind.thought_stream.push_tokens(
                self.hand_id, f"ERROR: {err_str[:100]}"
            )
            return f"ERROR: {err_str[:200]}"

    async def run_loop(self) -> list[dict[str, Any]]:
        """Main loop: claim tasks, execute, repeat until queue empty or halted.

        This is one hand's autonomous loop. It runs concurrently with
        other hands, all sharing the same page and SwarmMind.

        Before each claim, the hand reads other hands' live thoughts
        from the ThoughtStream (the Inter-Agent Attention Layer).
        """
        results = []
        while True:
            if await self._check_halt():
                break

            # ── Read other hands' live thoughts before deciding ────
            live = await self.mind.thought_stream.get_live_thoughts(
                exclude_agent=self.hand_id, max_chars=500
            )
            if live:
                log.debug("  %s: sees live thoughts:\n%s", self.hand_id, live)

            task = await self.mind.claim_next_task(self.hand_id)
            if task is None:
                break  # No more tasks (or all remaining conflict)

            log.info("  %s: claimed %s on '%s'",
                     self.hand_id, task.get("action"), task.get("selector", "?"))

            # Execute (broadcasts intent + signals dispatch_event internally)
            result = await self.execute_action(task)
            results.append({"task": task, "result": result})

            # Report completion + complete thought block
            error = str(result) if result and str(result).startswith("ERROR") else None
            await self.mind.complete_task(self.hand_id, result=result, error=error)
            await self.mind.thought_stream.complete_thought(self.hand_id)

            # Signal other staggered hands to wake up
            self.mind._dispatch_event.set()
            self.mind._dispatch_count += 1
            await asyncio.sleep(0)
            self.mind._dispatch_event.clear()

        log.info("  %s: done — %d actions executed", self.hand_id, self._action_count)
        await self.mind.thought_stream.clear_agent(self.hand_id)
        return results


class PersonaChatModel:
    """Persona as a browser-use LLM — implements BaseChatModel protocol.

    This is the product. When browser-use calls ainvoke(), Persona
    intercepts and creates concurrency FROM INSIDE the LLM call.

    Architecture (thin wrapper + shared mind):
        ONE BrowserSession, ONE browser process, N tabs.
        Each tab gets a TabLane: own Page object with its own CDP session,
        own focus pointer (target_id), shared awareness via SharedMind.

        The concurrency uses browser-use's own Page class — CDP commands
        are coupled to target_id, not visual focus. Each Page.goto(),
        Page.evaluate(), Page.screenshot() operates independently.

        SharedMind (blackboard pattern) gives every tab awareness of what
        ALL other tabs are doing. This is coordinated concurrency, not
        blind tab driving.

    What this preserves from browser-use:
        - Page class: DOM parsing, JS evaluation, navigation, screenshots
        - CDP session management: each tab has its own session_id
        - Error recovery: Page handles CDP errors per-tab
        - The single BrowserSession's lifecycle management

    What Persona adds:
        - Concurrent tab creation from the LLM position
        - SharedMind for cross-tab coordination
        - Aggregated results returned as one AgentOutput
        - The host Agent never knows concurrency happened

    Usage:
        from browser_use import Agent, BrowserSession
        session = BrowserSession(browser_profile=profile)
        persona = PersonaChatModel(api_key="...", browser_session=session)
        persona.set_concurrent_tasks([
            {"url": "https://...", "extract": "() => document.title"},
            {"url": "https://...", "extract": "() => ..."},
        ])
        agent = Agent(task="...", llm=persona, browser_session=session)
        # SAME session. Persona creates tabs inside it.
    """

    def __init__(
        self,
        api_key: str = "",
        backing_model: str = "gpt-4o-mini",
        base_url: str | None = None,
        headless: bool = True,
        max_concurrent: int = 3,
        browser_session: Any = None,
    ):
        self.api_key = api_key
        self.backing_model = backing_model
        self.base_url = base_url
        self.headless = headless
        self.max_concurrent = max_concurrent
        self.model = "persona-concurrent"
        self._verified_api_keys = False
        self._sub_tasks: list[str | dict[str, str]] = []
        self._results: dict[str, str] = {}
        # Shared session — set externally or created on first use
        self.browser_session = browser_session
        # Shared mind — created per concurrent batch
        self._shared_mind: SharedMind | None = None

    @property
    def provider(self) -> str:
        return 'persona'

    @property
    def name(self) -> str:
        return f'persona-concurrent-{self.max_concurrent}'

    @property
    def model_name(self) -> str:
        return self.model

    def set_concurrent_tasks(self, tasks: list[str | dict[str, str]]) -> None:
        """Set the sub-tasks Persona will run concurrently.

        Each task can be:
        - A string (URL to navigate and extract title from)
        - A dict: {"url": "...", "extract": "() => JS expression"}

        In production, Persona's Mind analyzes the conversation and
        decides how to split. For the proof, we pre-set them.
        """
        self._sub_tasks = list(tasks)

    def _get_backing_llm(self) -> Any:
        """Create the real LLM for passthrough mode."""
        try:
            from browser_use.llm.openai.chat import ChatOpenAI as BUChatOpenAI
            params: dict[str, Any] = {"model": self.backing_model}
            if self.api_key:
                params["api_key"] = self.api_key
            if self.base_url:
                params["base_url"] = self.base_url
            return BUChatOpenAI(**params)
        except ImportError:
            pass
        # Fallback: try langchain
        try:
            from langchain_openai import ChatOpenAI
            params = {"model": self.backing_model}
            if self.api_key:
                params["api_key"] = self.api_key
            if self.base_url:
                params["base_url"] = self.base_url
            return ChatOpenAI(**params)
        except ImportError:
            raise ImportError(
                "No LLM provider found. Install browser-use or langchain-openai."
            )

    async def ainvoke(self, messages, output_format=None, **kwargs):
        """The interception point — browser-use calls this as its LLM.

        If concurrent tasks are set, Persona creates N tabs in the SAME
        browser and drives them concurrently via Page objects.
        Otherwise, forwards to the backing LLM.
        """
        from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

        if self._sub_tasks:
            # ── CONCURRENT MODE ──────────────────────────────────────
            log.info("PersonaChatModel.ainvoke: splitting into %d concurrent R-lanes "
                     "(ONE browser, ONE session, shared mind)",
                     len(self._sub_tasks))
            aggregated = await self._run_concurrent(self._sub_tasks)
            self._sub_tasks = []  # consume

            if output_format is not None:
                result_text = "CONCURRENT RESULTS (one browser, shared mind):\n"
                for tid, res in aggregated.items():
                    result_text += f"\n[{tid}]: {res}\n"

                # Include shared mind summary
                if self._shared_mind:
                    mind_summary = await self._shared_mind.get_summary()
                    result_text += f"\nSHARED MIND STATE:\n{mind_summary}\n"

                output_data = {
                    "evaluation_previous_goal": "Success — concurrent multi-tab execution",
                    "memory": f"Drove {len(aggregated)} tabs concurrently in ONE browser with shared mind",
                    "next_goal": "Report aggregated results",
                    "action": [{"done": {"text": result_text, "success": True}}],
                }
                completion = output_format.model_validate(output_data)
            else:
                completion = str(aggregated)

            return ChatInvokeCompletion(
                completion=completion,
                usage=ChatInvokeUsage(
                    prompt_tokens=0, prompt_cached_tokens=0,
                    prompt_cache_creation_tokens=0, prompt_image_tokens=0,
                    completion_tokens=0, total_tokens=0,
                ),
            )
        else:
            # ── PASSTHROUGH ──────────────────────────────────────────
            backing = self._get_backing_llm()
            return await backing.ainvoke(messages, output_format=output_format, **kwargs)

    async def _run_concurrent(self, tasks: list[str | dict[str, str]]) -> dict[str, str]:
        """Drive N tabs concurrently in ONE browser via thin wrappers + shared mind.

        Architecture:
            ONE BrowserSession (self.browser_session or created here)
            ├── Tab 0: TabLane(target_id_0, Page_0) ──┐
            ├── Tab 1: TabLane(target_id_1, Page_1) ──┤── SharedMind
            └── Tab N: TabLane(target_id_N, Page_N) ──┘   (blackboard)

        Each TabLane has its own Page object with its own CDP session.
        CDP commands are coupled to target_id — no visual focus needed.
        SharedMind gives cross-tab awareness before every action.

        This is what browser-use cloud does when working on unfocused tabs.
        Persona just does it concurrently, with shared awareness.
        """
        from browser_use import BrowserProfile, BrowserSession

        # ── Shared mind for this batch ─────────────────────────────
        self._shared_mind = SharedMind()
        results: dict[str, str] = {}
        created_target_ids: list[str] = []

        # ── Get or create shared session ───────────────────────────
        own_session = False
        session = self.browser_session
        if session is None:
            session = BrowserSession(
                browser_profile=BrowserProfile(headless=self.headless)
            )
            own_session = True

        try:
            # Start session if not already started
            if not getattr(session, '_browser', None):
                await session.start()

            # ── Normalize tasks ────────────────────────────────────
            normalized: list[dict[str, str]] = []
            for t in tasks:
                if isinstance(t, str):
                    normalized.append({"url": t, "extract": "() => document.title"})
                else:
                    normalized.append(t)

            # ── Create tabs + TabLanes ─────────────────────────────
            lanes: list[TabLane] = []
            for i, task in enumerate(normalized):
                tab_id = f"tab-{i}"
                target_id = await session._cdp_create_new_page('about:blank')
                created_target_ids.append(target_id)
                lane = TabLane(tab_id, session, target_id, self._shared_mind)
                lanes.append(lane)
                await self._shared_mind.register_tab(
                    tab_id, task["url"], task.get("extract", "")
                )
                log.info("  Created %s (target=%s) for %s",
                         tab_id, target_id[-6:], task["url"][:50])

            # ── Drive all lanes concurrently ───────────────────────
            async def drive_lane(lane: TabLane, task: dict[str, str]) -> tuple[str, str]:
                t0 = time.monotonic()
                url = task["url"]
                extract_js = task.get("extract", "() => document.title")

                try:
                    # Read cross-tab context BEFORE acting (shared mind)
                    ctx = await lane.cross_tab_context()
                    log.info("  %s: cross-tab context: %s", lane.tab_id,
                             ctx[:80] if ctx else "none")

                    # Navigate
                    await lane.navigate(url)
                    await asyncio.sleep(0.5)  # let content render

                    # Extract title for shared mind
                    title = await lane.get_title()

                    # Extract result
                    result = await lane.evaluate(extract_js)
                    result_str = str(result) if result else "empty"

                    # Update shared mind with result
                    await self._shared_mind.update_tab(
                        lane.tab_id,
                        status="done",
                        last_result=result_str[:500],
                    )

                    elapsed = time.monotonic() - t0
                    log.info("  %s: done in %.1fs — %s",
                             lane.tab_id, elapsed, result_str[:80])
                    return lane.tab_id, result_str

                except Exception as e:
                    elapsed = time.monotonic() - t0
                    await self._shared_mind.update_tab(
                        lane.tab_id, status="error", error=str(e)[:200]
                    )
                    log.error("  %s: failed in %.1fs — %s",
                              lane.tab_id, elapsed, e)
                    return lane.tab_id, f"ERROR: {e}"

            log.info("═══ Driving %d tabs concurrently (ONE browser, shared mind) ═══",
                     len(lanes))
            t_start = time.monotonic()

            raw = await asyncio.gather(*[
                drive_lane(lane, task)
                for lane, task in zip(lanes, normalized)
            ], return_exceptions=True)

            for item in raw:
                if isinstance(item, Exception):
                    log.error("  Lane failed: %s", item)
                else:
                    tid, res = item
                    results[tid] = res

            elapsed_total = time.monotonic() - t_start
            log.info("═══ All %d tabs completed in %.1fs (ONE browser, shared mind) ═══",
                     len(lanes), elapsed_total)

        finally:
            # Clean up tabs we created (keep the browser session clean)
            for target_id in created_target_ids:
                try:
                    await session._cdp_close_page(target_id)
                except Exception:
                    pass
            # Only close session if we created it
            if own_session:
                try:
                    await session.close()
                except Exception:
                    pass

        return results

    async def _run_same_page_swarm(
        self, page: Any, tasks: list[dict[str, Any]],
        num_hands: int = 2,
    ) -> dict[str, Any]:
        """Run multiple hands on ONE page concurrently via SwarmMind.

        This is the "unified swarm intelligence" — one mind, N bodies,
        all operating on a single DOM simultaneously.

        Architecture:
            ONE Page (ONE tab, ONE DOM)
            ├── SwarmHand 0 ──┐
            ├── SwarmHand 1 ──┤── SwarmMind (coordinator)
            └── SwarmHand N ──┘

        The SwarmMind prevents logical conflicts (two hands clicking the
        same button) while allowing physical concurrency (two hands acting
        on different parts of the DOM simultaneously).

        Args:
            page: browser-use Page object (the shared tab)
            tasks: list of task dicts with {action, selector, data, priority}
            num_hands: how many concurrent hands (default 2)

        Returns:
            {results: [...], summary: str, mutations: int, halted: bool}
        """
        mind = SwarmMind()
        mind.load_tasks(tasks)

        # Create hands — all share the SAME page
        hands = [
            SwarmHand(f"hand-{i}", page, mind)
            for i in range(num_hands)
        ]

        log.info("═══ Same-page swarm: %d hands on ONE DOM, %d tasks ═══",
                 num_hands, len(tasks))
        t_start = time.monotonic()

        # Run all hands concurrently
        raw = await asyncio.gather(*[
            hand.run_loop() for hand in hands
        ], return_exceptions=True)

        elapsed = time.monotonic() - t_start

        # Aggregate results
        all_results = []
        for item in raw:
            if isinstance(item, Exception):
                log.error("  Hand failed: %s", item)
            else:
                all_results.extend(item)

        completed = await mind.get_results()
        context = await mind.get_hand_context()

        log.info("═══ Swarm done: %d tasks in %.2fs, %d mutations, halted=%s ═══",
                 len(completed), elapsed, mind._dom_version, mind._halted)

        return {
            "results": all_results,
            "completed": completed,
            "summary": context,
            "mutations": mind._dom_version,
            "halted": mind._halted,
            "halt_reason": mind._halt_reason,
            "elapsed": elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════════
# The Link — one instance per registered external agent
# ═══════════════════════════════════════════════════════════════════════════

# Detection order: specific agents first, then generic fallback
_HIJACK_DRIVERS = [HermesDriver, CursorDriver, ClaudeCodeDriver, ClineDriver, GenericHijackDriver]
_NATIVE_DRIVERS = [BrowserUseNativeDriver]
_API_DRIVERS = [BrowserUseDriver, OpenAIBridgeDriver]


class Link:
    """One Link instance per registered external agent.

    The Link is the single point of integration between Persona and one agent.
    It handles all the protocol translation so that Persona only ever sees
    the hands WebSocket protocol (hello/ready/call/result frames).
    """

    def __init__(self, agent: AgentProfile, driver: AgentDriver | None = None):
        self.agent = agent
        self.driver = driver
        self._lanes: dict[str, LinkedLane] = {}
        # Strong references to process monitors. An unheld ``create_task`` is
        # only weakly referenced by the loop and can be collected mid-await,
        # which would silently drop the death notice for a spawned engine.
        self._monitor_tasks: set[asyncio.Task] = set()
        self._hub = None         # set by LinkRegistry when hub is available
        self._persona_id = 0     # set by LinkRegistry

    # ────────────────────────────────────────────────────── detection

    @classmethod
    def detect(cls, config: dict) -> "Link":
        """Probe an agent config and return a Link with the right strategy.

        Config keys:
            name (str): human label
            tools (list[dict]): OpenAI function schemas the agent advertises
            url (str, optional): for hijack — the agent's API endpoint
            api_url (str, optional): for API bridge — the agent's completions URL
            api_key (str, optional): API key for the bridge
            model (str, optional): model name for the bridge
            binary (str, optional): for spawn — path to engine script
            persona_url (str): Persona's own URL (for spawn callback)
            persona_key (str): Persona key (for spawn auth)
            roots (list[str]): folders the spawned engine may access
            approve (str): "auto"|"writes"|"ask"
            place (str): "computer"|"cloud"
            agent_type (str, optional): force a specific agent type
        """
        name = str(config.get("name") or "unknown")
        tools = config.get("tools") or []

        # 1. Try hijack: does the agent have a native delegation tool?
        for DriverClass in _HIJACK_DRIVERS:
            profile = DriverClass.detect(tools, config)
            if profile is not None:
                driver = DriverClass()
                log.info("Link: %s detected as %s (hijack via %s, steer=%s, stop=%s)",
                         name, profile.agent_type, profile.delegation_tool,
                         profile.supports_steer, profile.supports_stop)
                return cls(profile, driver)

        # 2. Try native: agent used as an in-process library (e.g. browser-use)
        for NativeClass in _NATIVE_DRIVERS:
            profile = NativeClass.detect_from_config(config)
            if profile is not None:
                log.info("Link: %s detected as %s (native in-process)",
                         name, profile.agent_type)
                return cls(profile)

        # 3. Try API bridge: does the config have an API URL?
        for BridgeClass in _API_DRIVERS:
            profile = BridgeClass.detect_from_config(config)
            if profile is not None:
                log.info("Link: %s detected as %s (API bridge at %s)",
                         name, profile.agent_type, profile.url)
                return cls(profile)

        # 4. Fall back to spawn: launch engine instances per lane
        binary = str(config.get("binary") or "")
        if not binary:
            binary = str(Path(__file__).parent.parent / "sawii_hands.py")

        log.info("Link: %s has no delegation and no API — will spawn engine instances", name)
        profile = AgentProfile(
            name=name,
            strategy="spawn",
            agent_type="engine",
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

    # ────────────────────────────────────────────────────── delegate

    async def delegate(self, lane_id: str, instruction: str, **kwargs) -> LinkedLane:
        """Start work on an R-lane through this agent.

        ``kwargs`` carries the identity the motor must run as:
            persona_id (int): the persona whose lane this is (hub addressing)
            persona_key (str): that persona's own key, forwarded to the spawned
                engine so it registers as ITS runner, not as some other persona.
                A caller that does not pass one falls back to the key this Link
                was registered with.
        """
        if lane_id in self._lanes and self._lanes[lane_id].status == "active":
            raise RuntimeError(f"lane {lane_id} is already active on {self.agent.name}")

        linked = LinkedLane(lane_id=lane_id, agent=self.agent, instruction=instruction,
                            started_at=time.monotonic())
        linked.persona_id = int(kwargs.get("persona_id") or self._persona_id or 0)
        linked.persona_key = str(kwargs.get("persona_key") or "")
        self._lanes[lane_id] = linked

        if self.agent.strategy == "hijack":
            self._hijack_delegate(linked, instruction)
        elif self.agent.strategy == "native":
            await self._native_delegate(linked, instruction)
        elif self.agent.strategy == "api_bridge":
            await self._bridge_delegate(linked, instruction)
        else:
            await self._spawn_delegate(linked, instruction)

        return linked

    async def steer(self, lane_id: str, note: str) -> bool:
        """Send mid-work guidance to a running lane."""
        linked = self._lanes.get(lane_id)
        if not linked or linked.status != "active":
            return False

        if self.agent.strategy == "hijack" and self.driver:
            call = self.driver.build_steer_call(
                lane_id, linked.host_agent_id, note, self.agent)
            if call:
                linked._prepared_calls.append(call)
                return True

        # Spawn and API bridge: guidance goes through Persona's own lane notes,
        # which the spawned worker reads via the WebSocket.
        return True

    async def stop(self, lane_id: str) -> bool:
        """Stop work on a lane."""
        linked = self._lanes.get(lane_id)
        if not linked or linked.status not in ("active", "starting"):
            return False

        if self.agent.strategy == "hijack" and self.driver:
            call = self.driver.build_stop_call(
                lane_id, linked.host_agent_id, self.agent)
            if call:
                linked._prepared_calls.append(call)

        elif self.agent.strategy in ("api_bridge", "native"):
            if linked._bridge_task and not linked._bridge_task.done():
                linked._bridge_task.cancel()
                try:
                    await asyncio.wait_for(linked._bridge_task, timeout=PROCESS_KILL_GRACE)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

        elif self.agent.strategy == "spawn" and linked.process:
            try:
                linked.process.terminate()
                await asyncio.sleep(PROCESS_KILL_GRACE)
                if linked.process.returncode is None:
                    linked.process.kill()
            except (OSError, ProcessLookupError):
                pass

        linked.status = "stopped"
        return True

    async def status(self, lane_id: str) -> str:
        """Check the status of a lane, honestly.

        Reconciles with HandsHub first: the hub is what a tool call actually
        reaches, so a lane the hub cannot see is not running, whatever the
        process table says.
        """
        linked = self._lanes.get(lane_id)
        if not linked:
            return "unknown"

        await self.observe_hub()

        if self.agent.strategy == "spawn" and linked.process:
            rc = linked.process.returncode
            if rc is not None:
                # Cleanly exiting is only "done" if the engine ever got as far
                # as connecting. A process that died at the handshake is a
                # failure with the engine's own words attached, never a success.
                connected = linked.connected is not None and linked.connected.done() \
                    and not linked.connected.cancelled() and linked.connected.exception() is None
                if rc == 0 and connected:
                    linked.status = "done"
                else:
                    linked.status = "failed"
                    linked.error = linked.error or f"engine exited with code {rc} without working"

        if self.agent.strategy in ("api_bridge", "native") and linked._bridge_task:
            if linked._bridge_task.done():
                exc = linked._bridge_task.exception()
                linked.status = "failed" if exc else "done"
                if exc:
                    linked.error = str(exc)[:200]

        return linked.status

    def active_lanes(self) -> list[str]:
        """Lane IDs that have a live motor behind them right now.

        A lane whose engine died is not active, so ``/v1/link`` can never list a
        hand that ``/v1/hands`` cannot see.
        """
        return [lid for lid, ln in self._lanes.items() if ln.alive]

    def lane(self, lane_id: str) -> LinkedLane | None:
        """Get a linked lane by ID."""
        return self._lanes.get(lane_id)

    # ────────────────────────────────────────────────────── hijack: call pipeline

    def pending_calls(self, lane_id: str = "") -> list[dict]:
        """Hijack strategy: return prepared calls for the inference loop to pick up.

        The inference loop calls this each round. Prepared calls are consumed
        (returned once and cleared). This is how the Link feeds its delegation
        calls into the host's tool-call pipeline without modifying the inference
        loop itself.

        If lane_id is given, return only calls for that lane.
        If lane_id is empty, return calls for ALL lanes.
        """
        if self.agent.strategy != "hijack":
            return []
        out: list[dict] = []
        targets = ([self._lanes[lane_id]] if lane_id and lane_id in self._lanes
                   else list(self._lanes.values()))
        for linked in targets:
            out.extend(linked._prepared_calls)
            linked._prepared_calls.clear()
        return out

    def receive_result(self, call_id: str, result: str) -> None:
        """Hijack strategy: receive a result from the host for a delegation call.

        The inference loop calls this when a tool result arrives for one of our
        delegation calls. The driver parses it and updates the linked lane.
        """
        if self.agent.strategy != "hijack" or not self.driver:
            return
        # Find which lane owns this call
        for linked in self._lanes.values():
            if linked.host_call_id == call_id:
                parsed = self.driver.parse_delegate_result(result)
                if parsed.get("agent_id"):
                    linked.host_agent_id = parsed["agent_id"]
                status = parsed.get("status", "active")
                if status in ("done", "completed", "finished"):
                    linked.status = "done"
                elif status in ("failed", "error"):
                    linked.status = "failed"
                    linked.error = parsed.get("output", "")[:200]
                return

    # ────────────────────────────────────────────────────── hijack internals

    def _hijack_delegate(self, linked: LinkedLane, instruction: str):
        """Use the agent's native delegation tool to start work."""
        call = self.driver.build_delegate_call(linked.lane_id, instruction, self.agent)
        linked.host_call_id = call["id"]
        linked.status = "active"
        linked._prepared_calls.append(call)

    # ────────────────────────────────────────────────────── API bridge internals

    async def _bridge_delegate(self, linked: LinkedLane, instruction: str):
        """Launch an API bridge coroutine for this lane."""
        bridge_class = BrowserUseDriver if self.agent.agent_type == "browser_use" else OpenAIBridgeDriver
        linked._bridge_task = asyncio.create_task(
            bridge_class.run_bridge(linked, self._hub, self._persona_id))
        linked.status = "active"
        log.info("Link: started API bridge for lane %s via %s (%s)",
                 linked.lane_id, self.agent.name, self.agent.url)

    # ────────────────────────────────────────────────────── native internals

    async def _native_delegate(self, linked: LinkedLane, instruction: str):
        """Launch browser-use with Persona as the LLM for this lane.

        Persona implements browser-use's BaseChatModel protocol. When
        browser-use calls ainvoke(), Persona intercepts and creates
        concurrency FROM INSIDE the LLM call — spawning N concurrent
        browser-use Agents using browser-use's own Agent class.
        """
        if self.agent.agent_type == "browser_use_native":
            linked._bridge_task = asyncio.create_task(
                BrowserUseNativeDriver.run_native(linked, self._hub, self._persona_id))
        else:
            linked.status = "failed"
            linked.error = f"no native driver for {self.agent.agent_type}"
            return
        linked.status = "active"
        log.info("Link: started native agent for lane %s via %s",
                 linked.lane_id, self.agent.name)

    # ────────────────────────────────────────────────────── spawn internals

    async def _spawn_delegate(self, linked: LinkedLane, instruction: str):
        """Launch a new sawii_hands.py process for this lane, and PROVE it connected.

        The spawned process connects back to Persona's /v1/hands WebSocket
        with the lane ID. HandsHub registers it and handles all subsequent
        call/result traffic. The Link manages the process lifecycle *and*
        resolves ``linked.connected`` the moment the hub sees this lane's
        runner, so a caller can await real liveness instead of trusting that a
        started process is a working motor. If the process dies or never
        checks in, the lane ends ``failed`` carrying the engine's own stderr —
        "running with no motor" is never an allowed state.
        """
        binary = self.agent.binary
        if not binary or not os.path.isfile(binary):
            linked.status = "failed"
            linked.error = f"binary not found: {binary}"
            log.error("Link: spawn binary not found: %s", binary)
            return

        # The engine authenticates as the persona that owns this lane. Falling
        # back to the registration key keeps single-persona deployments working,
        # but a lane created by persona B must never connect as persona A: its
        # runner would register on the wrong hub slot and the card would show
        # "running" with no motor under its own persona.
        key = linked.persona_key or self.agent.key
        url = self.agent.url
        if not key or not url:
            linked.status = "failed"
            linked.error = ("engine has no persona url/key to connect with "
                            f"(url={'set' if url else 'empty'}, key={'set' if key else 'empty'})")
            log.error("Link: cannot spawn %s for lane %s: %s",
                      self.agent.name, linked.lane_id, linked.error)
            return

        cmd = [
            sys.executable, binary,
            "--url", url,
            "--key", key,
            "--approve", self.agent.approve,
            "--place", self.agent.place,
            "--lane", linked.lane_id,
        ]
        for root in self.agent.roots:
            cmd += ["--root", root]
        cmd += self.agent.extra_args

        env = {**os.environ}
        # Pass the lane instruction so the worker can read it on start.
        env["SAWII_LANE_INSTRUCTION"] = instruction[:4000]
        # Belt and braces: the engine also reads its connection from the
        # environment, so a quoting mistake in argv cannot produce a child
        # that dies on "--url and --key are required".
        env["SAWII_HANDS_URL"] = url
        env["SAWII_HANDS_KEY"] = key
        env["SAWII_HANDS_LANE"] = linked.lane_id

        try:
            linked.process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            linked.status = "active"
            log.info("Link: spawned %s (pid %d) for lane %s (persona %s, place %s, roots %s)",
                     binary, linked.process.pid, linked.lane_id,
                     linked.persona_id or self._persona_id, self.agent.place,
                     ", ".join(self.agent.roots) or "(none)")

            # Keep a strong reference: an unheld create_task can be garbage
            # collected mid-flight, which would silently drop the monitor and
            # leave a dead process looking active forever.
            task = asyncio.create_task(self._monitor_process(linked))
            self._monitor_tasks.add(task)
            task.add_done_callback(self._monitor_tasks.discard)

            # Do not return "active" until the hub has actually registered this
            # lane's runner (or the engine proves it is dead).
            await self._await_connection(linked)

        except Exception as exc:
            linked.status = "failed"
            linked.error = str(exc)[:200]
            log.exception("Link: failed to spawn %s: %s", binary, exc)

    def hub_lane_seen(self, linked: LinkedLane) -> bool:
        """Has HandsHub registered this lane's runner under the right persona?

        The hub is the only proof that a motor exists: it is the thing tool
        calls are actually routed through. Reading it here (instead of trusting
        that ``create_subprocess_exec`` returned) is what makes an honest
        ``running`` possible. The persona must match too — a runner that checked
        in as somebody else is not this lane's hand.
        """
        hub = self._hub
        pid = linked.persona_id or self._persona_id
        if hub is None or not pid:
            return False
        try:
            runner = hub.runners.get(hub._key(pid, self.agent.place, linked.lane_id))
        except Exception:  # noqa: BLE001 - a hub we cannot read proves nothing
            return False
        return runner is not None

    async def observe_hub(self) -> None:
        """Reconcile every spawn lane with what the hub can actually see.

        Called on each status/diagnostics read. A lane whose engine vanished is
        moved to ``failed`` with a reason, so ``/v1/link`` and the card agree
        with ``/v1/hands`` rather than contradicting it.
        """
        for linked in list(self._lanes.values()):
            if self.agent.strategy != "spawn" or linked.status not in ("active", "starting"):
                continue
            if linked.process is not None and linked.process.returncode is not None:
                continue  # _monitor_process owns the verdict for a dead process
            seen = self.hub_lane_seen(linked)
            if seen and not linked.connected:
                linked.connected = True
                log.info("Link: lane %s engine is live on /v1/hands (%s)",
                         linked.lane_id, self.agent.name)
            elif linked.connected and not seen:
                linked.status = "failed"
                linked.error = "the engine disconnected from /v1/hands"
                log.warning("Link: lane %s lost its engine: %s", linked.lane_id, linked.error)
                self._fire_exit(linked, None, linked.error)

    async def _await_connection(self, linked: LinkedLane) -> bool:
        """Block until this lane's runner is on the hub, or it dies, or timeout.

        Returns True only when HandsHub shows the runner. Anything else marks
        the lane ``failed`` with the reason, so nothing upstream can report a
        motor that does not exist.
        """
        deadline = time.monotonic() + SPAWN_CONNECT_TIMEOUT
        while time.monotonic() < deadline:
            if self.hub_lane_seen(linked):
                linked.connected = True
                log.info("Link: lane %s engine connected (%s pid %s)",
                         linked.lane_id, self.agent.name,
                         linked.process.pid if linked.process else "?")
                return True
            if linked.process is not None and linked.process.returncode is not None:
                break  # died before connecting; the monitor fills in the reason
            await asyncio.sleep(0.1)
        if self.hub_lane_seen(linked):
            linked.connected = True
            return True
        if linked.status not in ("active", "starting"):
            return False  # the monitor already recorded why it died
        linked.status = "failed"
        if linked.process is not None and linked.process.returncode is None:
            linked.error = (f"engine started but never connected to /v1/hands within "
                            f"{int(SPAWN_CONNECT_TIMEOUT)} s (check the persona key and url)")
        else:
            rc = linked.process.returncode if linked.process is not None else None
            linked.error = f"engine exited before connecting (exit code {rc})"
        log.error("Link: lane %s: %s", linked.lane_id, linked.error)
        self._fire_exit(linked, None, linked.error)
        return False

    def _fire_exit(self, linked: LinkedLane, rc: int | None, error: str) -> None:
        """Tell whoever opened this lane that its motor is gone."""
        if linked.on_exit is None:
            return
        try:
            linked.on_exit(rc, error)
        except Exception:  # noqa: BLE001 - a hook must not break the Link
            log.exception("Link: engine exit hook failed for lane %s", linked.lane_id)

    async def _monitor_process(self, linked: LinkedLane):
        """Wait for a spawned process to finish and update status."""
        if not linked.process:
            return
        try:
            stdout, stderr = await linked.process.communicate()
            rc = linked.process.returncode
            detail = (stderr or stdout or b"").decode("utf-8", errors="replace").strip()
            # The engine talks to its own stdout/stderr; capture the tail of it
            # in the error so an operator sees WHY (bad key, refused, no roots).
            if rc != 0:
                linked.status = "failed"
                linked.error = (detail[-400:] or f"engine exited with code {rc}")
                log.warning("Link: spawned process for lane %s exited with %d: %s",
                            linked.lane_id, rc, linked.error)
            elif linked.status in ("active", "starting"):
                # A process that exits without ever being seen on /v1/hands did
                # no work, however clean its exit code looks.
                if linked.connected:
                    linked.status = "done"
                    if detail:
                        log.info("Link: engine for lane %s finished cleanly: %s",
                                 linked.lane_id, detail[-200:])
                else:
                    linked.status = "failed"
                    linked.error = (detail[-400:]
                                    or f"engine exited (code {rc}) without ever connecting")
                    log.warning("Link: spawned process for lane %s exited with %d: %s",
                                linked.lane_id, rc, linked.error)
            # Tell whoever opened the lane. A silent death mid-goal is worse
            # than an honest failure.
            self._fire_exit(linked, rc, linked.error)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Link: process monitor failed for lane %s", linked.lane_id)

    # ────────────────────────────────────────────────────── cleanup

    async def shutdown(self):
        """Stop all active lanes."""
        for lane_id in list(self._lanes):
            await self.stop(lane_id)

    def info(self) -> dict:
        """Summary for diagnostics and the health endpoint."""
        return {
            "name": self.agent.name,
            "strategy": self.agent.strategy,
            "agent_type": self.agent.agent_type,
            "delegation_tool": self.agent.delegation_tool or None,
            "supports_steer": self.agent.supports_steer,
            "supports_stop": self.agent.supports_stop,
            "active_lanes": self.active_lanes(),
            "total_lanes": len(self._lanes),
        }


# ═══════════════════════════════════════════════════════════════════════════
# LinkRegistry — all registered agent Links for a persona
# ═══════════════════════════════════════════════════════════════════════════

class LinkRegistry:
    """All registered agent Links, keyed by agent name.

    The registry is the single place where Persona discovers what agents are
    available, what strategies they use, and how to connect to them. It is
    set on ``mind.links`` by stitch.py at startup.
    """

    def __init__(self):
        self._links: dict[str, Link] = {}
        self._hub = None
        self._persona_id = 0
        # persona id -> the persona key that persona authenticated with, seen on
        # the request that opened its work (never persisted). A spawned engine
        # must connect as the persona whose lane it drives: the hub keys runners
        # by (persona, place, lane), so a hand registered under another persona
        # is invisible to the very lane that spawned it.
        self._persona_keys: dict[int, str] = {}

    def remember_persona_key(self, persona_id: int, key: str) -> None:
        """Note the key a persona is using, so its motors can authenticate as it."""
        pid = int(persona_id or 0)
        key = str(key or "")
        if pid and key and len(key) < 200:
            self._persona_keys[pid] = key

    def persona_key(self, persona_id: int) -> str:
        """The key to spawn this persona's engines with ('' when unknown)."""
        return self._persona_keys.get(int(persona_id or 0), "")

    def set_hub(self, hub, persona_id: int = 0):
        """Provide the HandsHub reference so API bridges can use it.

        Called by stitch.py after both the registry and the hub are initialized.
        """
        self._hub = hub
        self._persona_id = persona_id
        for link in self._links.values():
            link._hub = hub
            link._persona_id = persona_id

    def register(self, config: dict) -> Link:
        """Detect and register an agent. Returns the Link."""
        link = Link.detect(config)
        link._hub = self._hub
        link._persona_id = self._persona_id
        self._links[link.agent.name] = link
        return link

    def get(self, name: str) -> Link | None:
        return self._links.get(name)

    def all(self) -> dict[str, Link]:
        return dict(self._links)

    def compatible(self, name: str) -> bool:
        """Can we connect to this agent at all?"""
        return name in self._links

    def strategies(self) -> dict[str, str]:
        """Agent name -> strategy for all registered agents."""
        return {name: link.agent.strategy for name, link in self._links.items()}

    def by_strategy(self, strategy: str) -> list[Link]:
        """All links using a given strategy."""
        return [link for link in self._links.values() if link.agent.strategy == strategy]

    def find_for_tools(self, tools: list[dict]) -> Link | None:
        """Find a registered Link whose agent matches these tool schemas.

        Used when the host advertises tools and Persona needs to know if any
        registered agent can handle delegation through them.
        """
        for link in self._links.values():
            if link.agent.strategy == "hijack" and link.agent.delegation_tool:
                for tool in tools:
                    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
                    if isinstance(fn, dict) and fn.get("name") == link.agent.delegation_tool:
                        return link
        return None

    def pending_calls(self) -> list[dict]:
        """All pending hijack calls across all agents, for the inference loop."""
        out: list[dict] = []
        for link in self._links.values():
            out.extend(link.pending_calls())
        return out

    def receive_result(self, call_id: str, result: str) -> bool:
        """Route a result to the right Link. Returns True if any Link claimed it."""
        for link in self._links.values():
            for linked in link._lanes.values():
                if linked.host_call_id == call_id:
                    link.receive_result(call_id, result)
                    return True
        return False

    async def observe_hub(self) -> None:
        """Reconcile every Link's lanes with what HandsHub can actually see."""
        for link in self._links.values():
            await link.observe_hub()

    def info(self) -> dict:
        """Diagnostics summary for all registered agents."""
        return {name: link.info() for name, link in self._links.items()}

    async def shutdown_all(self):
        """Stop all active lanes on all agents."""
        for link in self._links.values():
            await link.shutdown()
