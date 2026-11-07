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
from typing import Any

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

    # Lifecycle
    status: str = "starting"               # starting | active | done | failed | stopped
    started_at: float = 0.0
    error: str = ""


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


# ═══════════════════════════════════════════════════════════════════════════
# The Link — one instance per registered external agent
# ═══════════════════════════════════════════════════════════════════════════

# Detection order: specific agents first, then generic fallback
_HIJACK_DRIVERS = [HermesDriver, CursorDriver, ClaudeCodeDriver, ClineDriver, GenericHijackDriver]
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

        # 2. Try API bridge: does the config have an API URL?
        for BridgeClass in _API_DRIVERS:
            profile = BridgeClass.detect_from_config(config)
            if profile is not None:
                log.info("Link: %s detected as %s (API bridge at %s)",
                         name, profile.agent_type, profile.url)
                return cls(profile)

        # 3. Fall back to spawn: launch engine instances per lane
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
        """Start work on an R-lane through this agent."""
        if lane_id in self._lanes and self._lanes[lane_id].status == "active":
            raise RuntimeError(f"lane {lane_id} is already active on {self.agent.name}")

        linked = LinkedLane(lane_id=lane_id, agent=self.agent, instruction=instruction,
                            started_at=time.monotonic())
        self._lanes[lane_id] = linked

        if self.agent.strategy == "hijack":
            self._hijack_delegate(linked, instruction)
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

        elif self.agent.strategy == "api_bridge":
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
        """Check the status of a lane."""
        linked = self._lanes.get(lane_id)
        if not linked:
            return "unknown"

        if self.agent.strategy == "spawn" and linked.process:
            if linked.process.returncode is not None:
                linked.status = "done" if linked.process.returncode == 0 else "failed"

        if self.agent.strategy == "api_bridge" and linked._bridge_task:
            if linked._bridge_task.done():
                exc = linked._bridge_task.exception()
                linked.status = "failed" if exc else "done"
                if exc:
                    linked.error = str(exc)[:200]

        return linked.status

    def active_lanes(self) -> list[str]:
        """Lane IDs currently active on this agent."""
        return [lid for lid, ln in self._lanes.items() if ln.status == "active"]

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

    # ────────────────────────────────────────────────────── spawn internals

    async def _spawn_delegate(self, linked: LinkedLane, instruction: str):
        """Launch a new sawii_hands.py process for this lane.

        The spawned process connects back to Persona's /v1/hands WebSocket
        with the lane ID. HandsHub registers it and handles all subsequent
        call/result traffic. The Link only manages the process lifecycle.
        """
        binary = self.agent.binary
        if not binary or not os.path.isfile(binary):
            linked.status = "failed"
            linked.error = f"binary not found: {binary}"
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
        # Pass the lane instruction so the worker can read it on start.
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

            # Monitor the process in the background so we know when it exits
            asyncio.create_task(self._monitor_process(linked))

        except Exception as exc:
            linked.status = "failed"
            linked.error = str(exc)[:200]
            log.exception("Link: failed to spawn %s: %s", binary, exc)

    async def _monitor_process(self, linked: LinkedLane):
        """Wait for a spawned process to finish and update status."""
        if not linked.process:
            return
        try:
            stdout, stderr = await linked.process.communicate()
            rc = linked.process.returncode
            if rc == 0:
                if linked.status == "active":
                    linked.status = "done"
            else:
                if linked.status == "active":
                    linked.status = "failed"
                    linked.error = (stderr or stdout or b"").decode("utf-8", errors="replace")[:200]
                    log.warning("Link: spawned process for lane %s exited with %d: %s",
                                linked.lane_id, rc, linked.error)
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

    def info(self) -> dict:
        """Diagnostics summary for all registered agents."""
        return {name: link.info() for name, link in self._links.items()}

    async def shutdown_all(self):
        """Stop all active lanes on all agents."""
        for link in self._links.values():
            await link.shutdown()
