"""Hands voice sidecar — reuses the existing voice-module sandbox.

Does not rebuild STT/TTS/barge-in. It finds the sibling (or configured)
``voice-module`` package and starts its HTTP server so Hands can test live
voice conversation alongside Persona's built-in ``/v1/voice`` WebSocket.

Discovery order for the package root:
  1. ``VOICE_MODULE_ROOT`` env
  2. ``<hands>/../voice-module`` (sawii-persona workspace layout)
  3. ``<hands>/voice-module`` (vendored copy, if present)

The sidecar listens on ``VOICE_MODULE_HOST`` / ``VOICE_MODULE_PORT``
(defaults ``127.0.0.1:18766``). Endpoints: ``/health``, ``/speak``,
``/transcribe``, ``/turn``, ``/interrupt``.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("hands.voice")

HANDS_ROOT = Path(__file__).resolve().parent


def find_voice_module_root() -> Path | None:
    env = (os.environ.get("VOICE_MODULE_ROOT") or "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(HANDS_ROOT.parent / "voice-module")
    candidates.append(HANDS_ROOT / "voice-module")
    for path in candidates:
        if (path / "voice_module" / "server.py").is_file():
            return path.resolve()
    return None


def voice_status() -> dict[str, Any]:
    """Diagnostics for /v1/voice-status — what is wired and how to test."""
    root = find_voice_module_root()
    host = os.environ.get("VOICE_MODULE_HOST", "127.0.0.1")
    port = int(os.environ.get("VOICE_MODULE_PORT", "18766"))
    return {
        "persona_realtime": {
            "websocket": "/v1/voice",
            "web_ui": "/app",
            "source": "persona/voice/realtime.py + persona/web/index.html",
            "notes": (
                "Primary live conversation path. Mic button in /app opens "
                "the Persona VoiceSession WebSocket (ASR/TTS from Persona settings)."
            ),
        },
        "voice_module_sidecar": {
            "root": str(root) if root else None,
            "found": root is not None,
            "url": f"http://{host}:{port}",
            "health": f"http://{host}:{port}/health",
            "endpoints": ["/health", "/speak", "/transcribe", "/turn", "/interrupt"],
            "source": "voice-module/ (DeepInfra Whisper STT + MiMo TTS, barge-in always on)",
            "notes": (
                "Sandbox sidecar for Hands. Reused as-is — not rebuilt. "
                "Requires DEEPINFRA_API_KEY."
            ),
        },
        "brains_voice_a": {
            "source": "persona/brains/voice_a.py",
            "notes": "Lobe-A speak path used inside Persona turns; not a separate server.",
        },
    }


class VoiceSidecar:
    """Background process running ``python -m voice_module.server``."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.root: Path | None = None
        self.url: str | None = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> dict[str, Any]:
        if self.running:
            return {"ok": True, "already": True, "url": self.url}

        root = find_voice_module_root()
        if root is None:
            log.warning(
                "voice-module not found — Persona /v1/voice still works; "
                "set VOICE_MODULE_ROOT to enable the Hands voice sidecar"
            )
            return {"ok": False, "error": "voice_module_not_found", "status": voice_status()}

        host = os.environ.get("VOICE_MODULE_HOST", "127.0.0.1")
        port = int(os.environ.get("VOICE_MODULE_PORT", "18766"))
        env = os.environ.copy()
        env["VOICE_MODULE_HOST"] = host
        env["VOICE_MODULE_PORT"] = str(port)
        # Prefer workspace keys if the sidecar cwd would miss them.
        for key_file in (
            HANDS_ROOT.parent / "keys.env",
            HANDS_ROOT.parent / ".env",
            root / ".env",
        ):
            if key_file.is_file():
                env.setdefault("VOICE_MODULE_DOTENV", str(key_file))
                break

        cmd = [sys.executable, "-m", "voice_module.server"]
        log.info("Starting voice-module sidecar: %s (cwd=%s)", " ".join(cmd), root)
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            log.exception("Failed to start voice-module")
            return {"ok": False, "error": str(exc)}

        self.root = root
        self.url = f"http://{host}:{port}"

        # Brief wait + health probe (non-fatal if slow).
        ready = False
        for _ in range(20):
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() if self.proc.stderr else b"")[:500]
                log.error("voice-module exited early: %s", err.decode("utf-8", "replace"))
                return {"ok": False, "error": "exited_early", "stderr": err.decode("utf-8", "replace")}
            if _probe_health(self.url):
                ready = True
                break
            time.sleep(0.25)

        log.info("voice-module sidecar %s at %s", "ready" if ready else "starting", self.url)
        return {
            "ok": True,
            "url": self.url,
            "ready": ready,
            "pid": self.proc.pid,
            "root": str(root),
        }

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            log.info("Stopping voice-module sidecar pid=%s", self.proc.pid)
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def _probe_health(base_url: str) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(base_url + "/health", timeout=0.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_in_thread(sidecar: VoiceSidecar) -> None:
    """Start without blocking the asyncio loop's first tick for long."""
    threading.Thread(target=sidecar.start, name="voice-sidecar", daemon=True).start()
