#!/usr/bin/env python3
"""Hands -- the one-command app.

Starts the local Hands backend (the deck, the Link, the per-lane engines that
drive THIS computer) with its thinking pointed at your deployed Persona, serves
the Hands UI on the same port, and opens it in an app window.

    python hands_app.py                       # first run: set the Persona URL + key in the UI
    python hands_app.py --root ~/Projects     # folders the engine may touch

The Persona brain is any OpenAI-compatible Persona endpoint (the Modal
``persona-crossing`` deployment, or the Railway gateway). It is configured in the
UI (Brain screen) and kept in ``<data-dir>/config.json``.  Nothing here edits the
mind: it only decides which endpoint the mind's lobes call.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_BRAIN_URL = "https://anasalsawy--persona-crossing-personacrossing-web.us-east.modal.run/v1"
DEFAULT_MODEL = "persona-l"
CLOUD = os.environ.get("HANDS_CLOUD") == "1"   # hosted: no computer, a sandbox workspace
ACCESS_KEY = ""                                   # cloud login secret (set in main)
RESTART_CODE = 75  # supervised launchers (Hands.bat / hands.sh) restart on this


# --------------------------------------------------------------------- config
def data_dir() -> Path:
    return Path(os.environ.get("HANDS_DATA_DIR") or (Path.home() / ".hands"))


def config_path() -> Path:
    return data_dir() / "config.json"


def load_config() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    config_path().parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(config_path())


def normalise_brain_url(url: str) -> str:
    """Accept the bare host, the /v1 root or the full completions URL."""
    url = (url or "").strip().rstrip("/")
    for tail in ("/chat/completions", "/v1/chat/completions"):
        if url.endswith(tail):
            url = url[: -len(tail)]
            break
    if url and not url.endswith("/v1"):
        url += "/v1"
    return url


def apply_brain_env(cfg: dict) -> bool:
    """Point every lobe at the Persona endpoint. Returns True when a brain is set."""
    url = normalise_brain_url(cfg.get("brain_url", ""))
    key = (cfg.get("brain_key") or "").strip()
    model = (cfg.get("brain_model") or DEFAULT_MODEL).strip()
    if not (url and key):
        return False
    for prefix in ("A", "B", "C", "INFER_A", "STEER"):
        os.environ[f"SAWII_{prefix}_BASE_URL"] = url
        os.environ[f"SAWII_{prefix}_API_KEY"] = key
        os.environ[f"SAWII_{prefix}_MODEL"] = model
    # Persona reasons on its own; do not ask the provider for a thinking stream twice.
    os.environ.setdefault("SAWII_DISABLE_THINKING", "1")
    return True


def engine_ready() -> bool:
    """The Agent TARS engine: Node 22+, the loop's dependencies and the built TARS core."""
    return ((shutil.which("node") or shutil.which("node.exe")) is not None
            and (ROOT / "engine" / "node_modules").exists()
            and (ROOT / "engine/ui-tars-desktop/multimodal/agent-tars/core/dist/index.js").exists())


def missing_parts() -> list[str]:
    """What the cards need that this machine does not have yet (shown in the UI)."""
    out = []
    if not (shutil.which("node") or shutil.which("node.exe")):
        out.append("Node.js 22+ (runs the Agent TARS engine)")
    elif not (ROOT / "engine" / "node_modules").exists() or not engine_ready():
        out.append("Agent TARS engine (npm ci --prefix engine && npm run build:tars --prefix engine)")
    return out


def _loopback(request) -> bool:
    """Config + key routes answer only the app itself: loopback client, own Host,
    and no foreign Origin (blocks DNS-rebinding and other websites' fetches)."""
    client = request.client.host if request.client else ""
    if client not in ("127.0.0.1", "::1", "localhost"):
        return False
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
        return False
    origin = request.headers.get("origin")
    if origin:
        h = origin.split("://", 1)[-1].split("/")[0].split(":")[0].lower()
        if h not in ("127.0.0.1", "localhost", "[::1]", "tauri.localhost"):
            return False
    return True


# ---------------------------------------------------------------------- routes
def build_routes(*, app, rt, settings, persona_key, persona_id, links=None, port: int, roots, approve, ui_dir):
    from starlette.responses import FileResponse, JSONResponse, RedirectResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    def bearer_ok(request) -> bool:
        import secrets
        tok = (request.headers.get("authorization") or "")[7:].strip() or request.query_params.get("key", "")
        return bool(persona_key) and bool(tok) and secrets.compare_digest(tok, persona_key)

    def owner(request) -> bool:
        """The person this runs for: the machine itself, or (hosted) a signed-in client."""
        return _loopback(request) or (CLOUD and bearer_ok(request))

    def deny():
        return JSONResponse({"error": {"message": "local app only"}}, status_code=403)

    async def bootstrap(request):
        if not owner(request):
            return deny()
        cfg = load_config()
        return JSONResponse({
            "persona_key": persona_key,
            "persona_id": persona_id,
            "brain": {
                "url": cfg.get("brain_url") or "",
                "model": cfg.get("brain_model") or DEFAULT_MODEL,
                "key_set": bool((cfg.get("brain_key") or "").strip()),
                "default_url": DEFAULT_BRAIN_URL,
                "configured": bool(cfg.get("brain_url") and cfg.get("brain_key")),
            },
            "roots": list(roots),
            "approve": approve,
            "port": port,
            "lan": bool(cfg.get("lan")),
            "cloud": CLOUD,
            "engine": "agent-tars", "engine_ready": engine_ready(),
            "missing": missing_parts(),
        })

    pair = {"code": "", "until": 0.0}
    fails: dict[str, list[float]] = {}

    def _lan_urls() -> list[str]:
        import socket
        ips: list[str] = []
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if not ip.startswith("127.") and ip not in ips:
                    ips.append(ip)
        except OSError:
            pass
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith("127.") and ip not in ips:
                ips.insert(0, ip)
        except OSError:
            pass
        return [f"http://{ip}:{port}/ui/" for ip in ips]

    async def pair_info(request):
        """Loopback only: the code a phone types, and the addresses it can reach."""
        if not owner(request):
            return deny()
        import secrets
        if request.query_params.get("new") or time.time() > pair["until"]:
            pair["code"] = f"{secrets.randbelow(10**6):06d}"
            pair["until"] = time.time() + 600
        cfg = load_config()
        return JSONResponse({
            "lan": bool(cfg.get("lan")),
            "listening": bool(cfg.get("lan")),
            "code": pair["code"],
            "expires_in": max(0, int(pair["until"] - time.time())),
            "urls": _lan_urls() if cfg.get("lan") else [],
        })

    async def pair_claim(request):
        """A phone on the same network trades the code for the persona key."""
        import secrets
        client = request.client.host if request.client else "?"
        now = time.time()
        recent = [t for t in fails.get(client, []) if now - t < 300]
        if len(recent) >= 5:
            return JSONResponse({"error": {"message": "too many tries, wait a few minutes"}}, status_code=429)
        try:
            body = await request.json()
        except ValueError:
            body = {}
        code = str(body.get("code", "")).strip()
        if CLOUD:
            ok = bool(ACCESS_KEY) and secrets.compare_digest(code, ACCESS_KEY)
        else:
            ok = bool(pair["code"]) and now <= pair["until"] and secrets.compare_digest(code, pair["code"])
        if not ok:
            recent.append(now)
            fails[client] = recent
            return JSONResponse({"error": {"message": "wrong code or key"}}, status_code=403)
        if not CLOUD:
            pair["code"] = ""  # one use
        return JSONResponse({"persona_key": persona_key, "persona_id": persona_id})

    async def put_config(request):
        if not owner(request):
            return deny()
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"message": "bad json"}}, status_code=400)
        cfg = load_config()
        for k_in, k_out in (("brain_url", "brain_url"), ("brain_model", "brain_model")):
            if isinstance(body.get(k_in), str):
                cfg[k_out] = body[k_in].strip()
        if isinstance(body.get("brain_key"), str) and body["brain_key"].strip():
            cfg["brain_key"] = body["brain_key"].strip()  # blank keeps the saved one
        if isinstance(body.get("roots"), list):
            cfg["roots"] = [str(r).strip() for r in body["roots"] if str(r).strip()]
        if body.get("approve") in ("ask", "writes", "auto"):
            cfg["approve"] = body["approve"]
        if isinstance(body.get("lan"), bool):
            cfg["lan"] = body["lan"]
        save_config(cfg)
        _schedule_restart()
        return JSONResponse({"ok": True, "restarting": True})

    async def brain_check(request):
        """A real call through the mind's lobe: plain reply, then a tool-calling probe."""
        if not owner(request):
            return deny()
        lobe = getattr(rt.mind, "a", None)
        out: dict = {"ok": False, "url": getattr(getattr(lobe, "endpoint", None), "base_url", "")}
        if lobe is None or not getattr(getattr(lobe, "endpoint", None), "configured", False):
            out["error"] = "no brain configured"
            return JSONResponse(out)
        t0 = time.monotonic()
        try:
            reply = await asyncio.wait_for(lobe.chat(
                [{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=40, temperature=0.0), timeout=90)
            out["ok"] = True
            out["reply"] = str(getattr(reply, "text", "") or "").strip()[:200]
            out["seconds"] = round(time.monotonic() - t0, 2)
        except Exception as exc:  # noqa: BLE001 - say what really happened
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return JSONResponse(out)
        tool = {"type": "function", "function": {
            "name": "get_time", "description": "Return the current time.",
            "parameters": {"type": "object", "properties": {}}}}
        try:
            t1 = time.monotonic()
            reply = await asyncio.wait_for(lobe.chat(
                [{"role": "user", "content": "What time is it? Use the tool."}],
                tools=[tool], max_tokens=120, temperature=0.0), timeout=90)
            calls = list(getattr(reply, "tool_calls", None) or [])
            out["tools"] = {"supported": bool(calls), "seconds": round(time.monotonic() - t1, 2),
                            "called": calls[0].name if calls else ""}
        except Exception as exc:  # noqa: BLE001
            out["tools"] = {"supported": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
        return JSONResponse(out)

    ws_root = Path(roots[0]).expanduser().resolve() if roots else Path.cwd()
    SKIP = {".git", "node_modules", "__pycache__", ".cache", ".venv"}

    def _inside(rel: str) -> Path | None:
        p = (ws_root / rel.lstrip("/\\")).resolve()
        return p if p == ws_root or ws_root in p.parents else None

    async def ws_tree(request):
        """Every file the cards have made, newest first."""
        if not owner(request):
            return deny()
        files = []
        for dirpath, dirs, names in os.walk(ws_root):
            dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
            for n in names:
                fp = Path(dirpath) / n
                try:
                    st = fp.stat()
                except OSError:
                    continue
                files.append({"path": fp.relative_to(ws_root).as_posix(), "size": st.st_size, "mtime": st.st_mtime})
            if len(files) > 3000:
                break
        files.sort(key=lambda f: -f["mtime"])
        return JSONResponse({"root": ws_root.name, "files": files[:500], "now": time.time()})

    async def ws_file(request):
        """One file, to read or show. HTML is sandboxed; nothing here can run scripts on the app's origin."""
        if not owner(request):
            return deny()
        fp = _inside(request.query_params.get("path", ""))
        if fp is None or not fp.is_file():
            return JSONResponse({"error": {"message": "no such file"}}, status_code=404)
        import mimetypes
        ctype = mimetypes.guess_type(fp.name)[0] or "text/plain"
        if fp.stat().st_size > 8_000_000:
            return JSONResponse({"error": {"message": "file too large to show"}}, status_code=413)
        headers = {"Content-Security-Policy": "sandbox allow-scripts; default-src 'none'; img-src data: blob: https:; "
                   "style-src 'unsafe-inline' https:; script-src 'unsafe-inline' https:; font-src data: https:",
                   "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
        if request.query_params.get("download"):
            headers["Content-Disposition"] = f'attachment; filename="{fp.name}"'
            return FileResponse(fp, media_type="application/octet-stream", headers=headers)
        if ctype.startswith("text/") or ctype in ("application/json", "application/javascript", "application/xml") or not Path(fp).suffix:
            ctype = ctype if ctype == "text/html" else (ctype if ctype.startswith("image/") else "text/plain; charset=utf-8")
        return FileResponse(fp, media_type=ctype, headers=headers)

    async def index(request):
        return RedirectResponse("/ui/")

    async def ui_index(request):
        return FileResponse(Path(ui_dir) / "index.html")

    routes = [
        Route("/hands/bootstrap", bootstrap, methods=["GET"]),
        Route("/hands/config", put_config, methods=["POST"]),
        Route("/hands/pair-info", pair_info, methods=["GET"]),
        Route("/hands/pair", pair_claim, methods=["POST"]),
        Route("/hands/brain-check", brain_check, methods=["POST"]),
        Route("/hands/workspace", ws_tree, methods=["GET"]),
        Route("/hands/workspace/file", ws_file, methods=["GET"]),
        Route("/", index, methods=["GET"]),
    ]
    if Path(ui_dir, "index.html").exists():
        routes.append(Route("/ui/", ui_index, methods=["GET"]))
        routes.append(Mount("/ui", app=StaticFiles(directory=str(ui_dir), html=True), name="hands-ui"))
    # insert(0) reverses; hand them back so the first listed ends up first
    return list(reversed(routes))


def _schedule_restart() -> None:
    def go():
        time.sleep(0.6)
        if os.environ.get("HANDS_SUPERVISED") == "1":
            os._exit(RESTART_CODE)
        argv = [sys.executable] + sys.argv
        if os.name == "nt":
            subprocess.Popen(argv, creationflags=0x00000008)  # DETACHED_PROCESS
            os._exit(0)
        os.execv(sys.executable, argv)

    threading.Thread(target=go, daemon=True).start()


# ------------------------------------------------------------------------- run
def _open_window(url: str) -> None:
    """Prefer a chromeless Edge/Chrome app window; fall back to the default browser."""
    cands = []
    if os.name == "nt":
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if not base:
                continue
            cands += [Path(base) / "Microsoft/Edge/Application/msedge.exe",
                      Path(base) / "Google/Chrome/Application/chrome.exe"]
    else:
        for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
            p = shutil.which(name)
            if p:
                cands.append(Path(p))
        if sys.platform == "darwin":
            cands += [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                      Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")]
    for c in cands:
        if c and c.exists():
            try:
                profile = data_dir() / "window-profile"
                subprocess.Popen([str(c), f"--app={url}", f"--user-data-dir={profile}",
                                  "--window-size=1380,900", "--no-first-run"])
                return
            except OSError:
                continue
    webbrowser.open(url)


def main() -> None:
    global CLOUD, ACCESS_KEY
    ap = argparse.ArgumentParser(description="Hands: the Persona computer-use app")
    ap.add_argument("--port", type=int, default=int(os.environ.get("HANDS_PORT", "8800")))
    ap.add_argument("--root", action="append", default=[], help="folder the engine may touch (repeatable)")
    ap.add_argument("--approve", choices=["ask", "writes", "auto"], default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-open", action="store_true", help="do not open the app window")
    ap.add_argument("--no-voice", action="store_true", default=True)
    ap.add_argument("--ui-dir", default=None)
    ap.add_argument("--cloud", action="store_true", default=CLOUD,
                    help="hosted mode: no computer, work happens in a sandbox workspace; login with HANDS_ACCESS_KEY")
    a = ap.parse_args()

    if a.cloud:
        CLOUD = True
        os.environ["HANDS_CLOUD"] = "1"
        a.no_open = True
        a.port = int(os.environ.get("PORT") or a.port)
    if a.data_dir:
        os.environ["HANDS_DATA_DIR"] = a.data_dir
    cfg = load_config()
    if CLOUD:
        # a hosted app is configured from its environment (env wins on restart)
        for env_k, cfg_k in (("HANDS_BRAIN_URL", "brain_url"), ("HANDS_BRAIN_KEY", "brain_key"),
                             ("HANDS_BRAIN_MODEL", "brain_model")):
            if os.environ.get(env_k, "").strip():
                cfg[cfg_k] = os.environ[env_k].strip()
        ACCESS_KEY = os.environ.get("HANDS_ACCESS_KEY", "").strip()
        if not ACCESS_KEY:
            kf = data_dir() / "access_key"
            try:
                ACCESS_KEY = kf.read_text().strip()
            except OSError:
                import secrets as _s
                ACCESS_KEY = _s.token_urlsafe(18)
                data_dir().mkdir(parents=True, exist_ok=True)
                kf.write_text(ACCESS_KEY)
                try:
                    kf.chmod(0o600)
                except OSError:
                    pass
        cfg.setdefault("approve", "auto")
        save_config(cfg)  # so /hands/bootstrap and the restart path see the same setup
    default_root = (data_dir() / "workspace") if CLOUD else (Path.home() / "Hands")
    roots = a.root or ([] if CLOUD else cfg.get("roots")) or [str(default_root)]
    for r in roots:
        try:
            Path(r).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    roots = [str(Path(r).expanduser()) for r in roots]
    approve = a.approve or cfg.get("approve") or "auto"
    has_brain = apply_brain_env(cfg)

    ui_dir = a.ui_dir or next((str(p) for p in (ROOT / "ui", ROOT / "desktop" / "dist") if (p / "index.html").exists()),
                              str(ROOT / "ui"))
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("hands.app")
    log.info("brain: %s", normalise_brain_url(cfg.get("brain_url", "")) if has_brain else "NOT SET (open the app and add your Persona URL + key)")

    import stitch

    def extra_routes(**kw):
        return build_routes(**kw, port=a.port, roots=roots, approve=approve, ui_dir=ui_dir)

    ns = SimpleNamespace(
        port=a.port, host=("0.0.0.0" if (cfg.get("lan") or CLOUD) else "127.0.0.1"), place=("cloud" if CLOUD else "computer"), root=roots, approve=approve, deepinfra_key="",
        persona_key="", data_dir=str(data_dir()), agent=[], no_voice=a.no_voice,
        extra_routes=extra_routes,
    )
    url = f"http://127.0.0.1:{a.port}/ui/"
    if not a.no_open:
        def opener():
            import urllib.request
            for _ in range(120):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{a.port}/hands/bootstrap", timeout=1)
                    break
                except Exception as exc:  # noqa: BLE001 - 403 means it is up
                    if getattr(exc, "code", 0):
                        break
                    time.sleep(0.5)
            _open_window(url)
        threading.Thread(target=opener, daemon=True).start()
    print(f"Hands: {url}")
    asyncio.run(stitch.run_local(ns))


if __name__ == "__main__":
    main()
