#!/usr/bin/env python3
"""Build the downloadable Hands bundles.

  python packaging/package.py portable   # Hands-portable.zip  (needs Python 3.10+ on the machine)
  python packaging/package.py windows    # Hands-windows.zip   (bundles its own Python; run on Windows)

Both expect the UI to be built first:  (cd desktop && npm ci && npm run build)
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist-release"
PY_EMBED = "3.12.8"
APP_FILES = ["hands_app.py", "stitch.py", "voice_sidecar.py", "requirements.txt"]
SKIP_DIRS = {"__pycache__", ".git"}


def copy_engine(app: Path) -> None:
    """Agent TARS (the vendored upstream tree) + Persona's engine entry point.
    Dependencies and the TARS build are produced in the bundle by `build_engine`."""
    shutil.copytree(ROOT / "engine", app / "engine",
                    ignore=lambda _d, n: [x for x in n if x in {"node_modules", "dist", "images"}])


def build_engine(app: Path, npm: str, env: dict) -> None:
    """Install and build the real Agent TARS core inside the bundle, then check it is there."""
    subprocess.run([npm, "ci", "--prefix", str(app / "engine"), "--no-audit", "--no-fund"], check=True, env=env)
    subprocess.run([npm, "run", "build:tars", "--prefix", str(app / "engine")], check=True, env=env)
    dist = app / "engine/ui-tars-desktop/multimodal/agent-tars/core/dist/index.js"
    if not dist.exists():
        sys.exit("Agent TARS build failed: " + str(dist))


def copy_app(dest: Path) -> None:
    app = dest / "app"
    app.mkdir(parents=True)
    for f in APP_FILES:
        shutil.copy2(ROOT / f, app / f)

    def ignore(_d, names):
        return [n for n in names if n in SKIP_DIRS or n.endswith(".pyc")]

    shutil.copytree(ROOT / "persona", app / "persona", ignore=ignore)
    copy_engine(app)
    ui = ROOT / "desktop" / "dist"
    if not (ui / "index.html").exists():
        sys.exit("UI not built: run  (cd desktop && npm ci && npm run build)  first")
    shutil.copytree(ui, dest / "ui")


def zip_dir(src: Path, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    target = OUT / name
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, Path("Hands") / p.relative_to(src))
    print(f"wrote {target} ({target.stat().st_size/1e6:.1f} MB)")
    return target


NODE_VERSION = "22.12.0"


README = """Hands
=====
Persona's hands on your computer: type a goal, Persona splits it into parallel
lines of work, and every line is a live card driving this machine.

1. Start Hands.  First run asks for your Persona URL and key (the key of the
   persona you want to think for the cards) and the folders the hands may touch.
2. Type a goal on the Courts tab and press Run.

Every card runs on Agent TARS (browser, files, shell, MCP), which needs Node.js 22+.
The first start builds it (about 10 minutes, once); the windows bundle ships it built.

Settings live in %USERPROFILE%\\.hands\\config.json (Windows) or ~/.hands/config.json.
"""


def build_portable() -> None:
    tmp = ROOT / "build-portable"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    copy_app(tmp)
    (tmp / "README.txt").write_text(README, encoding="utf-8")
    (tmp / "Hands.bat").write_text(
        "@echo off\r\nsetlocal\r\ncd /d \"%~dp0\"\r\nset HANDS_SUPERVISED=1\r\n"
        "if not exist .venv (\r\n  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)\r\n"
        "  .venv\\Scripts\\python -m pip install -q -r app\\requirements.txt\r\n)\r\n"
        "if not exist app\\engine\\ui-tars-desktop\\multimodal\\agent-tars\\core\\dist\\index.js (call npm ci --prefix app\\engine && call npm run build:tars --prefix app\\engine)\r\n"
        ":loop\r\n.venv\\Scripts\\python app\\hands_app.py --ui-dir ui %*\r\n"
        "if %errorlevel%==75 goto loop\r\n", encoding="utf-8")
    sh = tmp / "hands.sh"
    sh.write_text(
        "#!/usr/bin/env sh\ncd \"$(dirname \"$0\")\"\nexport HANDS_SUPERVISED=1\n"
        "[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r app/requirements.txt; }\n"
        "[ -f app/engine/ui-tars-desktop/multimodal/agent-tars/core/dist/index.js ] || { npm ci --prefix app/engine && npm run build:tars --prefix app/engine; }\n"
        "while :; do .venv/bin/python app/hands_app.py --ui-dir ui \"$@\"; [ $? -eq 75 ] || break; done\n",
        encoding="utf-8")
    sh.chmod(0o755)
    zip_dir(tmp, "Hands-portable.zip")
    shutil.rmtree(tmp, ignore_errors=True)


def build_windows() -> None:
    if sys.platform != "win32":
        sys.exit("the windows bundle must be built on Windows (it installs Windows wheels)")
    tmp = ROOT / "build-windows"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    copy_app(tmp)
    py = tmp / "python"
    url = f"https://www.python.org/ftp/python/{PY_EMBED}/python-{PY_EMBED}-embed-amd64.zip"
    print("downloading", url)
    data = urllib.request.urlopen(url, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(py)
    site = py / "Lib" / "site-packages"
    site.mkdir(parents=True)
    pth = next(py.glob("python*._pth"))
    pth.write_text(f"{pth.stem.replace('._pth','')}.zip\n.\nLib\\site-packages\n..\\app\nimport site\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:",
                    "--target", str(site), "-r", str(ROOT / "requirements.txt")], check=True)
    # Node 22 (runs the Omni-TARS loop), the loop's dependencies, and the screen-control helper
    nurl = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-win-x64.zip"
    print("downloading", nurl)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(nurl, timeout=300).read())).extractall(tmp)
    (tmp / f"node-v{NODE_VERSION}-win-x64").rename(tmp / "node")
    env = {**__import__("os").environ, "PATH": f"{tmp / 'node'};" + __import__("os").environ["PATH"]}
    npm = str(tmp / "node" / "npm.cmd")
    build_engine(tmp / "app", npm, env)
    (tmp / "README.txt").write_text(README, encoding="utf-8")
    (tmp / "Hands.bat").write_text(
        "@echo off\r\nsetlocal\r\ncd /d \"%~dp0\"\r\nset HANDS_SUPERVISED=1\r\n"
        "set PATH=%~dp0node;%PATH%\r\n"
        ":loop\r\npython\\python.exe app\\hands_app.py --ui-dir ui %*\r\n"
        "if %errorlevel%==75 goto loop\r\n", encoding="utf-8")
    # smoke: the bundled interpreter must import the app's own modules
    subprocess.run([str(py / "python.exe"), "-c",
                    "import sys; sys.path.insert(0, r'%s'); import starlette, uvicorn, httpx, websockets, numpy, mss, PIL; "
                    "import persona.api.app, persona.tars_engine; print('bundle imports ok')" % (tmp / "app")], check=True)
    zip_dir(tmp, "Hands-windows.zip")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "portable"
    {"portable": build_portable, "windows": build_windows}[which]()
