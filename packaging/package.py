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
APP_FILES = ["hands_app.py", "stitch.py", "sawii_hands.py", "winhand.py", "webtools.py", "ocu_mcp.py", "voice_sidecar.py", "requirements.txt"]
SKIP_DIRS = {"__pycache__", ".git"}


ENGINE_KEEP = ["persona-loop.ts", "package.json", "package-lock.json", "PROVENANCE.md",
               "ui-tars-desktop/multimodal/omni-tars/core"]


def copy_engine(app: Path) -> None:
    """The Omni-TARS loop: persona-loop.ts + the slice of UI-TARS it imports."""
    src = ROOT / "engine"
    for rel in ENGINE_KEEP:
        a, b = src / rel, app / "engine" / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        if a.is_dir():
            shutil.copytree(a, b, ignore=lambda _d, n: [x for x in n if x == "node_modules"])
        else:
            shutil.copy2(a, b)


def copy_app(dest: Path) -> None:
    app = dest / "app"
    app.mkdir(parents=True)
    for f in APP_FILES:
        shutil.copy2(ROOT / f, app / f)

    def ignore(_d, names):
        return [n for n in names if n in SKIP_DIRS or n.endswith(".pyc")]

    shutil.copytree(ROOT / "persona", app / "persona", ignore=ignore)
    copy_engine(app)
    shutil.copytree(ROOT / "vendor", app / "vendor")
    (app / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts" / "install_browser_sdk.py", app / "scripts" / "install_browser_sdk.py")
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


def install_browser_sdk(site: Path, workdir: Path) -> None:
    """Pinned, patched Browser-Use (what the Browser cards drive), installed into `site`."""
    import hashlib
    import json

    vendor = ROOT / "vendor" / "browser-use"
    source = json.loads((vendor / "SOURCE.json").read_text())
    for name, expected in source["patches"].items():
        if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != expected:
            sys.exit(f"Browser SDK patch integrity failure: {name}")
    dest = workdir / "browser-use-sdk"
    git = ["git", "-c", "core.autocrlf=false"]
    subprocess.run(git + ["clone", "--filter=blob:none", "--no-checkout",
                          "https://github.com/browser-use/browser-use.git", str(dest)], check=True)
    subprocess.run(git + ["-C", str(dest), "checkout", "--detach", source["upstream_commit"]], check=True)
    for name in source["patches"]:
        subprocess.run(git + ["-C", str(dest), "apply", str(vendor / name)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", str(site),
                    str(ROOT / "vendor" / "persona-lanes"), str(dest)], check=True)


README = """Hands
=====
Persona's hands on your computer: type a goal, Persona splits it into parallel
lines of work, and every line is a live card driving this machine.

1. Start Hands.  First run asks for your Persona URL and key (the key of the
   persona you want to think for the cards) and the folders the hands may touch.
2. Type a goal on the Courts tab and press Run.

Screen / mouse / keyboard control needs Node.js and the open-computer-use helper:
    npm install -g open-computer-use
(Hands.bat tries this for you when Node is installed.) Without it the cards can
still read and write files inside your folders.

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
        "where open-computer-use >nul 2>nul || (where npm >nul 2>nul && call npm install -g open-computer-use)\r\n"
        "if not exist app\\engine\\node_modules (where npm >nul 2>nul && call npm ci --prefix app\\engine)\r\n"
        ":loop\r\n.venv\\Scripts\\python app\\hands_app.py --ui-dir ui %*\r\n"
        "if %errorlevel%==75 goto loop\r\n", encoding="utf-8")
    sh = tmp / "hands.sh"
    sh.write_text(
        "#!/usr/bin/env sh\ncd \"$(dirname \"$0\")\"\nexport HANDS_SUPERVISED=1\n"
        "[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r app/requirements.txt; }\n"
        "command -v open-computer-use >/dev/null 2>&1 || { command -v npm >/dev/null 2>&1 && npm install -g open-computer-use; }\n"
        "[ -d app/engine/node_modules ] || { command -v npm >/dev/null 2>&1 && npm ci --prefix app/engine; }\n"
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
    subprocess.run([npm, "ci", "--prefix", str(tmp / "app" / "engine"), "--no-audit", "--no-fund"], check=True, env=env)
    subprocess.run([npm, "install", "--prefix", str(tmp / "tools"), "open-computer-use", "--no-audit", "--no-fund"],
                   check=True, env=env)
    install_browser_sdk(site, tmp)
    (tmp / "README.txt").write_text(README, encoding="utf-8")
    (tmp / "Hands.bat").write_text(
        "@echo off\r\nsetlocal\r\ncd /d \"%~dp0\"\r\nset HANDS_SUPERVISED=1\r\n"
        "set PATH=%~dp0node;%~dp0tools\\node_modules\\.bin;%PATH%\r\n"
        ":loop\r\npython\\python.exe app\\hands_app.py --ui-dir ui %*\r\n"
        "if %errorlevel%==75 goto loop\r\n", encoding="utf-8")
    # smoke: the bundled interpreter must import the app's own modules
    subprocess.run([str(py / "python.exe"), "-c",
                    "import sys; sys.path.insert(0, r'%s'); import starlette, uvicorn, httpx, websockets, numpy, mss, PIL; "
                    "import persona.api.app, browser_use.persona.agent; print('bundle imports ok')" % (tmp / "app")], check=True)
    zip_dir(tmp, "Hands-windows.zip")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "portable"
    {"portable": build_portable, "windows": build_windows}[which]()
