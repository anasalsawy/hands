"""Why does `import piper` hang? Each suspect imported in a child process that
can be killed, so the diagnostic itself can never wedge.

piper_resident.py measured a real 588 ms p50 an hour earlier, then `import
piper` began hanging forever with no output at all. Something in the
environment changed underneath this working copy.

Run: python persona/voice/bench/piper_import_diag.py
"""
from __future__ import annotations

import json
import subprocess
import sys

TIMEOUT_S = 40

SUSPECTS = [
    ("onnxruntime", "import onnxruntime; print(onnxruntime.__version__)"),
    ("piper", "import piper; print('piper ok')"),
    ("piper_voice", "from piper import PiperVoice; print('PiperVoice ok')"),
]


def probe(name: str, code: str) -> dict:
    try:
        p = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True,
                           text=True, timeout=TIMEOUT_S)
        return {"mod": name, "rc": p.returncode,
                "out": (p.stdout or "").strip()[:80],
                "err": (p.stderr or "").strip().splitlines()[-1][:110] if p.stderr else ""}
    except subprocess.TimeoutExpired:
        return {"mod": name, "rc": "TIMEOUT", "note": f"no return in {TIMEOUT_S}s"}
    except Exception as exc:  # noqa: BLE001
        return {"mod": name, "rc": "ERR", "note": type(exc).__name__}


def main() -> None:
    import importlib.metadata as md

    pkgs = {}
    for dist in ("piper-tts", "piper", "onnxruntime", "numpy"):
        try:
            pkgs[dist] = md.version(dist)
        except Exception:  # noqa: BLE001
            pkgs[dist] = None
    print(json.dumps({"event": "packages", **pkgs}), flush=True)
    for name, code in SUSPECTS:
        print(json.dumps(probe(name, code)), flush=True)


if __name__ == "__main__":
    main()