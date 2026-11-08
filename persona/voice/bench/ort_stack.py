"""Dump the stack of a hanging `import onnxruntime`, then give up cleanly.

faulthandler prints the C-level Python stack of the stuck frame after N seconds
without the process having to return on its own. Run in a child so this file can
never wedge a session.

Run: python persona/voice/bench/ort_stack.py
"""
from __future__ import annotations

import subprocess
import sys

CODE = (
    "import faulthandler, time, os\n"
    "faulthandler.dump_traceback_later(45, exit=True)\n"
    "print('importing onnxruntime pid', os.getpid(), flush=True)\n"
    "t=time.perf_counter()\n"
    "import onnxruntime\n"
    "print('IMPORT OK', round(time.perf_counter()-t,1), flush=True)\n"
)


def main() -> None:
    try:
        p = subprocess.run([sys.executable, "-u", "-c", CODE], capture_output=True,
                           text=True, timeout=110)
        print("rc", p.returncode)
        print((p.stdout or "").strip()[:400])
        err = (p.stderr or "").strip().splitlines()
        for line in err[-30:]:
            print(line[:140])
    except subprocess.TimeoutExpired:
        print("TIMEOUT: child never returned even after dump_traceback_later(exit=True)")


if __name__ == "__main__":
    main()