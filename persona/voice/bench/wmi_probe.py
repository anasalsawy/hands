"""Is the machine's WMI query path stuck right now?

`import onnxruntime` hangs at platform.uname() -> win32_ver() -> _wmi_query().
If a bare stdlib call to the same function also blocks, the fault is the machine
(WinMGMT), not our code and not onnxruntime. Runs in a killable child so this
diagnostic can never wedge a session.

Run: python persona/voice/bench/wmi_probe.py
"""
from __future__ import annotations

import subprocess
import sys

CODE = (
    "import faulthandler, time, platform\n"
    "faulthandler.dump_traceback_later(30, exit=True)\n"
    "t = time.perf_counter()\n"
    "print('win32_ver:', platform.win32_ver(), flush=True)\n"
    "print('WMI OK in', round(time.perf_counter() - t, 2), 's', flush=True)\n"
)


def main() -> None:
    try:
        p = subprocess.run([sys.executable, "-u", "-c", CODE], capture_output=True,
                           text=True, timeout=90)
        print("rc", p.returncode)
        print((p.stdout or "").strip()[:200])
        err = (p.stderr or "").strip().splitlines()
        for line in err[-6:]:
            print(line[:120])
    except subprocess.TimeoutExpired:
        print("TIMEOUT: stdlib platform.win32_ver() itself blocks -> machine WMI")


if __name__ == "__main__":
    main()