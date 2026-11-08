"""Smoke test: --computer-use degrades gracefully, and works when OCU is there.

    python scripts/ocu_smoke_test.py

Three things, all offline:

1. MISSING ``ocu_mcp.py``  - ``load_ocu()`` returns (None, reason) and a
   ``Hands(..., computer_use=True)`` still builds, still advertises the six
   file/shell tools, and a ``ocu_*`` call comes back as a clean JSON error.
   This is the crash the old code had: ``spec`` was None and
   ``spec.loader.exec_module`` blew up before the process could connect.
2. PRESENT ``ocu_mcp.py``  - ``ocu_*`` tools show up in ``hands.tools()``. Uses
   the stub MCP server at ``scripts/ocu_fixture_server.py`` through
   ``SAWII_OCU_COMMAND``, so no real screen or mouse is touched.
3. BROKEN ``ocu_mcp.py``   - a file that raises on import also degrades instead
   of killing the engine.

Prints a PASS/FAIL line per check and exits non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sawii_hands  # noqa: E402  (the engine under test)

FIXTURE = ROOT / "scripts" / "ocu_fixture_server.py"
BASE = {t["function"]["name"] for t in sawii_hands.TOOLS}
WANT = {"ocu_list_apps", "ocu_get_app_state", "ocu_click", "ocu_type_text",
        "ocu_scroll", "ocu_keypress"}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    from sawii_hands import Hands, load_ocu

    ok = True
    print("ocu smoke test")

    # ── 1. missing file: graceful, not a crash ────────────────────────────
    print("\n1. ocu_mcp.py missing  (the old crash path)")
    empty = Path(tempfile.mkdtemp(prefix="ocu_missing_"))
    (empty / "sawii_hands.py").write_text("# placeholder so the message is sane\n", encoding="utf-8")
    ocu, err = load_ocu(str(empty / "ocu_mcp.py"))
    ok &= check("load_ocu returns (None, reason) instead of raising",
                ocu is None and bool(err), f"reason={err[:70]!r}")

    saved = None
    real = ROOT / "ocu_mcp.py"
    try:  # temporarily hide the real file so the engine itself takes that path
        saved = real.with_suffix(".py.hidden-for-smoke")
        real.rename(saved)
        h = Hands([str(empty)], approve="auto", computer_use=True)
        names = {t["function"]["name"] for t in h.tools()}
        ok &= check("Hands(computer_use=True) still constructs", h.ocu is None)
        ok &= check("advertises the 6 file/shell tools anyway",
                    {"shell", "read_file", "write_file", "edit_file", "list_files",
                     "search_files"} <= names and not any(n.startswith("ocu_") for n in names),
                    f"{len(names)} tools")
        out = json.loads(h.run("ocu_get_app_state", {"app": "notepad"}))
        ok &= check("an ocu_* call returns a clean error, no exception",
                    "error" in out and "unavailable" in out["error"], str(out)[:90])
    finally:
        if saved and saved.is_file():
            saved.rename(real)

    # ── 2. present file: ocu_* tools appear ───────────────────────────────
    print("\n2. ocu_mcp.py present  (stub MCP server, no real screen)")
    if not FIXTURE.is_file():
        check("fixture server exists", False, str(FIXTURE))
        print("\nOCU SMOKE FAILED")
        return 1
    os.environ["SAWII_OCU_COMMAND"] = f'"{sys.executable}" "{FIXTURE}"'
    try:
        h = Hands([str(empty)], approve="auto", computer_use=True)
        got = {t["function"]["name"] for t in h.tools()}
        ok &= check("ocu_* tools appear in hands.tools()", WANT <= got,
                    f"+{len(got - BASE)} gui tools advertised")
        schemas = {t["function"]["name"]: t["function"] for t in h.tools()}
        ok &= check("schemas are OpenAI function form with parameters",
                    all(schemas[n].get("parameters", {}).get("type") == "object"
                        and schemas[n].get("description")
                        for n in got if n.startswith("ocu_")))

        state = json.loads(h.run("ocu_get_app_state", {"app": "notepad.exe"}))
        text = " ".join(b.get("text", "") for b in state.get("result", state).get("content", [])
                        if isinstance(b, dict))
        ok &= check("real call->result round trip over ocu_get_app_state",
                    "[7] Button Save" in text or "[7] Button Save" in json.dumps(state),
                    text[:60])
        click = json.loads(h.run("ocu_click", {"app": "notepad.exe", "element_index": "7"}))
        ok &= check("ocu_click after reading state succeeds",
                    "clicked 7" in json.dumps(click), json.dumps(click)[:70])
        denied = json.loads(h.run("ocu_click", {"app": "calc.exe", "element_index": "7"}))
        ok &= check("click without reading that app's state is refused",
                    "isError" in json.dumps(denied), json.dumps(denied)[:70])
        ok &= check("MCP child booted", h.ocu is not None)
        if h.ocu is not None:
            h.ocu.close()
            ok &= check("close() shuts the MCP child down", h.ocu.process is None)
    finally:
        os.environ.pop("SAWII_OCU_COMMAND", None)

    # ── 3. broken file: also graceful ─────────────────────────────────────
    print("\n3. ocu_mcp.py broken  (raises on import)")
    bad = Path(tempfile.mkdtemp(prefix="ocu_broken_")) / "ocu_mcp.py"
    bad.write_text("raise RuntimeError('simulated broken module')\n", encoding="utf-8")
    ocu, err = load_ocu(str(bad))
    ok &= check("broken module degrades with a reason", ocu is None and "simulated" in err, err[:70])

    # A present-but-unreachable server (no npm install) must also degrade.
    missing_server = ROOT / "scripts" / "does_not_exist.py"
    os.environ["SAWII_OCU_COMMAND"] = f'"{sys.executable}" "{missing_server}"'
    try:
        h = Hands([str(empty)], approve="auto", computer_use=True)
        names = {t["function"]["name"] for t in h.tools()}
        ok &= check("unreachable open-computer-use server still boots the engine",
                    {"shell", "read_file"} <= names and not any(n.startswith("ocu_") for n in names),
                    h.ocu_error[:70])
    finally:
        os.environ.pop("SAWII_OCU_COMMAND", None)

    print("\nOCU SMOKE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())