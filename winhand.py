"""Window hands: fast, window-bound computer use for one R lane.

Several lanes share one computer. Each lane's engine owns exactly one app window:

  * it SEES only that window (its own pixels, even when other windows cover it),
  * it ACTS only on that window, in the background where the app allows it
    (posted window messages: no cursor, no focus steal, so lanes run at once),
  * every action answers with a fresh screenshot of that window, so the model
    never pays a second round trip just to look.

Apps that ignore posted input (Chromium shells, UWP/XAML) are retried in the
foreground. That retry takes a cross-process input lock, so two lanes never fight
over the one real mouse; lanes that use background input never take it.

Coordinates are in the pixels of the screenshot the model was given (top-left of
the window = 0,0). The scale back to the real window is applied here.

Backends: Windows (ctypes, no extra dependency but Pillow) and X11 (python-xlib;
used to test the lane logic on Linux). Nothing here talks to a model.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

SHOT_MAX = int(os.environ.get("HANDS_SHOT_MAX", "1280"))
SHOT_QUALITY = int(os.environ.get("HANDS_SHOT_QUALITY", "70"))


def state_dir() -> Path:
    d = Path(os.environ.get("HANDS_DATA_DIR") or (Path.home() / ".hands"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Win:
    id: int
    title: str
    pid: int
    left: int
    top: int
    width: int
    height: int
    app: str = ""

    def public(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- file locks
@contextlib.contextmanager
def _flock(path: Path, timeout: float = 30.0):
    """Cross-process exclusive lock on a file (msvcrt on Windows, fcntl elsewhere)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not take {path.name}")
                time.sleep(0.01)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


class LaneWindows:
    """Which lane owns which window. Shared by every engine and read by the app,
    so a card can show exactly its lane's window and nothing else."""

    def __init__(self, path: Path | None = None):
        self.path = path or (state_dir() / "lane_windows.json")
        self.lock = self.path.with_suffix(".lock")

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def all(self) -> dict:
        return self._read()

    def get(self, lane: str) -> dict | None:
        return self._read().get(lane)

    def bind(self, lane: str, win: Win) -> None:
        with _flock(self.lock):
            data = self._read()
            for other, row in data.items():
                if other != lane and int(row.get("id", -1)) == win.id:
                    raise PermissionError(f"that window already belongs to lane {other}")
            data[lane] = {**win.public(), "at": time.time()}
            self._write(data)

    def release(self, lane: str) -> None:
        with _flock(self.lock):
            data = self._read()
            if data.pop(lane, None) is not None:
                self._write(data)


# ------------------------------------------------------------------- imaging
def encode_shot(img, max_side: int = SHOT_MAX) -> tuple[bytes, float]:
    """JPEG of the window, downscaled so its longest side is <= max_side.
    Returns (bytes, scale) where scale = shot_pixels / real_pixels."""
    from PIL import Image
    w, h = img.size
    scale = min(1.0, max_side / max(w, h)) if max_side else 1.0
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=SHOT_QUALITY, optimize=False)
    return buf.getvalue(), scale


# ============================================================ Windows backend
class Win32Backend:
    name = "win32"

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        u = self.user32
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        u.GetDC.argtypes = [wintypes.HWND]
        u.GetDC.restype = wintypes.HDC
        u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        self.gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        self.gdi32.CreateCompatibleDC.restype = wintypes.HDC
        self.gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        self.gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        self.gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        self.gdi32.SelectObject.restype = wintypes.HGDIOBJ
        self.gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        self.gdi32.DeleteDC.argtypes = [wintypes.HDC]
        try:  # real pixel coordinates, not DPI-virtualised ones
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:  # noqa: BLE001
            with contextlib.suppress(Exception):
                u.SetProcessDPIAware()
        self._last_child: dict[int, int] = {}

    # ---- windows
    def _info(self, hwnd: int) -> Win | None:
        ct, wt, u = self.ct, self.wt, self.user32
        if not u.IsWindowVisible(hwnd):
            return None
        n = u.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return None
        buf = ct.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        ex = u.GetWindowLongW(hwnd, -20)
        if ex & 0x80:  # WS_EX_TOOLWINDOW
            return None
        cloaked = ct.c_int(0)
        with contextlib.suppress(Exception):
            ct.windll.dwmapi.DwmGetWindowAttribute(wt.HWND(hwnd), 14, ct.byref(cloaked), 4)
        if cloaked.value:
            return None
        r = wt.RECT()
        u.GetWindowRect(hwnd, ct.byref(r))
        pid = wt.DWORD()
        u.GetWindowThreadProcessId(hwnd, ct.byref(pid))
        return Win(int(hwnd), buf.value, int(pid.value), r.left, r.top,
                   max(0, r.right - r.left), max(0, r.bottom - r.top), self._exe(int(pid.value)))

    def _exe(self, pid: int) -> str:
        h = self.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            buf = self.ct.create_unicode_buffer(520)
            size = self.wt.DWORD(520)
            if self.kernel32.QueryFullProcessImageNameW(h, 0, buf, self.ct.byref(size)):
                return os.path.basename(buf.value)
        finally:
            self.kernel32.CloseHandle(h)
        return ""

    def list_windows(self) -> list[Win]:
        out: list[Win] = []
        proc = self.ct.WINFUNCTYPE(self.ct.c_bool, self.wt.HWND, self.wt.LPARAM)

        def cb(hwnd, _lp):
            w = self._info(int(hwnd))
            if w and w.width > 1 and w.height > 1:
                out.append(w)
            return True

        self.user32.EnumWindows(proc(cb), 0)
        return out

    def window(self, wid: int) -> Win | None:
        if not self.user32.IsWindow(wid):
            return None
        return self._info(wid) or Win(wid, "", 0, 0, 0, 0, 0)

    # ---- capture: the window's own pixels, even when covered
    def capture(self, wid: int):
        from PIL import Image
        ct, wt, u, g = self.ct, self.wt, self.user32, self.gdi32
        if u.IsIconic(wid):
            u.ShowWindow(wid, 4)  # SW_SHOWNOACTIVATE: render it without taking focus
            time.sleep(0.08)
        r = wt.RECT()
        u.GetWindowRect(wid, ct.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        if w <= 0 or h <= 0:
            raise RuntimeError("window has no size")
        hdc = u.GetDC(0)
        mem = g.CreateCompatibleDC(hdc)
        bmp = g.CreateCompatibleBitmap(hdc, w, h)
        old = g.SelectObject(mem, bmp)
        try:
            ok = u.PrintWindow(wid, mem, 2)  # PW_RENDERFULLCONTENT
            if not ok:
                ok = u.PrintWindow(wid, mem, 0)
            if not ok:
                raise RuntimeError("PrintWindow refused this window")

            class BMIH(ct.Structure):
                _fields_ = [("biSize", wt.DWORD), ("biWidth", ct.c_long), ("biHeight", ct.c_long),
                            ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                            ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", ct.c_long),
                            ("biYPelsPerMeter", ct.c_long), ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]

            bi = BMIH(ct.sizeof(BMIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
            buf = ct.create_string_buffer(w * h * 4)
            g.GetDIBits(mem, bmp, 0, h, buf, ct.byref(bi), 0)
            return Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1)
        finally:
            g.SelectObject(mem, old)
            g.DeleteObject(bmp)
            g.DeleteDC(mem)
            u.ReleaseDC(0, hdc)

    # ---- background input (posted messages: no cursor, no focus)
    def _client_origin(self, wid: int) -> tuple[int, int]:
        ct, wt = self.ct, self.wt
        r = wt.RECT()
        self.user32.GetWindowRect(wid, ct.byref(r))
        p = wt.POINT(0, 0)
        self.user32.ClientToScreen(wid, ct.byref(p))
        return p.x - r.left, p.y - r.top

    def _target(self, wid: int, x: int, y: int) -> tuple[int, int, int]:
        """Deepest child control under window-relative (x,y); returns (hwnd, cx, cy) in its client space."""
        ct, wt, u = self.ct, self.wt, self.user32
        ox, oy = self._client_origin(wid)
        hwnd, cx, cy = wid, x - ox, y - oy
        for _ in range(8):
            child = u.ChildWindowFromPointEx(hwnd, wt.POINT(cx, cy), 0x0001 | 0x0002 | 0x0004)
            if not child or child == hwnd:
                break
            p = wt.POINT(cx, cy)
            u.MapWindowPoints(hwnd, child, ct.byref(p), 1)
            hwnd, cx, cy = int(child), p.x, p.y
        self._last_child[wid] = hwnd
        return hwnd, cx, cy

    @staticmethod
    def _lp(x: int, y: int) -> int:
        return ((y & 0xFFFF) << 16) | (x & 0xFFFF)

    def bg_click(self, wid: int, x: int, y: int, button: str = "left", count: int = 1) -> None:
        hwnd, cx, cy = self._target(wid, x, y)
        down, up, mk = {"left": (0x201, 0x202, 0x1), "right": (0x204, 0x205, 0x2),
                        "middle": (0x207, 0x208, 0x10)}[button]
        p = self.user32.PostMessageW
        lp = self._lp(cx, cy)
        p(hwnd, 0x0200, 0, lp)  # WM_MOUSEMOVE
        for i in range(count):
            p(hwnd, down if i == 0 else {0x201: 0x203, 0x204: 0x206, 0x207: 0x209}[down], mk, lp)
            p(hwnd, up, 0, lp)

    def bg_drag(self, wid: int, x1: int, y1: int, x2: int, y2: int) -> None:
        hwnd, cx, cy = self._target(wid, x1, y1)
        _, ex, ey = self._target(wid, x2, y2)
        p = self.user32.PostMessageW
        p(hwnd, 0x0201, 0x1, self._lp(cx, cy))
        steps = 8
        for i in range(1, steps + 1):
            p(hwnd, 0x0200, 0x1, self._lp(cx + (ex - cx) * i // steps, cy + (ey - cy) * i // steps))
        p(hwnd, 0x0202, 0, self._lp(ex, ey))

    def bg_type(self, wid: int, text: str) -> None:
        hwnd = self._last_child.get(wid, wid)
        for ch in text:
            data = ch.encode("utf-16-le")
            for i in range(0, len(data), 2):
                self.user32.PostMessageW(hwnd, 0x0102, int.from_bytes(data[i:i + 2], "little"), 0)  # WM_CHAR

    def bg_scroll(self, wid: int, x: int, y: int, dy: int) -> None:
        hwnd, _, _ = self._target(wid, x, y)
        r = self.wt.RECT()
        self.user32.GetWindowRect(wid, self.ct.byref(r))
        delta = (-dy * 120) & 0xFFFF
        self.user32.PostMessageW(hwnd, 0x020A, delta << 16, self._lp(r.left + x, r.top + y))  # WM_MOUSEWHEEL

    # ---- foreground input (the retry path; callers hold the input lock)
    def _vk(self, name: str) -> int:
        names = {"ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "enter": 0x0D,
                 "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B, "space": 0x20, "backspace": 0x08,
                 "delete": 0x2E, "del": 0x2E, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
                 "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28}
        n = name.lower()
        if n in names:
            return names[n]
        if len(n) == 2 and n[0] == "f" and n[1].isdigit():
            return 0x70 + int(n[1]) - 1
        if n.startswith("f") and n[1:].isdigit() and 1 <= int(n[1:]) <= 12:
            return 0x70 + int(n[1:]) - 1
        if len(n) == 1:
            return self.user32.VkKeyScanW(ord(n)) & 0xFF
        raise ValueError(f"unknown key {name!r}")

    def _send(self, inputs) -> None:
        ct, wt = self.ct, self.wt

        class KI(ct.Structure):
            _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                        ("dwExtraInfo", ct.c_size_t)]

        class MI(ct.Structure):
            _fields_ = [("dx", ct.c_long), ("dy", ct.c_long), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                        ("time", wt.DWORD), ("dwExtraInfo", ct.c_size_t)]

        class U(ct.Union):
            _fields_ = [("ki", KI), ("mi", MI)]

        class INPUT(ct.Structure):
            _fields_ = [("type", wt.DWORD), ("u", U)]

        arr = (INPUT * len(inputs))()
        for i, (kind, a, b, c) in enumerate(inputs):
            arr[i].type = kind
            if kind == 1:
                arr[i].u.ki = KI(a, 0, b, 0, 0)
            else:
                arr[i].u.mi = MI(0, 0, c, b, 0, 0)
        self.user32.SendInput(len(inputs), arr, ct.sizeof(INPUT))

    def _to_front(self, wid: int):
        u = self.user32
        prev = u.GetForegroundWindow()
        if u.IsIconic(wid):
            u.ShowWindow(wid, 9)
        # a synthetic Alt tap lets this process take the foreground
        self._send([(1, 0x12, 0, 0), (1, 0x12, 2, 0)])
        u.SetForegroundWindow(wid)
        time.sleep(0.03)
        return prev

    def fg_click(self, wid: int, x: int, y: int, button: str = "left", count: int = 1) -> None:
        u, wt, ct = self.user32, self.wt, self.ct
        prev = self._to_front(wid)
        pt = wt.POINT()
        u.GetCursorPos(ct.byref(pt))
        r = wt.RECT()
        u.GetWindowRect(wid, ct.byref(r))
        u.SetCursorPos(r.left + x, r.top + y)
        down, up = {"left": (0x2, 0x4), "right": (0x8, 0x10), "middle": (0x20, 0x40)}[button]
        for _ in range(count):
            self._send([(0, 0, down, 0), (0, 0, up, 0)])
        time.sleep(0.02)
        u.SetCursorPos(pt.x, pt.y)
        if prev and prev != wid:
            u.SetForegroundWindow(prev)

    def fg_type(self, wid: int, text: str) -> None:
        prev = self._to_front(wid)
        for ch in text:
            code = ord(ch)
            self._send([(1, code, 0x4, 0), (1, code, 0x4 | 0x2, 0)])  # KEYEVENTF_UNICODE
        if prev and prev != wid:
            self.user32.SetForegroundWindow(prev)

    def fg_hotkey(self, wid: int, keys: list[str]) -> None:
        prev = self._to_front(wid)
        vks = [self._vk(k) for k in keys]
        self._send([(1, v, 0, 0) for v in vks] + [(1, v, 2, 0) for v in reversed(vks)])
        time.sleep(0.02)
        if prev and prev != wid:
            self.user32.SetForegroundWindow(prev)

    def fg_scroll(self, wid: int, x: int, y: int, dy: int) -> None:
        prev = self._to_front(wid)
        r = self.wt.RECT()
        self.user32.GetWindowRect(wid, self.ct.byref(r))
        self.user32.SetCursorPos(r.left + x, r.top + y)
        self._send([(0, 0, 0x800, (-dy * 120) & 0xFFFFFFFF)])
        if prev and prev != wid:
            self.user32.SetForegroundWindow(prev)

    def launch(self, command: str) -> None:
        subprocess.Popen(command, shell=True, creationflags=0x00000008 if os.name == "nt" else 0)


# ================================================================ X11 backend
class X11Backend:
    """Linux/X11 backend. Input is XSendEvent straight to the window (no pointer,
    no focus), which is what lets the lane logic be tested with real windows."""
    name = "x11"

    def __init__(self):
        from Xlib import display
        self.d = display.Display()
        self.root = self.d.screen().root
        self.atom_name = self.d.intern_atom("_NET_WM_NAME")
        self.atom_pid = self.d.intern_atom("_NET_WM_PID")

    def _info(self, w) -> Win | None:
        try:
            geo = w.get_geometry()
            attrs = w.get_attributes()
            if attrs.map_state != 2:  # IsViewable
                return None
            name = ""
            p = w.get_full_property(self.atom_name, 0)
            if p is not None:
                name = p.value.decode("utf-8", "replace") if isinstance(p.value, bytes) else str(p.value)
            if not name:
                name = w.get_wm_name() or ""
            if not name:
                return None
            pid = 0
            pp = w.get_full_property(self.atom_pid, 0)
            if pp is not None and len(pp.value):
                pid = int(pp.value[0])
            return Win(int(w.id), name, pid, geo.x, geo.y, geo.width, geo.height, "")
        except Exception:  # noqa: BLE001
            return None

    def list_windows(self) -> list[Win]:
        out = []
        for w in self.root.query_tree().children:
            i = self._info(w)
            if i and i.width > 1 and i.height > 1:
                out.append(i)
        return out

    def window(self, wid: int) -> Win | None:
        try:
            w = self.d.create_resource_object("window", wid)
            return self._info(w)
        except Exception:  # noqa: BLE001
            return None

    def capture(self, wid: int):
        from PIL import Image
        from Xlib import X
        w = self.d.create_resource_object("window", wid)
        geo = w.get_geometry()
        raw = w.get_image(0, 0, geo.width, geo.height, X.ZPixmap, 0xFFFFFFFF)
        return Image.frombuffer("RGB", (geo.width, geo.height), raw.data, "raw", "BGRX", 0, 1)

    def _send(self, wid: int, ev_cls, **kw):
        from Xlib import X, protocol
        w = self.d.create_resource_object("window", wid)
        ev = ev_cls(time=X.CurrentTime, root=self.root, window=w, same_screen=1, child=X.NONE,
                    root_x=0, root_y=0, **kw)
        w.send_event(ev, propagate=True)
        self.d.flush()

    def bg_click(self, wid, x, y, button="left", count=1):
        from Xlib import X, protocol
        b = {"left": 1, "middle": 2, "right": 3}[button]
        for _ in range(count):
            self._send(wid, protocol.event.ButtonPress, detail=b, state=0, event_x=x, event_y=y)
            self._send(wid, protocol.event.ButtonRelease, detail=b, state=0, event_x=x, event_y=y)

    def bg_drag(self, wid, x1, y1, x2, y2):
        from Xlib import protocol
        self._send(wid, protocol.event.ButtonPress, detail=1, state=0, event_x=x1, event_y=y1)
        self._send(wid, protocol.event.MotionNotify, detail=0, state=0x100, event_x=x2, event_y=y2)
        self._send(wid, protocol.event.ButtonRelease, detail=1, state=0x100, event_x=x2, event_y=y2)

    def bg_type(self, wid, text):
        from Xlib import XK, protocol
        for ch in text:
            sym = XK.string_to_keysym(ch if ch != " " else "space") or ord(ch)
            code = self.d.keysym_to_keycode(sym)
            self._send(wid, protocol.event.KeyPress, detail=code, state=0, event_x=0, event_y=0)
            self._send(wid, protocol.event.KeyRelease, detail=code, state=0, event_x=0, event_y=0)

    def bg_scroll(self, wid, x, y, dy):
        from Xlib import protocol
        b = 5 if dy > 0 else 4
        for _ in range(abs(dy) or 1):
            self._send(wid, protocol.event.ButtonPress, detail=b, state=0, event_x=x, event_y=y)
            self._send(wid, protocol.event.ButtonRelease, detail=b, state=0, event_x=x, event_y=y)

    # X11 has no foreground retry path here: background input is the only input.
    def fg_click(self, wid, x, y, button="left", count=1):
        self.bg_click(wid, x, y, button, count)

    def fg_type(self, wid, text):
        self.bg_type(wid, text)

    def fg_hotkey(self, wid, keys):
        raise RuntimeError("hotkeys are not supported on the X11 test backend")

    def fg_scroll(self, wid, x, y, dy):
        self.bg_scroll(wid, x, y, dy)

    def launch(self, command: str) -> None:
        subprocess.Popen(command, shell=True)


def make_backend():
    """The backend for this machine, or None when there is no screen to drive."""
    try:
        if sys.platform == "win32":
            return Win32Backend()
        if os.environ.get("DISPLAY"):
            return X11Backend()
    except Exception as exc:  # noqa: BLE001 - no window hand is not a reason to die
        print(f"window hand OFF - {type(exc).__name__}: {exc}", file=sys.stderr)
    return None


# ================================================================ the lane's hand
class WindowHand:
    """One lane's hand: bound to one window, seeing and acting only there."""

    def __init__(self, lane: str = "", backend=None, registry: LaneWindows | None = None):
        self.lane = lane or f"pid{os.getpid()}"
        self.be = backend or make_backend()
        self.reg = registry or LaneWindows()
        self.win_id: int | None = None
        self.scale = 1.0
        self.last_hash = ""
        self.input_lock = state_dir() / "input.lock"

    @property
    def available(self) -> bool:
        return self.be is not None

    # ---- result shape: MCP-style content, so the app's observer picks the image up
    @staticmethod
    def _out(text: dict, shot: bytes | None = None) -> str:
        content = [{"type": "text", "text": json.dumps(text)}]
        if shot:
            content.append({"type": "image", "data": base64.b64encode(shot).decode(), "mimeType": "image/jpeg"})
        return json.dumps({"content": content})

    @staticmethod
    def _err(msg: str) -> str:
        return json.dumps({"error": msg})

    # ---- binding
    def _own(self, w: Win) -> bool:
        t = w.title.lower()
        return w.pid == os.getpid() or t == "hands" or t.startswith("hands -")

    def list_windows(self) -> str:
        taken = {int(r["id"]): lane for lane, r in self.reg.all().items()}
        rows = []
        for w in self.be.list_windows():
            if self._own(w):
                continue
            row = w.public()
            if w.id in taken and taken[w.id] != self.lane:
                row["owned_by_lane"] = taken[w.id]
            rows.append(row)
        return self._out({"windows": rows, "bound": self.win_id})

    def bind(self, *, window_id: int | None = None, title: str = "", app: str = "") -> str:
        cands = [w for w in self.be.list_windows() if not self._own(w)]
        if window_id is not None:
            cands = [w for w in cands if w.id == int(window_id)]
        if title:
            cands = [w for w in cands if title.lower() in w.title.lower()]
        if app:
            cands = [w for w in cands if app.lower() in (w.app or "").lower() or app.lower() in w.title.lower()]
        taken = {int(r["id"]) for l, r in self.reg.all().items() if l != self.lane}
        free = [w for w in cands if w.id not in taken]
        if not free:
            return self._err("no free window matches" + (" (matching ones belong to other lanes)" if cands else ""))
        return self._bind_win(free[0])

    def _bind_win(self, w: Win) -> str:
        self.reg.bind(self.lane, w)
        self.win_id = w.id
        return self._seen({"bound": w.public()}, w)

    def launch(self, command: str, title: str = "", timeout_s: float = 12.0) -> str:
        before = {w.id for w in self.be.list_windows()}
        self.be.launch(command)
        deadline = time.monotonic() + max(1.0, min(float(timeout_s), 60.0))
        taken = {int(r["id"]) for l, r in self.reg.all().items() if l != self.lane}
        while time.monotonic() < deadline:
            for w in self.be.list_windows():
                if w.id in before or w.id in taken or self._own(w):
                    continue
                if title and title.lower() not in w.title.lower():
                    continue
                time.sleep(0.15)  # let it paint
                return self._bind_win(self.be.window(w.id) or w)
            time.sleep(0.05)
        return self._err("no new window appeared; use win_list then win_bind")

    def release(self) -> str:
        self.reg.release(self.lane)
        self.win_id = None
        return self._out({"released": True})

    # ---- seeing
    def _frame(self):
        if self.win_id is None:
            raise LookupError("this lane has no window yet: win_launch or win_bind first")
        w = self.be.window(self.win_id)
        if w is None:
            raise LookupError("the window was closed")
        img = self.be.capture(self.win_id)
        shot, self.scale = encode_shot(img)
        h = hashlib.blake2b(shot, digest_size=8).hexdigest()
        return w, shot, h, img.size

    def _seen(self, info: dict, w: Win | None = None) -> str:
        w, shot, h, size = self._frame()
        self.last_hash = h
        info = {**info, "window": w.title, "shot": {"w": round(size[0] * self.scale),
                                                    "h": round(size[1] * self.scale)}}
        return self._out(info, shot)

    def screenshot(self) -> str:
        try:
            return self._seen({})
        except LookupError as e:
            return self._err(str(e))

    def screen_text(self) -> str:
        w = self.be.window(self.win_id) if self.win_id is not None else None
        return self._out({"lane": self.lane, "window": w.public() if w else None})

    # ---- acting
    def _real(self, x: float, y: float) -> tuple[int, int]:
        s = self.scale or 1.0
        return int(round(x / s)), int(round(y / s))

    def _act(self, label: str, bg, fg, settle: float = 0.12) -> str:
        """Run an action in the background; if the window did not react, retry in the
        foreground under the input lock. Always answers with the new frame."""
        if self.win_id is None:
            return self._err("this lane has no window yet: win_launch or win_bind first")
        try:
            before = self.last_hash
            bg()
            time.sleep(settle)
            w, shot, h, size = self._frame()
            mode = "background"
            if h == before and fg is not None:
                time.sleep(0.25)
                w, shot, h, size = self._frame()
                if h == before:
                    with _flock(self.input_lock):
                        fg()
                        time.sleep(settle)
                        w, shot, h, size = self._frame()
                    mode = "foreground"
            self.last_hash = h
            return self._out({"ok": True, "did": label, "mode": mode, "changed": h != before,
                              "window": w.title, "shot": {"w": round(size[0] * self.scale),
                                                          "h": round(size[1] * self.scale)}}, shot)
        except (LookupError, RuntimeError, TimeoutError, ValueError) as e:
            return self._err(str(e))

    def click(self, x: float, y: float, button: str = "left", count: int = 1) -> str:
        rx, ry = self._real(x, y)
        return self._act(f"{button} click x{count} at {int(x)},{int(y)}",
                         lambda: self.be.bg_click(self.win_id, rx, ry, button, count),
                         lambda: self.be.fg_click(self.win_id, rx, ry, button, count))

    def drag(self, x1: float, y1: float, x2: float, y2: float) -> str:
        a, b = self._real(x1, y1), self._real(x2, y2)
        return self._act(f"drag {int(x1)},{int(y1)} -> {int(x2)},{int(y2)}",
                         lambda: self.be.bg_drag(self.win_id, *a, *b), None)

    def type_text(self, text: str) -> str:
        return self._act(f"type {len(text)} chars",
                         lambda: self.be.bg_type(self.win_id, text),
                         lambda: self.be.fg_type(self.win_id, text))

    def hotkey(self, keys: str) -> str:
        parts = [k.strip() for k in keys.replace(" ", "+").split("+") if k.strip()]
        return self._hotkey(parts)

    def _hotkey(self, parts: list[str]) -> str:
        if self.win_id is None:
            return self._err("this lane has no window yet: win_launch or win_bind first")
        try:
            before = self.last_hash
            with _flock(self.input_lock):
                self.be.fg_hotkey(self.win_id, parts)
                time.sleep(0.15)
                w, shot, h, size = self._frame()
            self.last_hash = h
            return self._out({"ok": True, "did": "hotkey " + "+".join(parts), "mode": "foreground",
                              "changed": h != before, "window": w.title}, shot)
        except (LookupError, RuntimeError, TimeoutError, ValueError) as e:
            return self._err(str(e))

    def scroll(self, x: float, y: float, dy: int = 3) -> str:
        rx, ry = self._real(x, y)
        return self._act(f"scroll {dy} at {int(x)},{int(y)}",
                         lambda: self.be.bg_scroll(self.win_id, rx, ry, int(dy)),
                         lambda: self.be.fg_scroll(self.win_id, rx, ry, int(dy)))

    def wait(self, seconds: float = 0.5) -> str:
        time.sleep(max(0.0, min(float(seconds), 10.0)))
        return self.screenshot()


# ---------------------------------------------------------------- tool schemas
def _tool(name: str, desc: str, props: dict | None = None, required: list | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


_N = {"type": "number"}
WIN_TOOLS = [
    _tool("win_launch", "Start an app and take over its window (this lane sees and acts only there). "
          "Every action below returns a fresh screenshot of that window, so you rarely need win_screenshot.",
          {"command": {"type": "string", "description": "e.g. notepad, calc, explorer C:\\Users"},
           "title": {"type": "string", "description": "optional: part of the expected window title"}}, ["command"]),
    _tool("win_list", "List app windows you could take over (others' windows are marked)."),
    _tool("win_bind", "Take over an already-open window by id, title or app name.",
          {"window_id": {"type": "integer"}, "title": {"type": "string"}, "app": {"type": "string"}}),
    _tool("win_screenshot", "Look at your window now."),
    _tool("win_click", "Click at x,y in the screenshot's pixels. count=2 double-clicks.",
          {"x": _N, "y": _N, "button": {"type": "string", "enum": ["left", "right", "middle"]},
           "count": {"type": "integer"}}, ["x", "y"]),
    _tool("win_drag", "Drag from x1,y1 to x2,y2.", {"x1": _N, "y1": _N, "x2": _N, "y2": _N}, ["x1", "y1", "x2", "y2"]),
    _tool("win_type", "Type text into the focused control of your window.", {"text": {"type": "string"}}, ["text"]),
    _tool("win_hotkey", "Press a key combination, e.g. ctrl+s, alt+f4, enter.", {"keys": {"type": "string"}}, ["keys"]),
    _tool("win_scroll", "Scroll at x,y; positive dy scrolls down.", {"x": _N, "y": _N, "dy": {"type": "integer"}},
          ["x", "y"]),
    _tool("win_wait", "Wait a moment (max 10 s) then look.", {"seconds": _N}),
    _tool("win_release", "Let go of your window (another lane may take it)."),
]
WIN_READS = {"win_list", "win_screenshot", "win_wait", "win_release"}


def dispatch(hand: WindowHand, name: str, a: dict) -> str:
    """Run one win_* tool. Raises TypeError on bad arguments like the other tools."""
    if name == "win_launch":
        return hand.launch(a["command"], a.get("title", ""))
    if name == "win_list":
        return hand.list_windows()
    if name == "win_bind":
        return hand.bind(window_id=a.get("window_id"), title=a.get("title", ""), app=a.get("app", ""))
    if name == "win_screenshot":
        return hand.screenshot()
    if name == "win_click":
        return hand.click(a["x"], a["y"], a.get("button", "left"), int(a.get("count", 1) or 1))
    if name == "win_drag":
        return hand.drag(a["x1"], a["y1"], a["x2"], a["y2"])
    if name == "win_type":
        return hand.type_text(str(a["text"]))
    if name == "win_hotkey":
        return hand.hotkey(str(a["keys"]))
    if name == "win_scroll":
        return hand.scroll(a["x"], a["y"], int(a.get("dy", 3) or 3))
    if name == "win_wait":
        return hand.wait(a.get("seconds", 0.5))
    if name == "win_release":
        return hand.release()
    if name == "screen_text":
        return hand.screen_text()
    return json.dumps({"error": f"unknown tool {name}"})
