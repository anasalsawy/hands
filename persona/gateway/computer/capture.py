"""Windows-friendly live screen capture for Background computer cards.

Primary path: ``mss`` (fast monitor/region grabs) + Pillow JPEG encode.
Fallback: GDI BitBlt via ctypes when mss is unavailable.

Self-exclusion is best-effort only: after a desktop/region grab we black out
rects belonging to Sawii/companion/webview windows. True OS exclude-from-capture
(WGC) is not required for v1 and may still leave a mirror if the companion
overlaps the target heavily.
"""
from __future__ import annotations

import io
import logging
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Titles / process names we try to black out on desktop/region streams.
_SELF_HINTS = (
    "sawii",
    "companion",
    "persona",
    "webview2",
    "msedgewebview2",
)


@dataclass(frozen=True)
class WinInfo:
    hwnd: int
    title: str
    pid: int
    left: int
    top: int
    right: int
    bottom: int
    visible: bool

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


def _win32():
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    gdi32 = ctypes.windll.gdi32
    return ctypes, wintypes, user32, kernel32, gdi32


def list_windows(*, include_minimized: bool = False) -> list[WinInfo]:
    """Enumerate top-level windows suitable as capture targets."""
    pack = _win32()
    if pack is None:
        return []
    ctypes, wintypes, user32, _kernel32, _gdi32 = pack

    results: list[WinInfo] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lp):
        if not user32.IsWindow(hwnd):
            return True
        visible = bool(user32.IsWindowVisible(hwnd))
        if not visible and not include_minimized:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title:
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        if (rect.right - rect.left) < 40 or (rect.bottom - rect.top) < 40:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        results.append(WinInfo(
            hwnd=int(hwnd),
            title=title[:200],
            pid=int(pid.value),
            left=int(rect.left),
            top=int(rect.top),
            right=int(rect.right),
            bottom=int(rect.bottom),
            visible=visible,
        ))
        return True

    user32.EnumWindows(EnumWindowsProc(_cb), 0)
    results.sort(key=lambda w: (w.title.lower(), w.pid, w.hwnd))
    return results


def find_window(*, hwnd: int | None = None, pid: int | None = None, title: str | None = None) -> WinInfo | None:
    wins = list_windows(include_minimized=True)
    if hwnd is not None:
        for w in wins:
            if w.hwnd == int(hwnd):
                return w
    needle = (title or "").strip().lower()
    if pid is not None:
        cands = [w for w in wins if w.pid == int(pid)]
        if needle:
            cands = [w for w in cands if needle in w.title.lower()] or cands
        return cands[0] if cands else None
    if needle:
        for w in wins:
            if needle in w.title.lower():
                return w
    return None


def self_windows() -> list[WinInfo]:
    """Best-effort companion/Sawii windows to black out on full-desktop grabs."""
    out: list[WinInfo] = []
    for w in list_windows(include_minimized=False):
        t = w.title.lower()
        if any(h in t for h in _SELF_HINTS):
            out.append(w)
            continue
        # WebView2 hosts often title the page; also match our process name.
        try:
            import os
            if w.pid == os.getpid():
                out.append(w)
        except Exception:  # noqa: BLE001
            pass
    return out


def _encode_jpeg(rgb_bytes: bytes, width: int, height: int, *, quality: int = 55) -> bytes:
    from PIL import Image

    img = Image.frombytes("RGB", (width, height), rgb_bytes)
    # Cap stream size for multi-card UI.
    max_w = 1280
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, max(1, int(img.height * ratio))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _blackout(img, rects: list[tuple[int, int, int, int]], origin: tuple[int, int]) -> None:
    """Paint black over absolute-screen rects relative to capture origin."""
    from PIL import ImageDraw

    if not rects:
        return
    ox, oy = origin
    draw = ImageDraw.Draw(img)
    for left, top, right, bottom in rects:
        x0 = left - ox
        y0 = top - oy
        x1 = right - ox
        y1 = bottom - oy
        if x1 <= 0 or y1 <= 0 or x0 >= img.width or y0 >= img.height:
            continue
        draw.rectangle(
            [max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)],
            fill=(8, 10, 16),
        )


def _grab_mss(bbox: dict, *, exclude: list[WinInfo], quality: int) -> bytes:
    import mss
    from PIL import Image

    with mss.mss() as sct:
        shot = sct.grab(bbox)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    if exclude:
        _blackout(img, [(w.left, w.top, w.right, w.bottom) for w in exclude], (bbox["left"], bbox["top"]))
    max_w = 1280
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, max(1, int(img.height * ratio))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _grab_gdi(bbox: dict, *, exclude: list[WinInfo], quality: int) -> bytes:
    pack = _win32()
    if pack is None:
        raise RuntimeError("screen capture requires Windows")
    ctypes, wintypes, user32, _kernel32, gdi32 = pack
    left, top = int(bbox["left"]), int(bbox["top"])
    width, height = int(bbox["width"]), int(bbox["height"])
    if width < 1 or height < 1:
        raise ValueError("empty capture region")

    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    old = gdi32.SelectObject(hdc_mem, hbmp)
    try:
        SRCCOPY = 0x00CC0020
        if not gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, left, top, SRCCOPY):
            raise OSError("BitBlt failed")
        # BITMAPINFOHEADER
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0
        buf = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(hdc_mem, hbmp, 0, height, buf, ctypes.byref(bmi), 0)
        from PIL import Image

        img = Image.frombuffer("RGB", (width, height), buf, "raw", "BGRX", 0, 1).copy()
        if exclude:
            _blackout(img, [(w.left, w.top, w.right, w.bottom) for w in exclude], (left, top))
        max_w = 1280
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, max(1, int(img.height * ratio))), Image.Resampling.BILINEAR)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    finally:
        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)


def grab_jpeg(
    *,
    kind: str = "desktop",
    hwnd: int | None = None,
    pid: int | None = None,
    title: str | None = None,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
    exclude_self: bool = True,
    quality: int = 55,
) -> tuple[bytes, dict]:
    """Capture one JPEG frame. Returns (jpeg_bytes, meta)."""
    if sys.platform != "win32":
        raise RuntimeError("live computer capture is Windows-only in v1")

    exclude: list[WinInfo] = self_windows() if exclude_self else []
    meta: dict = {"kind": kind, "exclude_self": exclude_self, "excluded": len(exclude)}

    if kind == "window":
        win = find_window(hwnd=hwnd, pid=pid, title=title)
        if win is None:
            raise LookupError("window not found")
        # Don't black out the target itself.
        exclude = [e for e in exclude if e.hwnd != win.hwnd]
        bbox = {"left": win.left, "top": win.top, "width": win.width, "height": win.height}
        meta.update({"hwnd": win.hwnd, "pid": win.pid, "title": win.title, **bbox})
    elif kind == "region":
        if None in (x, y, w, h):
            raise ValueError("region requires x,y,w,h")
        bbox = {"left": int(x), "top": int(y), "width": max(1, int(w)), "height": max(1, int(h))}
        meta.update(bbox)
    else:
        # Full virtual desktop.
        pack = _win32()
        assert pack is not None
        _ctypes, _wintypes, user32, _kernel32, _gdi32 = pack
        left = int(user32.GetSystemMetrics(76))   # SM_XVIRTUALSCREEN
        top = int(user32.GetSystemMetrics(77))    # SM_YVIRTUALSCREEN
        width = int(user32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
        height = int(user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
        bbox = {"left": left, "top": top, "width": width, "height": height}
        meta.update(bbox)
        kind = "desktop"
        meta["kind"] = "desktop"

    try:
        import mss  # noqa: F401
        jpeg = _grab_mss(bbox, exclude=exclude, quality=quality)
        meta["backend"] = "mss"
    except Exception as exc:  # noqa: BLE001
        log.debug("mss grab failed (%s); trying GDI", exc)
        jpeg = _grab_gdi(bbox, exclude=exclude, quality=quality)
        meta["backend"] = "gdi"
    meta["bytes"] = len(jpeg)
    return jpeg, meta


def capture_available() -> dict:
    info = {
        "platform": sys.platform,
        "ok": sys.platform == "win32",
        "backend": None,
        "exclude_self": "best-effort blackout of Sawii/companion window rects (not OS WGC exclude)",
    }
    if sys.platform != "win32":
        info["error"] = "Windows only"
        return info
    try:
        import mss
        info["backend"] = f"mss {getattr(mss, '__version__', '?')}"
    except Exception:  # noqa: BLE001
        info["backend"] = "gdi"
    try:
        from PIL import Image  # noqa: F401
        info["pillow"] = True
    except Exception:  # noqa: BLE001
        info["pillow"] = False
        info["ok"] = False
        info["error"] = "Pillow required"
    return info
