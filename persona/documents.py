"""Documents she makes for the person: a real file, handed over as a link.

`make_document` renders a title and markdown-ish text to PDF, Markdown or
plain text, stores it under ``<data>/files/<pid>/`` and returns a url the app
shows as a download chip. Only the persona that made a file can fetch it
(GET /v1/files/{id} with its key). A visit does not get this tool.

PDF uses fpdf2 with a Unicode TrueType font found on the machine (DejaVu,
Noto, FreeFont, Liberation, Arial). Arabic is shaped and laid out right to
left when uharfbuzz is installed; without it Arabic letters print unjoined.
With no Unicode font at all, non-Latin-1 characters print as "?".
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

FORMATS = {
    "pdf": ("application/pdf", ".pdf"),
    "md": ("text/markdown; charset=utf-8", ".md"),
    "txt": ("text/plain; charset=utf-8", ".txt"),
}
MAX_CONTENT = 200_000
MAX_BYTES = 10_000_000
KEEP = 200
ID = re.compile(r"^[0-9a-f]{32}$")

SCHEMA = {"type": "function", "function": {
    "name": "make_document",
    "description": ("Make a real file (PDF, Markdown or text) from a title and markdown text; the person gets "
                    "a download link in the chat. Use this, not the shell, when they ask for a document, and "
                    "never say a file exists unless this returned its url."),
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"},
        "format": {"type": "string", "enum": list(FORMATS)},
        "content": {"type": "string", "description": "The document body. # headings, - bullets, **bold**."},
    }, "required": ["title", "content"]}}}

_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
              "/Library/Fonts", "/System/Library/Fonts", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
# (regular, bold) file names, best first. All cover Arabic except Liberation.
_FAMILIES = [
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
    ("arial.ttf", "arialbd.ttf"),
    ("Arial.ttf", "Arial Bold.ttf"),
    ("FreeSans.ttf", "FreeSansBold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
]
_ARABIC_FALLBACKS = ["NotoSansArabic-Regular.ttf", "NotoNaskhArabic-Regular.ttf", "DejaVuSans.ttf",
                     "FreeSerif.ttf", "arial.ttf", "Arial.ttf"]
_ARABIC = re.compile(r"[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]")
_STRONG_LTR = re.compile(r"[A-Za-z\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff]")
_fonts_cache: dict | None = None


def _find_fonts() -> dict:
    global _fonts_cache
    if _fonts_cache is not None:
        return _fonts_cache
    found: dict[str, str] = {}
    wanted = {n for pair in _FAMILIES for n in pair} | set(_ARABIC_FALLBACKS)
    for root in _FONT_DIRS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name in wanted and name not in found:
                    found[name] = os.path.join(dirpath, name)
    regular = bold = ""
    for reg, bld in _FAMILIES:
        if reg in found:
            regular, bold = found[reg], found.get(bld, found[reg])
            break
    fallback = next((found[n] for n in _ARABIC_FALLBACKS if n in found and found[n] != regular), "")
    _fonts_cache = {"regular": regular, "bold": bold, "fallback": fallback}
    return _fonts_cache


def _shaping_available() -> bool:
    try:
        import uharfbuzz  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _escape_md(text: str) -> str:
    # fpdf2 markdown reads ** __ -- ~~ as styles; only ** is meant here.
    return re.sub(r"(__|--|~~)", r"\\\1", text)


def render_pdf(title: str, content: str) -> bytes:
    from fpdf import FPDF

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_title(title[:200])
    pdf.set_creator("Sawii")
    fonts = _find_fonts()
    unicode_font = bool(fonts["regular"])
    if unicode_font:
        pdf.add_font("body", "", fonts["regular"])
        pdf.add_font("body", "B", fonts["bold"])
        if fonts["fallback"]:
            pdf.add_font("fallback", "", fonts["fallback"])
            pdf.set_fallback_fonts(["fallback"], exact_match=False)
        if _shaping_available():
            pdf.set_text_shaping(True)
        family = "body"
    else:
        family = "helvetica"

    def clean(text: str) -> str:
        return text if unicode_font else text.encode("latin-1", "replace").decode("latin-1")

    def align(text: str) -> str:
        arabic, latin = _ARABIC.search(text), _STRONG_LTR.search(text)
        return "R" if arabic and (not latin or arabic.start() < latin.start()) else "L"

    def para(text: str, size: float, style: str = "", indent: float = 0, gap: float = 1.5):
        pdf.set_font(family, style, size)
        pdf.set_x(pdf.l_margin + indent)
        text = clean(text)
        pdf.multi_cell(pdf.epw - indent, size * 0.5, _escape_md(text), align=align(text),
                       markdown=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    pdf.add_page()
    para(title, 20, "B", gap=4)
    bullet = "\u2022" if unicode_font else "-"
    for raw in content.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            pdf.ln(2.5)
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            pdf.ln(2)
            para(heading.group(2), {1: 17, 2: 15}.get(level, 13), "B", gap=2)
            continue
        if re.fullmatch(r"[-*_]{3,}", stripped):
            y = pdf.get_y() + 1
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(4)
            continue
        depth = min((len(line) - len(line.lstrip())) // 2, 4) * 5
        item = re.match(r"^[-*\u2022]\s+(.*)$", stripped)
        if item:
            para(f"{bullet} {item.group(1)}", 11, indent=4 + depth)
            continue
        numbered = re.match(r"^(\d{1,3}[.)])\s+(.*)$", stripped)
        if numbered:
            para(f"{numbered.group(1)} {numbered.group(2)}", 11, indent=4 + depth)
            continue
        para(stripped, 11)
    return bytes(pdf.output())


def render(fmt: str, title: str, content: str) -> bytes:
    if fmt == "pdf":
        return render_pdf(title, content)
    if fmt == "md":
        body = content if content.lstrip().startswith("#") else f"# {title}\n\n{content}"
        return (body.rstrip() + "\n").encode("utf-8")
    return (f"{title}\n\n{content}".rstrip() + "\n").encode("utf-8")


def safe_name(title: str, ext: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "", title, flags=re.UNICODE).strip()
    stem = re.sub(r"\s+", "-", stem)[:60].strip("-.") or "document"
    return stem + ext


class Documents:
    def __init__(self, data_dir: str):
        self.root = Path(data_dir) / "files"

    def _dir(self, pid: int) -> Path:
        path = self.root / str(int(pid))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def make(self, pid: int, args: dict) -> str:
        title = " ".join(str(args.get("title") or "").split())[:200]
        content = str(args.get("content") or "")
        fmt = str(args.get("format") or "pdf").strip().lower().lstrip(".")
        fmt = {"markdown": "md", "text": "txt"}.get(fmt, fmt)
        if fmt not in FORMATS:
            return json.dumps({"ok": False, "error": f"format must be one of {', '.join(FORMATS)}"})
        if not title or not content.strip():
            return json.dumps({"ok": False, "error": "a document needs a title and content"})
        if len(content) > MAX_CONTENT:
            return json.dumps({"ok": False, "error": f"content is longer than {MAX_CONTENT} characters"})
        try:
            data = render(fmt, title, content)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"could not render: {type(exc).__name__}: {exc}"[:300]})
        if len(data) > MAX_BYTES:
            return json.dumps({"ok": False, "error": "the file came out too large"})
        fid = uuid.uuid4().hex
        folder = self._dir(pid)
        name = safe_name(title, FORMATS[fmt][1])
        (folder / f"{fid}.bin").write_bytes(data)
        (folder / f"{fid}.json").write_text(json.dumps(
            {"id": fid, "name": name, "format": fmt, "bytes": len(data), "title": title,
             "created": time.time()}, ensure_ascii=False), encoding="utf-8")
        self._prune(folder)
        return json.dumps({"ok": True, "url": f"/v1/files/{fid}", "name": name, "bytes": len(data),
                           "format": fmt, "id": fid}, ensure_ascii=False)

    def _prune(self, folder: Path) -> None:
        metas = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in metas[:-KEEP]:
            old.unlink(missing_ok=True)
            old.with_suffix(".bin").unlink(missing_ok=True)

    def get(self, pid: int, fid: str) -> tuple[Path, dict] | None:
        """The file and its record, only if this persona made it."""
        if not ID.match(fid or ""):
            return None
        folder = self.root / str(int(pid))
        meta_path, data_path = folder / f"{fid}.json", folder / f"{fid}.bin"
        if not meta_path.is_file() or not data_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if meta.get("format") not in FORMATS:
            return None
        return data_path, meta
