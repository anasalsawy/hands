"""Book harvest: a list of PDF urls -> stored pdfs, plus what they are.

One line per book in a plain text file: `title | url | note`. The note is for a
human. Nothing is guessed: a .pdf url that serves text/html becomes a
`skipped="type"` record, which is the site's problem, not a file we mangle.

This stores and classifies. Text extraction (fitz/pdfplumber) runs later over the
stored files via `pdf_text`, and a truly empty text layer means a scanned book:
that is reported, not silently swallowed — OCR is a separate, expensive pass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..normalize import content_hash
from .base import Fetched, Harvester


@dataclass
class BookSpec:
    """One book we set out to get. `url` is what the list can reach; `local` is
    what some other harvest (an archive.org borrow, a donor's dropbox) already put
    in data/, sha included, so nothing re-downloads it."""

    title: str
    url: str = ""
    note: str = ""
    sha: str = ""

    @property
    def key(self) -> str:
        """Identity of this book, even when two lists spell it differently."""
        return f"{content_hash(self.title)}::{self.url or self.sha}"


def load_book_list(path: str) -> list[BookSpec]:
    """Read a plain text list: one book per line, `title | url | note`.

    Blank lines and `#` comments are ignored. A line with no separator is treated
    as a url alone, since the common way this list is built is copy-paste.
    Returns [] for a missing or unreadable file rather than raising, so a missing
    list is a `0 books harvested` in the journal and not a dead crawl.
    """
    specs: list[BookSpec] = []
    if not os.path.isfile(path):
        return specs
    with open(path, encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 1 and parts[0].lower().startswith(("http://", "https://")):
                specs.append(BookSpec(title=os.path.basename(parts[0]) or parts[0], url=parts[0]))
                continue
            title = parts[0]
            url = parts[1] if len(parts) > 1 else ""
            note = parts[2] if len(parts) > 2 else ""
            if url and not title:
                title = os.path.basename(url)
            specs.append(BookSpec(title=title, url=url, note=note))
    return specs


def read_book(path: str) -> tuple[str, bool]:
    """(text, needs_ocr) for a stored pdf. Never raises.

    An empty text layer with pages present means a scanned book: reported for OCR
    instead of storing blank passages that would later retrieve as blank answers.
    """
    try:
        import fitz                        # PyMuPDF, already in this venv
    except ImportError:
        return _pdfminer_text(path)
    try:
        doc = fitz.open(path)
    except Exception:
        return "", False
    try:
        text = "\n".join(page.get_text("text") for page in doc)
        pages = doc.page_count
    finally:
        doc.close()
    return (text, False) if text.strip() else ("", bool(pages))


def _pdfminer_text(path: str) -> tuple[str, bool]:
    """Fallback reader, so a missing PyMuPDF does not stop a book harvest."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        return "", False
    try:
        text = extract_text(path) or ""
    except Exception:
        return "", False
    return (text, False) if text.strip() else ("", True)


async def crawl_books(h: Harvester, specs: list[BookSpec]) -> list[tuple[BookSpec, Fetched]]:
    """Fetch every book url once. Returns (spec, result) pairs — nothing silently dropped.

    A spec with no url is journalled as `pending`, not forgotten: it means the
    volume is not located yet, and the next report should list it as open.
    """
    todo = [s for s in specs if s.url]
    rows = await h.fetch_many([s.url for s in todo], kind="pdf")
    out = list(zip(todo, rows))
    for s in specs:
        if not s.url:
            h.journal.write(event="book", title=s.title, status="pending", note=s.note)
    for s, f in out:
        h.journal.write(event="book", title=s.title, url=f.url, status="ok" if f.ok else f.skipped,
                        sha=f.sha, bytes=f.size)
    return out

