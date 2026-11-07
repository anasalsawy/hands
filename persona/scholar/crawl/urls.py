"""Urls, journal, policy. Boring on purpose: resumability is one append-only file.
"""
from __future__ import annotations

import hashlib
import re
import time
from urllib.parse import urlsplit, urlunsplit, urlparse


def now() -> float:
    return time.time()


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob or b"").hexdigest()


def host_of(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def normalize_url(url: str) -> str:
    """One page, one spelling: drop the fragment, lowercase the host, drop a
    default port and one trailing slash. A bare root keeps its slash."""
    p = urlsplit((url or "").strip())
    net = p.netloc.lower()
    if net.endswith(":80") and p.scheme == "http":
        net = net[:-3]
    elif net.endswith(":443") and p.scheme in ("https", "http"):
        net = net[:-4]
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((p.scheme.lower(), net, path, p.query, ""))


# --- decoding -----------------------------------------------------------------
# Arabic fatwa sites are a graveyard of encodings: old ASP pages speak
# windows-1256, modern ones speak utf-8, and many declare one in a http header
# and another in a <meta>. Trust the document over the header, then try the
# codecs that actually produced garbage in the past, in that order. Never let a
# decode error stop the crawl for one page.
_META_CHARSET = re.compile(rb"""charset[ \t]*=[ \t]*["']?([A-Za-z0-9_\-:.]+)""", re.I)
_CANDIDATES = ("utf-8", "windows-1256", "iso-8859-6", "cp720", "latin-1")

_BOM = {b"\xef\xbb\xbf": "utf-8-sig"}


def decode_html(blob: bytes, ctype: str = "") -> str:
    """Best-effort text for an Arabic page, never an exception."""
    if not blob:
        return ""
    for bom, enc in _BOM.items():
        if blob.startswith(bom):
            return blob.decode(enc, "replace")
    declared: list[str] = []
    m = _META_CHARSET.search(blob[:4096])
    if m:
        declared.append(m.group(1).decode("ascii", "ignore").strip())
    h = (ctype or "").lower()
    if "charset=" in h:
        declared.append(h.split("charset=", 1)[1].split(";")[0].strip().strip('"\''))
    tried: list[str] = []
    for enc in declared + list(_CANDIDATES):
        e = (enc or "").strip()
        if not e or e in tried:
            continue
        tried.append(e)
        try:
            return blob.decode(e)
        except (UnicodeDecodeError, LookupError):
            continue
    return blob.decode("utf-8", "replace")


