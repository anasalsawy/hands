"""Quote-or-die: nothing is attributed to the user unless their exact words exist.

A note, reading, collateral item, hypothesis, entity or loop that claims the
user said something must carry a quote that is found verbatim in what the user
actually wrote. If it cannot be found, the item is dropped - in code, not by
asking the model nicely. The stored quote is the original slice of the user's
text (exact casing and spelling), never the model's paraphrase of it.
"""
from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
                         "–": "-", "—": "-"})


def _norm_char_map(text: str) -> tuple[str, list[int]]:
    """Normalized text plus a map from normalized index -> original index."""
    out_chars: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        ch2 = unicodedata.normalize("NFKC", ch).translate(_QUOTES).casefold()
        for c in ch2:
            if c.isspace():
                if prev_space:
                    continue
                c = " "
                prev_space = True
            else:
                prev_space = False
            out_chars.append(c)
            index.append(i)
    return "".join(out_chars), index


def _clean_quote(q: str) -> str:
    q = (q or "").strip()
    q = q.strip("\"'`“”‘’ ")
    q = re.sub(r"^(\.\.\.|…)\s*|\s*(\.\.\.|…)$", "", q)
    return q.strip()


def ground(quote: str, sources: list[str], *, min_len: int = 2) -> str | None:
    """Return the exact original slice of a source that the quote matches, else None."""
    q = _clean_quote(quote)
    if len(q) < min_len:
        return None
    nq, _ = _norm_char_map(q)
    nq = nq.strip()
    if not nq:
        return None
    for src in sources:
        if not src:
            continue
        ns, idx = _norm_char_map(src)
        pos = ns.find(nq)
        if pos == -1:
            # tolerate trailing punctuation differences ("I hate mornings." vs "I hate mornings")
            alt = nq.rstrip(".!?,;:")
            if len(alt) >= min_len and alt != nq:
                pos = ns.find(alt)
                nq_eff = alt
            else:
                continue
        else:
            nq_eff = nq
        if pos == -1:
            continue
        start = idx[pos]
        end = idx[pos + len(nq_eff) - 1] + 1
        return src[start:end]
    return None
