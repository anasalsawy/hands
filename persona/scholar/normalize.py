"""Making his Arabic searchable without changing a letter of his Arabic.

Two worlds, deliberately separate:

* **display** — exactly what he wrote. Quotes, citations and the spoken reply are
  cut from this. Diacritics, hamza forms and the alef he chose are part of his
  style and never touched.
* **norm** — a deterministic key used only for matching, dedup and hashing.
  ``أبناء`` and ``ابناء`` become one thing here, never in display.

Every function here is pure and total — returns a str for any input, including
None. The corpus ingest walks ten thousand pages; one exception kills a book.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# --- Unicode ranges -----------------------------------------------------------
_ARABIC = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufeff]")
# harakat, tatweel, superscript alef, Quranic annotation marks, half marks
_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640\u0653-\u0655]")
# the soft/bidi junk a PDF extraction drags in
_INVISIBLE = re.compile(r"[\u00ad\u0600\u061c\u06dd\u070f\u200b-\u200f\u202a-\u202e\ufeff]")

# Alef family -> bare alef.
_ALEF = "\u0622\u0623\u0625\u0671\u0672\u0673"          # آ أ إ ٱ ۲ ۳
_TAA = "\u0629"                                       # ة
_YAA = "\u0649\u06cc\u06be\u064a"                     # ى ی ۺ ي
_WAW_HAMZA = "\u0624"                                  # ؤ
_HAMZA_CARRIER = "\u0626\u0621"                        # ئ ء

_LIGATURE = {
    "\u0644\u0627": "\u0644\u0627",   # lam-alef stays two letters (already folded)
}

_DIGITS = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7",
    "٨": "8", "٩": "9",                                  # Arabic-Indic
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7",
    "۸": "8", "۹": "9",                                  # extended / Persian
}

_PUNCT = {
    "،": ",", "؛": ";", "؟": "?", "٪": "%", "\u066b": ".", "\u066c": ",",
    "\u066d": "'", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"', "\u2026": "...",
}

# Word characters for segmentation: Arabic letters in every form, latin, digits.
_WORD_CHARS = "\u0600-\u06ff\u0750-\u077f\ufb50-\ufeff0-9a-zA-Z"
_TOKEN = re.compile(rf"[{_WORD_CHARS}]+")


def strip_marks(text: str) -> str:
    """Remove harakat and tatweel, keep the consonantal skeleton.

    Diacritics carry meaning the *reader* needs and almost never the retriever.
    A scanned page that lost its tashkeel must match an authored page that kept
    it, or the same ruling in two books indexes as two unrelated texts.
    """
    if not text:
        return ""
    return _DIACRITICS.sub("", text)


def fold_letters(text: str) -> str:
    """Alef/taa/yaa/hamza variants into one search alphabet."""
    if not text:
        return ""
    out = []
    for ch in text:
        if ch in _ALEF:
            out.append("\u0627")                     # ا
        elif ch in _TAA:
            out.append("\u0647")                     # ه
        elif ch in _YAA:
            out.append("\u064a")                     # ي
        elif ch in _WAW_HAMZA:
            out.append("\u0648")                     # و
        elif ch in _HAMZA_CARRIER:
            out.append("")                           # the hamza itself is dropped
        else:
            out.append(ch)
    return "".join(out)


def digits_to_latin(text: str) -> str:
    return "".join(_DIGITS.get(ch, ch) for ch in text or "")


def punct_to_latin(text: str) -> str:
    return "".join(_PUNCT.get(ch, ch) for ch in text or "")


def normalize(text: str) -> str:
    """The matching key. Deterministic, idempotent, total.

    ``normalize(normalize(x)) == normalize(x)`` holds for every x. That is not a
    nicety — it is what makes a content hash stable enough to recognise one
    ruling that appears in a book *and* on the website as one fact with two
    witnesses (ARCHITECTURE.md section 11a), instead of two competing answers.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))   # presentation forms -> letters
    t = _INVISIBLE.sub("", t)
    t = fold_letters(t)
    t = strip_marks(t)
    t = digits_to_latin(t)
    t = punct_to_latin(t)
    t = t.replace("_", "")
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def content_hash(text: str) -> str:
    """Identity of *what was said*, ignoring diacritics, letter forms and spacing."""
    key = re.sub(r"\s+", " ", normalize(text)).strip().lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] if key else ""


def tokens(text: str) -> list[str]:
    """Words for lexical scoring. Normalized first, so أ and ا index together."""
    return _TOKEN.findall(normalize(text))


# --- clitics -----------------------------------------------------------------
# "الحكم" is "حكم" plus ال; "وبالحمدلله" is wa + bi + al + .... Arabic lexical
# recall misses heavily unless the stem is reachable. So the stem is emitted
# *alongside* the surface form — a reducing transform used for matching only,
# never a replacement, never shown to anyone as his word.
_PROCLITIC = re.compile(r"^[\u0648\u0641\u0628\u0643]")            # و ف ب ك
_AL = "\u0627\u0644"                                                # ال
# The sun letters ال assimilates into, and the moon letters it does not.
_SUN = "\u0637\u0635\u0631\u0634\u0633\u0630\u062f\u0636\u0639\u063a\u0644\u0631\u0646\u062a"
# Shortest remainder a strip may leave: the triliteral root, the commonest Arabic
# word-shape. Nothing shorter is "still a plausible word", so no prefix that would
# eat a root letter or bare an empty shell is ever removed.
_MIN_ROOT = 3


def stems(word: str) -> list[str]:
    """Surface form, then de-cliticked forms. Deterministic and small on purpose.

    This is not a morphological analyser. A real one (CrowdSALT / MADAN) is a
    dependency this project should adopt deliberately, not smuggle in here.
    """
    w = normalize(word)
    if not w:
        return []
    out = [w]
    seen = {w}

    # `w` came out of normalize(), so alphabet folding is already done and `_AL`
    # here is real, not an approximation. A prefix is removed only when what is
    # left is still a plausible word: at least _MIN_ROOT letters survive under ال,
    # and the single conjunction prefix (و ف ب ك) is stripped only when ال follows
    # it with such a root behind it ("والذبح" -> "الذبح" -> "ذبح"). Words whose
    # first letter merely *spells* a clitic keep all their letters: "كتاب" must
    # not become "تاب", nor "برتقال" become "رتقال", and "وال" must not be
    # reduced to a bare "ال" — a stripped prefix may never eat a root's first
    # letter or leave an empty shell, whatever the surface length.
    cur = w
    for _ in range(3):
        nxt = None
        if cur.startswith(_AL) and len(cur) >= len(_AL) + _MIN_ROOT:
            nxt = cur[len(_AL):]
        elif (_PROCLITIC.match(cur) and cur[1:len(_AL) + 1] == _AL
              and len(cur) >= 1 + len(_AL) + _MIN_ROOT):
            nxt = cur[1:]
        if nxt is None or nxt in seen:
            break
        seen.add(nxt)
        out.append(nxt)
        cur = nxt
    return out
