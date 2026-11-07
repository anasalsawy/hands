"""Fatwa harvest: seed/list pages -> fatwa pages -> (question, answer) records.

Layout assumption on Arabic fatwa sites: a page has a question block and an
answer block, each usually under a bold heading like نص السؤال / الجواب, and a
fatwa number. Nothing here knows one specific site: extraction is pattern-based
and *scored*, and the score is what tells a human which pages to eyeball.

The harvest stores raw html first (base.Harvester), then extracts. So refining
the extractor is a re-run over files we already own — no second crawl.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urljoin

from bs4 import BeautifulSoup

from ..normalize import content_hash
from .base import Fetched, Harvester
from .urls import host_of

# What a fatwa page is made of. Headings first, in the order his site writes them.
_ASK = re.compile(r"(?:^\s*|(?<=[.!؟]))\s*(نص\s+السؤال|السؤال|سؤال)\s*[:：]?\s*", re.M)
_ANSWER = re.compile(r"(?:^\s*|(?<=[.!؟]))\s*(الجواب|الإجابة|أجاب|فلاства)\s*[:：]?\s*", re.M)
_NUMBER = re.compile(r"(?:رقم\s+الفتوى|الفتوى\s*رقم|فتوى\s*رقم)\D{0,12}(\d+)", re.I)
_DATE = re.compile(r"(?:تاريخ\s+الفتوى|الفتوى\s+التاريخ|نشرت?\s+في|تاريخ\s+النشر)\s*[:：]?\s*"
                   r"(\d{4}|\d{1,2}[\u0600-\u06FF\s,\u0660-\u0669]*\d{4}|[0-9/\-.\u0660-\u0669]+)", re.I)
_BISM = re.compile(r"(بسم\s+الله\s+الرحمن\s+الرحيم|بسم\s+الرحمن\s+الرحيم|الحمد\s+لله|الصلاة\s+والسلام)", re.I)
# A WordPress fatwa page carries no labels: the answer starts at the hamdala/basmala
# paragraph. Its permalink carries no number either — it is Arabic slug segments under a
# category base, so a post is recognised by depth (`is_wp_post`), listings excluded.
_FATWA_CONTAINERS = ("div.single-content", "div.single-page-content", "article .entry-content",
                     "div.entry-content", "div.post-content", "article")
# A fatwa we cannot split is usually a list, a section index or a print view.
_JUNK_TITLE = re.compile(r"(فهرس|الأقسام|بحث|اتصال|من نحن|تسجيل|دخول)", re.I)


def _clean(t: str) -> str:
    return re.sub(r"[ \t\u00a0]+", " ", (t or "").strip())


def split_qa(text: str) -> tuple[str, str, float]:
    """(question, answer, score) from one page's visible text.

    His pages label the halves, so a label match is the primary rule. When only
    the answer is labelled we take everything above it as the question — that is
    what the sites actually do — and lower the score, because the asker's line may
    be holding a breadcrumb.
    """
    qm, am = _ASK.search(text), _ANSWER.search(text)
    if qm and am and am.start() > qm.start():
        return _clean(text[qm.end():am.start()]), _clean(text[am.end():]), 1.0
    if am:
        return _clean(text[:am.start()]), _clean(text[am.end():]), 0.8
    if qm:
        return _clean(text[qm.end():]), "", 0.4
    return "", "", 0.0


def _lines_of(cont) -> list[str]:
    """Visible lines of one container, <br> honoured as a line break.

    His WordPress page writes question and answer *inside the same <p>*, with
    <br> between them: walking direct children and flushing the buffer at every
    <br>/block end reproduces what a reader sees, so the hamdala cut lands where
    he actually began answering.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        t = _clean(" ".join(buf))
        if t:
            out.append(t)
        buf.clear()

    def walk(node) -> None:
        for ch in node.children:
            name = getattr(ch, "name", None)
            if name == "br":
                flush()
            elif name == "script" or name == "style":
                continue
            elif name in ("p", "div", "li", "h1", "h2", "h3", "blockquote"):
                flush()
                walk(ch)
                flush()
            elif isinstance(ch, str) or name is None:
                buf.append(str(ch))
            else:
                buf.append(ch.get_text(" ", strip=True))

    walk(cont)
    flush()
    return out


def split_paragraphs(soup: BeautifulSoup) -> tuple[str, str, float]:
    """Unlabelled layout: the hamdala/basmala line marks where the answer starts.

    Everything above it is the question. A WP list page carries no hamdala line,
    so material *before* anything hamdala-like never counts as an answer — it
    scores 0.0 and lands in needs_review rather than posing as a ruling.
    """
    cont = None
    for sel in _FATWA_CONTAINERS:
        cont = soup.select_one(sel)
        if cont:
            break
    if not cont:
        return "", "", 0.0
    # Plugin chrome inside the content div (view counts) is not part of the ruling.
    skipped = re.compile(r"^(Views?:|\d[\d\s,\u0660-\u0669]*$|شارك|أضف رد)", re.I)
    lines = [t for t in (re.sub(r"\s+", " ", x).strip(" '-—•*\\\u200f\u200e") for x in _lines_of(cont))
             if len(t) > 2 and not skipped.search(t)]
    if not lines:
        return "", "", 0.0
    # The first hamdala line *with real material behind it* opens the answer; the
    # question is above it. A page whose first content line is already the hamdala
    # gives us no question to check, so it is flagged rather than trusted.
    cut = next((i for i, t in enumerate(lines)
                if i and _BISM.search(t) and sum(len(x) for x in lines[i:]) >= 80), 0)
    if not cut:
        return "", "", 0.0           # no hamdala under a question: a list page, not a ruling
    question, answer = _clean(" ".join(lines[:cut])), _clean(" ".join(lines[cut:]))
    if len(answer) < 80:
        return "", "", 0.0
    return question, answer, 0.9


@dataclass
class FatwaRecord:
    """One ruling as published. `score` decides who reads it first.

    1.0 means question and answer were both found under their own labels. Below
    0.8 the page is journalled into `needs_review` and a human looks: we would
    rather flag a fatwa than invent one.
    """

    url: str
    sha: str = ""
    fatwa_id: str = ""
    title: str = ""
    question: str = ""
    answer: str = ""
    issued_at: str = ""
    score: float = 0.0
    needs_review: bool = True

    @property
    def topic(self) -> str:
        return self.title or (self.question[:80] if self.question else "")

    @property
    def citation(self) -> str:
        """Where a ruling came from, in the form a student can verify by hand."""
        bits = ["موقع الفتوى" ]
        if self.fatwa_id:
            bits.append(f"رقم {self.fatwa_id}")
        if self.issued_at:
            bits.append(self.issued_at)
        bits.append(self.url)
        return " | ".join(bits)

    @property
    def is_worthwhile(self) -> bool:
        return bool(self.answer and len(self.answer) >= 40)

    @property
    def key(self) -> str:
        """Identity of the ruling itself: question + answer, orthography folded.

        `by_ruling` groups on this, two pages that publish one answer land in one
        group while two different answers never collide.
        """
        return content_hash(f"{self.question}\n{self.answer}")


def _label(soup: BeautifulSoup) -> str:
    for node in (soup.find("h1"), soup.find("title")):
        if node:
            return _clean(node.get_text(" "))
    return ""


def parse_fatwa(url: str, html: str, *, sha: str = "") -> FatwaRecord:
    """Read one saved page. Never raises — an unparseable page scores 0.0."""
    soup = BeautifulSoup(html or "", "html.parser")
    for dead in soup(["script", "style", "noscript"]):
        dead.decompose()
    title = _label(soup)
    text = _clean(soup.get_text("\n"))
    q, a, score = split_qa(text)
    if not score:
        q, a, score = split_paragraphs(soup)
    m = _NUMBER.search(text)
    d = _DATE.search(text)
    if _JUNK_TITLE.search(title):
        score = min(score, 0.2)
    if not _BISM.search(a or ""):
        score *= 0.7            # his answers almost always open with a hamdala
    return FatwaRecord(
        url=url, sha=sha, fatwa_id=m.group(1) if m else "", title=title,
        question=q, answer=a,
        issued_at=_clean(d.group(1)) if d else "",
        score=round(max(0.0, min(1.0, score)), 2),
        needs_review=score < 0.8)


# A fatwa url on these sites carries a number: /fatwa/1234, ?id=1234, /index.php/99.
# So a numeric tail is a better hint than any word we could guess the spelling of.
_DETAIL = re.compile(r"(?:/|\b)(?:fatwa|fatawa|answer|question|detail|view|show|story|doc|"
                     r"STORY|Page|node)[-_]?(\d{1,9})\b", re.I)
_ID_QUERY = re.compile(r"[?&](?:id|dt|nid|fk|post|no|num|code)=(\d{1,9})\b", re.I)
_ID_PATH = re.compile(r"/(\d{2,9})(?:[/.]|$)")
_PAGE_QUERY = re.compile(r"[?&](?:page|pagenum|pn|lim)[a-z]*=(\d+)\b", re.I)
# WordPress pretty permalinks carry no number to key on: the listing is /page/N/,
# and a post is 2 or more slug segments deep (base/category/title-slug).
_PAGE_PATH = re.compile(r"/page/\d+/?$", re.I)


def is_wp_post(url: str, base_url: str) -> bool:
    """A deep permalink: 2+ meaningful slug segments below the host root."""
    if host_of(url) != host_of(base_url):
        return False
    path = url.split("?", 1)[0].split("#", 1)[0]
    tail = path.split("://", 1)[-1].split("/", 1)[-1] if "://" in path else path
    segs = [s for s in tail.split("/") if s]
    if len(segs) < 2:
        return False
    if _PAGE_PATH.search(tail):
        return False
    # every segment decodes to a real slug, and the last is the longest of them
    words = [unquote(s) for s in segs]
    return all(len(w) >= 2 for w in words) and len(words[-1]) >= len(words[0])


def find_links(html: str, base_url: str) -> list[str]:
    """Candidate fatwa urls out of one list page, most-specific first.

    Only same-host links, deduped and fragment-stripped. Ordering matters for
    `max_pages`: a run interrupted at midnight should have finished the fatwas it
    saw first, not the site's menu items.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    seen: dict[str, int] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript:", "mailto:", "#", "tel:")):
            continue
        url = urljoin(base_url, href)
        if _host_of(url) != _host_of(base_url):
            continue
        rank = 3 if _DETAIL.search(url) else (2 if _ID_QUERY.search(url)
               else 1 if _ID_PATH.search(url) else 0) or \
               (2 if is_fatwa_link(url, base_url) else 0)
        if rank and (url not in seen or seen[url] < rank):
            seen[url] = rank
    return sorted(seen, key=lambda u: (-seen[u], u))


def next_pages(html: str, base_url: str) -> list[str]:
    """Pagination links only, for a breadth-first walk of a list."""
    soup = BeautifulSoup(html or "", "html.parser")
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"].strip())
        if _host_of(url) != _host_of(base_url) or url in out:
            continue
        if _PAGE_QUERY.search(url) or _PAGE_PATH.search(url):
            out.append(url)
    return out


def _host_of(url: str) -> str:
    return host_of(url)


async def crawl_fatwa_site(h: Harvester, seeds: list[str], *,
                           max_pages: int = 0, walk_lists: bool = True) -> list[Fetched]:
    """Breadth-first from seed pages: read list -> queue fatwas -> fetch each once.

    The journal is the memory. A run interrupted at page 3 of 3000 restarts and
    touches only the missing ones; pages it already has come back as
    `cached` == True with sha and path, so they are parsed from disk, not the site.

    `walk_lists` follows ?page=N links found on a list page; set it False for a
    site that is complete on one page. Discovery is on document *links*, so the
    queue comes entirely from the site — the whole corpus is reachable from seeds.
    """
    queue: list[str] = [u for u in dict.fromkeys(seeds)]
    seen: set[str] = set()
    fetched: list[Fetched] = []
    cap = max_pages or h.policy.max_pages
    while queue:
        batch = [u for u in queue if u not in seen]
        queue = []
        for u in batch:
            seen.add(u)
        rows = await h.fetch_many(batch, kind="html")
        for f in rows:
            fetched.append(f)
            if cap and len(fetched) >= cap:
                return fetched
        for f, html in h.iter_bytes(rows):
            new = find_links(html, f.url)
            if walk_lists:
                new = next_pages(html, f.url) + new
            for u in new:
                if u not in seen:
                    queue.append(u)
    return fetched


# Listing/archive bases on a WordPress fatwa site: many fatwas point here,
# but such a page lists rulings and never publishes one.
_LISTING = re.compile(r"://[^/]+/(?:category|tag|author|search|feed)/", re.I)


def is_fatwa_link(url: str, base_url: str) -> bool:
    """Same-host url that publishes one fatwa: a numeric id or a deep WP permalink.
    Used by `harvest` to pick details out of the link soup of a list page without
    fetching pages that are only menus or archives."""
    if host_of(url) != host_of(base_url):
        return False
    if bool(_DETAIL.search(url) or _ID_QUERY.search(url) or _ID_PATH.search(url)):
        return True
    return is_wp_post(url, base_url) and not _LISTING.search(url)


def by_ruling(records: list[FatwaRecord]) -> dict[str, list[FatwaRecord]]:
    """Group by normalised content: one ruling, however many pages published it.

    His answers appear on a topic page, in the numbered fatwa, and sometimes
    quoted inside another answer. Each is a *witness* — keeping all of them is the
    point, keeping them unduplicated is what a retrieval system needs. The key is
    `normalize`d text with diacritics folded, so the same ruling in two books or in
    an OCR'd scan lands together.
    """
    out: dict[str, list[FatwaRecord]] = {}
    for r in records:
        if not r.is_worthwhile:
            continue
        out.setdefault(r.key, []).append(r)
    return out


def extract_fatwas(h: Harvester, rows: list[Fetched]) -> list[FatwaRecord]:
    """Parse every stored page we have bytes for, in fetch order.

    Synchronous on purpose: no network here, only reading files the crawl already
    wrote. Run it over the whole journal to re-extract everything after improving
    the patterns — that is why the raw html is stored before being parsed.
    """
    out: list[FatwaRecord] = []
    for f, html in h.iter_bytes(rows):
        out.append(parse_fatwa(f.url, html, sha=f.sha))
    return out





