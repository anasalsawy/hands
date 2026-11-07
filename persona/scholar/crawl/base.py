"""The polite core shared by the three harvests: robots first, per-host pacing,
sha-named storage, one append-only journal.

Holds no Arabic and judges no content. That is the harvesters' job; this module
only decides whether we may fetch, and remembers what we already have.
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from ..robots import Robots
from .urls import decode_html, normalize_url, now, sha256

log = logging.getLogger("sawii.crawl")

AGENT = "ScholarArchiveBot/1.1 (archiving a scholar's own published works)"
DELAY = 1.2
JOURNAL = "manifest.jsonl"
MAX_TRIES = 3
_EXT = {"text/html": ".html", "application/xhtml+xml": ".html", "application/pdf": ".pdf",
        "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
        "audio/webm": ".webm", "audio/ogg": ".ogg"}
_TYPES = {"html": ("text/html", "application/xhtml+xml"), "pdf": ("application/pdf",),
          "audio": ("audio/mpeg", "audio/mp4", "audio/x-m4a", "audio/webm", "audio/ogg"),
          "any": None}
_FALLBACK_EXT = {"html": ".html", "pdf": ".pdf", "audio": ".audio", "any": ".bin"}



def base_type(ctype: str) -> str:
    return (ctype or "").split(";")[0].strip().lower()


def accepts(kind: str, ctype: str) -> bool:
    """Ask for a pdf, get text/html, store nothing: that page is *about* the book."""
    allowed = _TYPES.get(kind)
    return True if allowed is None else base_type(ctype) in allowed


def suffix(kind: str, ctype: str, url: str) -> str:
    return _EXT.get(base_type(ctype)) or {
        "html": ".html", "pdf": ".pdf", "audio": ".bin", "any": ".bin"}[kind]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.0f} B"


@dataclass
class Policy:
    delay: float = DELAY
    timeout: float = 30.0
    concurrency: int = 4          # per host, small on purpose
    max_pages: int = 0            # 0 = uncapped: this harvest aims at 100%
    max_bytes: int = 64 * 1024 * 1024
    agent: str = AGENT

    @classmethod
    def from_env(cls) -> "Policy":
        """Manners from the environment. A bad number must say so and stop the run
        — crawling the site with `concurrency=1` because a typo silently fell back
        to 4, or the other way round, is not politeness, it is luck."""
        return cls(delay=float(os.getenv("SAWII_CRAWL_DELAY", DELAY)),
                   timeout=float(os.getenv("SAWII_CRAWL_TIMEOUT", 30.0)),
                   concurrency=max(1, int(os.getenv("SAWII_CRAWL_CONCURRENCY", "4"))),
                   max_pages=int(os.getenv("SAWII_CRAWL_MAX_PAGES", "0")),
                   max_bytes=int(os.getenv("SAWII_CRAWL_MAX_BYTES", str(64 * 1024 * 1024))),
                   agent=os.getenv("SAWII_CRAWL_AGENT", AGENT))


@dataclass
class Fetched:
    """What one url did. `skipped` is a reason for no file; it is not an error."""
    url: str
    status: int = 0
    sha: str = ""
    path: str = ""
    ctype: str = ""
    size: int = 0
    skipped: str = ""             # journal | robots | type | size | status | net | scheme | bad-url

    @property
    def ok(self) -> bool:
        """We hold this document, freshly fetched or from a previous run.

        `cached` is the journal hit; a sha is set whenever a page was stored by
        any earlier run too (type/size/robots/net obey not reach 200 and sha empty).
        A skipped reason literally means "no bytes to read", and that is fine.
        """
        return self.status == 200 and bool(self.sha)

    @property
    def cached(self) -> bool:
        return self.skipped == "journal"


@dataclass
class Journal:
    """Append-only record of every url. Resumable and auditable by the same file."""
    root: str

    def __post_init__(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, JOURNAL)
        self.done: dict[str, str] = {}
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("status") == 200 and rec.get("sha"):
                        self.done[rec.get("url", "")] = rec["sha"]

    def write(self, **rec) -> None:
        rec.setdefault("ts", round(now(), 3))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def done_url(self, url: str) -> str | None:
        """The sha already saved for this url, so a re-run skips without re-fetching."""
        return self.done.get(url)


class Harvester:
    """Async fetch-with-manners. Use as a context manager, then call `fetch`.

    All three harvests share one instance so pacing, robots cache and the journal
    stay global: certification is a site-level act, not a page-level one.
    """

    def __init__(self, root: str, policy: Policy | None = None):
        self.policy = policy or Policy.from_env()
        self.journal = Journal(root)
        self.data = os.path.join(root, "data")
        os.makedirs(self.data, exist_ok=True)
        self._client: httpx.AsyncClient | None = None
        self._robots: dict[str, Robots] = {}
        self._pace_locks: dict[str, asyncio.Lock] = {}
        self._last_hit: dict[str, float] = {}
        self._shas: dict[str, str] = {}      # sha -> path, this run's store too

    async def __aenter__(self) -> "Harvester":
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.policy.agent}, follow_redirects=True,
            timeout=self.policy.timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _load_robots(self, scheme: str, netloc: str) -> Robots:
        """Robots lives under the *same* scheme as the page, per RFC 9309 2.3.1.

        Hard-coding https would silently skip robots rules on plain-http hosts
        (a local test server, an intranet archive) or take the wrong host's rules:
        both are worse than missing obvious rules fall-back. Cache per URL root.
        """
        key = f"{scheme}://{netloc}"
        if key in self._robots:
            return self._robots[key]
        r = Robots(agent=self.policy.agent.split("/")[0])
        try:
            assert self._client
            resp = await self._client.get(f"{key}/robots.txt")
            if resp.status_code == 200:
                r.parse(resp.text)
        except (httpx.HTTPError, ValueError):
            pass                # missing robots.txt = all allowed, per RFC
        self._robots[key] = r
        return r

    async def _pace(self, scheme: str, netloc: str) -> None:
        """One request per host per delay; a longer Crawl-delay wins."""
        async with self._pace_locks.setdefault(netloc, asyncio.Lock()):
            wait = self.policy.delay
            r = self._robots.get(f"{scheme}://{netloc}")
            if r is not None and r.crawl_delay:
                wait = max(wait, r.crawl_delay)
            gap = self._last_hit.get(netloc, 0.0) + wait - now()
            if gap > 0:
                await asyncio.sleep(gap)
            self._last_hit[netloc] = now()

    def path_for(self, sha: str) -> str:
        """Where one document's bytes are, whoever wrote them and however named.

        Content-addressing means the *content* is the key, so the extension is a
        hint only: a fatwa served as text/html;charset=utf-8 and later as
        application/xhtml+xml is one file found by its sha, not two files by name.
        The journal may remember a path that no longer exists; the glob is the last
        answer. Returns "" when we genuinely have no copy.
        """
        if not sha:
            return ""
        if sha in self._shas and os.path.isfile(self._shas[sha]):
            return self._shas[sha]
        hit = next((p for p in glob.glob(os.path.join(self.data, sha + "*"))
                    if os.path.isfile(p)), "")
        if hit:
            self._shas[sha] = hit
        return hit

    def store_bytes(self, blob: bytes, kind: str, url: str = "") -> tuple[str, str]:
        """Content-addressed save. Returns (sha, path); path='' means already ours."""
        s = sha256(blob)
        known = self.path_for(s)
        if known:
            return s, ""                       # we hold these bytes from before
        p = os.path.join(self.data, s + suffix(kind, "", url))
        if os.path.isfile(p):
            self._shas[s] = p
            return s, ""                       # stored by an earlier run
        with open(p, "wb") as fh:
            fh.write(blob)
        self._shas[s] = p
        return s, p

    async def fetch(self, url: str, *, kind: str = "html") -> Fetched:
        """Fetch, store, journal one url. Never raises: a dead page is a record."""
        assert self._client is not None, "use Harvester as an async context manager"
        nu = normalize_url(url)
        f = Fetched(url=nu)
        loc = urlparse(nu)
        if loc.scheme not in ("http", "https") or not loc.netloc:
            f.skipped = "scheme"
            self.journal.write(url=nu, status=0, skipped="scheme")
            return f
        prior = self.journal.done_url(nu)
        if prior:
            f.status, f.sha, f.skipped = 200, prior, "journal"
            f.path = self.path_for(prior)     # by content, not by remembered extension
            return f
        netloc = loc.netloc.lower()
        rules = await self._load_robots(loc.scheme, netloc)
        if not rules.allowed(nu):
            f.skipped = "robots"
            self.journal.write(url=nu, status=0, skipped="robots")
            return f
        await self._pace(loc.scheme, netloc)
        last = ""
        for attempt in range(MAX_TRIES):
            try:
                resp = await self._client.get(nu)
                break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last = str(exc) or type(exc).__name__
                await asyncio.sleep(1.5 * (attempt + 1))
        else:
            f.skipped, f.status = "net", -1
            self.journal.write(url=nu, status=-1, skipped="net", error=last[:200])
            return f

        f.status, f.ctype, f.size = resp.status_code, base_type(
            resp.headers.get("content-type", "")), len(resp.content)
        if f.status != 200:
            f.skipped = "status"
        elif f.size > self.policy.max_bytes:
            f.skipped = "size"
        elif not accepts(kind, f.ctype):
            f.skipped = "type"
        if f.skipped:
            self.journal.write(url=nu, status=f.status, skipped=f.skipped,
                               ctype=f.ctype, bytes=f.size, kind=kind)
            return f

        f.sha, written = self.store_bytes(
            resp.content, kind, url="https://" + netloc + loc.path)
        f.path = self.path_for(f.sha)         # where it lives, new or kept from before
        self.journal.write(url=nu, status=200, sha=f.sha, saved=bool(written),
                           ctype=f.ctype, bytes=f.size, kind=kind)
        if written:
            log.info("saved %s %s (%s)", kind, nu, human(f.size))
        return f

    async def fetch_many(self, urls: list[str], *, kind: str = "html") -> list[Fetched]:
        """Concurrent fetch of a batch; the per-host lock keeps politeness intact."""
        sem = asyncio.Semaphore(self.policy.concurrency)

        async def one(u: str) -> Fetched:
            async with sem:
                return await self.fetch(u, kind=kind)

        return await asyncio.gather(*[one(u) for u in urls])

    def iter_bytes(self, rows: list[Fetched]) -> list[tuple[Fetched, str]]:
        """Load stored text bodies once, in order. Pages removed from fs are skipped quietly.

        encoding here is html-needed; reading Arabic web pages must not error out mid-list:
        """
        found: list[tuple[Fetched, str]] = []
        for f in rows:
            if not (f.ok and os.path.isfile(f.path)):
                continue
            try:
                with open(f.path, "rb") as fh:
                    blob = fh.read()
            except OSError:
                continue
            found.append((f, decode_html(blob, f.ctype)))
        return found


