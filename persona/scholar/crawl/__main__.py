"""CLI: one polite run of the whole harvest. `python -m sawii_persona.scholar.crawl`

Everything is journalled under the corpus root, so re-running fills gaps and
refetches nothing; Ctrl-C mid-crawl is safe and the next run continues.

Sources and manners come from args or from the environment, so the same command
is a smoke run today and the full corpus tomorrow:

* `SAWII_CRAWL_SEEDS`     whitespace/comma-separated list-page urls of the site
* `SAWII_CRAWL_ROOT`      corpus dir: `manifest.jsonl` + content-addressed `data/`
* `SAWII_CRAWL_BOOKS`     a `title | url | note` file of book pdfs
* `SAWII_CRAWL_VIDEOS`    channel/playlist urls (audio needs yt-dlp; absent -> `pending`)
* `SAWII_CRAWL_DELAY` / `_CONCURRENCY` / `_MAX_PAGES` / `_AGENT` — read by
  `Policy.from_env`, the existing knob set.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict

from .base import Harvester, Policy
from .fatwas import by_ruling, crawl_fatwa_site, extract_fatwas


def _split(value) -> list[str]:
    """Args come as lists, env vars as comma-separated strings; both are urls."""
    if isinstance(value, (list, tuple)):
        value = " ".join(value)
    return [v for v in (value or "").replace(",", " ").split() if v]


def _load_env_file() -> None:
    """Read the repo `.env` if there is one, before any source is resolved.

    The harvest's knobs are ordinary env vars, so the owner can keep the site url
    and the corpus dir in the same `.env` the persona app already uses instead of
    re-typing them per run. `load_dotenv` never overrides a variable that is
    already set, so `--seed` / an explicit `$env:SAWII_CRAWL_SEEDS` still wins.
    The import is lazy on purpose: this package must stay usable with no persona
    config installed, and a missing one is simply "env comes from the shell".
    """
    try:
        from ...config import load_dotenv
        load_dotenv()
    except Exception:                 # noqa: BLE001 - env file is a convenience
        pass


async def _harvest(h: Harvester, seeds: list[str], books: str, videos: list[str],
                   json_out: str) -> list[str]:
    """Run whichever legs have sources. Each leg writes the same journal."""
    report = [f"corpus: {os.path.abspath(h.journal.root)}"]
    if seeds:
        rows = await crawl_fatwa_site(h, seeds)
        held = [f for f in rows if f.ok]
        fresh = sum(1 for f in held if not f.cached)
        dropped: dict[str, int] = {}
        for f in rows:
            if not f.ok:
                dropped[f.skipped or "status"] = dropped.get(f.skipped or "status", 0) + 1
        recs = extract_fatwas(h, rows)
        rulings = by_ruling(recs)
        review = sum(1 for r in recs if r.needs_review)
        line = (f"fatwas: {fresh} fetched, {len(held) - fresh} cached, "
                f"{len(rulings)} distinct rulings, {review} need review")
        if dropped:
            line += " — dropped: " + ", ".join(f"{k}×{v}" for k, v in sorted(dropped.items()))
        report.append(line)
        if json_out:
            with open(json_out, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            report.append(f"wrote {len(recs)} records -> {json_out}")
    if books:
        from .books import crawl_books, load_book_list      # lazy: pulls pdf deps late
        specs = load_book_list(books)
        pairs = await crawl_books(h, specs)
        ok = sum(1 for _, f in pairs if f.ok)
        report.append(f"books: {ok}/{len(specs)} stored, {len(specs) - len(pairs)} pending (no url yet)")
    if videos:
        from .videos import harvest_videos
        total = have = 0
        for src in videos:
            items = await harvest_videos(h, src)
            total += len(items)
            have += sum(1 for it in items if it.status in ("ok", "stored"))
        report.append(f"videos: {have}/{total} have audio stored"
                      " (pending re-runs on its own install yt-dlp to fill them)")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sawii_persona.scholar.crawl",
        description="Polite harvest of the scholar's own published corpus.")
    ap.add_argument("--root", default="", help="corpus dir (env SAWII_CRAWL_ROOT)")
    ap.add_argument("--seed", action="append", default=[],
                    help="list-page url, repeatable (env SAWII_CRAWL_SEEDS)")
    ap.add_argument("--max-pages", type=int, default=0, help="cap pages this run; 0 = env/default")
    ap.add_argument("--books", default="", help="book list file (env SAWII_CRAWL_BOOKS)")
    ap.add_argument("--video", action="append", default=[],
                    help="channel/playlist url, repeatable (env SAWII_CRAWL_VIDEOS)")
    ap.add_argument("--json", default="", help="also dump extracted fatwas as jsonl here")
    args = ap.parse_args(argv)

    _load_env_file()           # same .env the persona app reads, if present
    root = args.root or os.getenv("SAWII_CRAWL_ROOT", os.path.join("data", "scholar-corpus"))
    seeds = _split(args.seed) or _split(os.getenv("SAWII_CRAWL_SEEDS", ""))
    books = args.books or os.getenv("SAWII_CRAWL_BOOKS", "")
    videos = _split(args.video) or _split(os.getenv("SAWII_CRAWL_VIDEOS", ""))
    if not (seeds or books or videos):
        print("no sources: set SAWII_CRAWL_SEEDS (or pass --seed/--books/--video).\n"
              "manners come from SAWII_CRAWL_DELAY / _CONCURRENCY / _MAX_PAGES / _AGENT.",
              file=sys.stderr)
        return 2

    policy = Policy.from_env()
    if args.max_pages:
        policy.max_pages = args.max_pages
    lines = asyncio.run(_run(root, policy, seeds, books, videos, args.json))
    for line in lines:
        print(line)
    return 0


async def _run(root: str, policy: Policy, seeds: list[str], books: str,
               videos: list[str], json_out: str) -> list[str]:
    async with Harvester(root, policy) as h:
        return await _harvest(h, seeds, books, videos, json_out)


if __name__ == "__main__":
    sys.exit(main())
