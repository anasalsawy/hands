"""Harvest the scholar's corpus: fatwas from his site, his videos, his books.

One polite async core (`Harvester`), three harvests, one journal. What
"100%, nothing lost, nothing duplicated" means in code:

* every url lands in `manifest.jsonl` with a status: completeness is one query
  away, and same bytes = same sha256, so a fatwa that moved pages is one document;
* a url recorded ok is never re-fetched, so re-running the crawl fills the gaps
  instead of overwriting answers a human may have edited since;
* a robots-disallowed url is recorded `skipped_robots`, never silently dropped;
* raw html/audio/pdf are stored *before* anything is parsed, so a better parser
  never needs the network again.

The crawlers save bytes and urls. They do not decide truth: extraction into
rulings is `fatwas.extract_fatwas`, and everything past that is the store's job.
"""
from __future__ import annotations

from .base import Fetched, Harvester, Journal, Policy
from .books import BookSpec, crawl_books, load_book_list
from .fatwas import FatwaRecord, crawl_fatwa_site, extract_fatwas
from .videos import VideoItem, harvest_videos

__all__ = [
    "Fetched", "Harvester", "Journal", "Policy",
    "BookSpec", "crawl_books", "load_book_list",
    "FatwaRecord", "crawl_fatwa_site", "extract_fatwas",
    "VideoItem", "harvest_videos",
]
