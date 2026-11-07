"""Video harvest: a channel/playlist -> stored audio + per-video journal.

Two uses for the same bytes: voice-clone reference (his spoken style), and
transcripts that become searchable teaching. Transcribing (Whisper) is separate;
this only guarantees we hold the audio and know which video it came from.

Downloads need yt-dlp; if it is absent every video is journalled `pending`
rather than failing the run. Already-archived videos are skipped by the journal,
so a 500-video channel restarts where it stopped.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from .base import Harvester
from .urls import normalize_url

@dataclass
class VideoItem:
    """One recording. `path` is the stored audio file, empty until it exists."""

    url: str
    title: str = ""
    vid: str = ""
    duration: float = 0.0
    uploaded: str = ""
    channel: str = ""
    path: str = ""
    sha: str = ""
    status: str = ""              # ok | stored | pending | error


def list_videos(source: str) -> list[VideoItem]:
    """Titles/urls for a channel, playlist, or single video, via yt-dlp.

    Without yt-dlp there is no honest listing, so return one bare item: the
    harvest will then mark it `pending` and the journal says so plainly.
    """
    try:
        import yt_dlp
    except ImportError:
        return [VideoItem(url=source)]
    opts = {"extract_flat": "in_playlist", "quiet": True, "skip_download": True,
            "ignoreerrors": True}
    limit = os.getenv("SAWII_VIDEO_LIMIT", "")
    if limit:
        opts["playlist_items"] = limit
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=False) or {}
    entries = info.get("entries")
    items: list[VideoItem] = []
    for e in (list(entries) if entries else [info]):
        if not e:
            continue
        items.append(VideoItem(
            url=e.get("webpage_url") or e.get("url") or source,
            title=e.get("title") or "", vid=e.get("id") or "",
            duration=float(e.get("duration") or 0),
            uploaded=str(e.get("upload_date") or ""),
            channel=e.get("channel") or e.get("uploader") or ""))
    return items


async def _download_audio(h: Harvester, url: str) -> tuple[str, str]:
    """One yt-dlp fetch -> (stored path, sha), or ('', '') on absence/failure.

    Runs on a worker thread: yt-dlp blocks. The temp file is copied into the sha
    store and deleted, so the archive holds bytes by content, not by a temp name.
    """
    try:
        import yt_dlp
    except ImportError:
        return "", ""
    tmp = os.path.join(h.journal.root, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    opts = {"outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"), "format": "bestaudio/best",
            "quiet": True, "noplaylist": True, "restrictfilenames": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
        path = (info.get("requested_downloads") or [{}])[0].get("filepath", "")
        if not path or not os.path.isfile(path):
            return "", ""
        with open(path, "rb") as fh:
            blob = fh.read()
    except Exception:
        return "", ""
    sha, stored = h.store_bytes(blob, "audio", url=url)
    try:
        os.remove(path)
    except OSError:
        pass
    return stored, sha


def _have_video(h: Harvester, url: str) -> bool:
    """Journal test: this already sat under this URL. It allows resumability and replay."""
    return bool(h.journal.done_url(normalize_url(url)))


async def harvest_videos(h: Harvester, source: str) -> list[VideoItem]:
    """List one channel/playlist and pull the audio for every video we don't have.

    Listing and download run through yt-dlp off the event loop; results land in
    the sha-addressed data/ dir so a later transcription or voice-clone pass
    finds the audio by content, not by a URL that may stop working. Every item
    is journalled, so 'what do we have, what failed' is one grep away.
    """
    items = await asyncio.to_thread(list_videos, source)
    if not items:
        h.journal.write(event="videos", source=source, status="empty-listing")
        return items
    for it in items:
        if _have_video(h, it.url):
            it.status = "skipped"
            continue
        path, sha = await _download_audio(h, it.url)
        if path:
            it.path, it.sha, it.status = path, sha, "ok"
        else:
            it.status = "pending"          # no yt-dlp, or this host refused us
        h.journal.write(event="video", url=normalize_url(it.url), vid=it.vid,
                        title=it.title[:160], channel=it.channel,
                        duration=it.duration, status=it.status, sha=it.sha)
    return items


__all__ = ["VideoItem", "list_videos", "harvest_videos"]


