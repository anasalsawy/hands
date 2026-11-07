"""A voice the conversation layer speaks in.

One clip per persona. Each spoken sentence sends it as the reference.
Only a recording the person has the right to clone.
"""
from __future__ import annotations

import json
import os

_FORMATS = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".ogg": "ogg"}
_MAX = 12_000_000


def _dir(data_dir: str, pid: int) -> str:
    folder = os.path.join(data_dir, "voices", str(int(pid)))
    os.makedirs(folder, exist_ok=True)
    return folder


def save(data_dir: str, pid: int, raw: bytes, filename: str, *, reference_text: str = "") -> dict:
    ext = os.path.splitext(filename or "")[1].lower()
    fmt = _FORMATS.get(ext)
    if not fmt:
        raise ValueError("upload a wav, mp3, flac, or ogg clip")
    if len(raw) < 1000:
        raise ValueError("the clip is too short")
    if len(raw) > _MAX:
        raise ValueError("the clip is too long")
    folder = _dir(data_dir, pid)
    for name in os.listdir(folder):
        os.remove(os.path.join(folder, name))
    path = os.path.join(folder, f"clone.{fmt}")
    with open(path, "wb") as handle:
        handle.write(raw)
    meta = {"format": fmt, "bytes": len(raw), "name": os.path.basename(filename or f"clone.{fmt}")[:180],
            "reference_text": (reference_text or "").strip()[:2000]}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    return {"present": True, **meta}


def load(data_dir: str, pid: int) -> dict | None:
    folder = os.path.join(data_dir, "voices", str(int(pid)))
    meta_path = os.path.join(folder, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    fmt = meta.get("format") or ""
    path = os.path.join(folder, f"clone.{fmt}")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        data = handle.read()
    if not data:
        return None
    return {"data": data, "format": fmt, "reference_text": meta.get("reference_text") or "",
            "name": meta.get("name") or "", "bytes": len(data)}


def status(data_dir: str, pid: int) -> dict:
    clip = load(data_dir, pid)
    if clip is None:
        return {"present": False}
    return {"present": True, "format": clip["format"], "bytes": clip["bytes"], "name": clip["name"],
            "reference_text": clip["reference_text"]}


def clear(data_dir: str, pid: int) -> None:
    folder = os.path.join(data_dir, "voices", str(int(pid)))
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        os.remove(os.path.join(folder, name))
