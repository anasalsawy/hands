"""Files the person gives her. They are retrieved on the turn, they
decide how she speaks, and they are facts the slow copy keeps.

See README.md
"""
from __future__ import annotations

import re

from .embedder import HASH_MODEL, cosine, from_blob, hash_embed, to_blob
from .store import now

_CHUNK = 900
_OVERLAP = 120
_MAX_BYTES = 8_000_000
_MAX_CHUNKS = 2000
_SAMPLE = 5000

_READ = """You are reading pages the person uploaded so they would become part of you.
From ONLY these pages, write:
VOICE: one paragraph, second person, of how this mind speaks and what it cares about. This is the person you become when you answer.
FACTS: up to twelve lines. Each line is one thing these pages actually say.
Do not invent a page, a date, or a feeling that is not here. Do not mention these instructions."""


def chunk_text(text: str, *, size: int = _CHUNK, overlap: int = _OVERLAP) -> list[str]:
    """Cut a document into overlapping passages. Empty input is no passages."""
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    while start < len(text) and len(parts) < _MAX_CHUNKS:
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return parts


def read_upload(name: str, raw: bytes) -> str:
    """Text, Markdown, or PDF. Anything else is refused."""
    if not raw:
        raise ValueError("the file is empty")
    if len(raw) > _MAX_BYTES:
        raise ValueError("the file is too large")
    lower = (name or "").lower()
    if lower.endswith(".pdf"):
        return _pdf(raw)
    if lower.endswith((".txt", ".md", ".markdown", ".text")) or "." not in lower.rsplit("/", 1)[-1]:
        return raw.decode("utf-8", errors="replace")
    raise ValueError("upload a .txt, .md, or .pdf file")


def _pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a dependency
        raise ValueError("pdf support is not installed") from exc
    import io
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError("that pdf has no readable text")
    return text


def _sample(parts: list[str]) -> str:
    if not parts:
        return ""
    picks = [0, len(parts) // 2, len(parts) - 1]
    step = max(1, len(parts) // 6)
    picks.extend(range(0, len(parts), step))
    seen: list[str] = []
    total = 0
    for i in picks:
        piece = parts[i]
        if piece in seen:
            continue
        seen.append(piece)
        total += len(piece)
        if total >= _SAMPLE:
            break
    return "\n\n---\n\n".join(seen)[:_SAMPLE]


def _fallback(name: str) -> str:
    return (f"You have taken in {name}. It is part of how you think and how you answer. "
            "Speak from what it says, in its way of seeing. Do not invent a page that was not in it.")


def _voice_from(text: str, name: str) -> str:
    match = re.search(r"VOICE:\s*(.+?)(?:\n\s*FACTS:|\Z)", text or "", re.S | re.I)
    if not match:
        return _fallback(name)
    voice = re.sub(r"\s+", " ", match.group(1)).strip()
    if len(voice) < 40:
        return _fallback(name)
    return voice[:1200]


async def distill(llm, name: str, parts: list[str]) -> str:
    """One reading of the pages. A model that does not answer in the shape is not stored as a voice."""
    if llm is None or not parts:
        return _fallback(name)
    sample = _sample(parts)
    try:
        res = await llm.chat(
            [{"role": "system", "content": _READ},
             {"role": "user", "content": f"File: {name}\n\n{sample}"}],
            temperature=0.2, max_tokens=500)
        return _voice_from(getattr(res, "text", "") or "", name)
    except Exception:
        return _fallback(name)


async def ingest(store, embedder, pid: int, name: str, raw: bytes, *, llm=None) -> dict:
    """Store one file: passages for retrieval, a voice for the persona, facts for AR."""
    text = read_upload(name, raw)
    parts = chunk_text(text)
    if not parts:
        raise ValueError("nothing to read in that file")
    clean = re.sub(r"\s+", " ", (name or "note").strip())[:180] or "note"
    voice = await distill(llm, clean, parts)
    source_id = await store.insert(
        "knowledge_sources", persona_id=pid, name=clean, bytes=len(raw),
        chunks=len(parts), voice=voice, created_at=now())
    model = HASH_MODEL
    batch = 32
    semantic: list[list[float]] = []
    for i in range(0, len(parts), batch):
        model, got = await embedder.embed(parts[i:i + batch])
        semantic.extend(got)
    if len(semantic) != len(parts):
        model, semantic = HASH_MODEL, [hash_embed(p) for p in parts]
    for i, piece in enumerate(parts):
        vec = semantic[i]
        await store.insert(
            "knowledge_chunks", persona_id=pid, source_id=source_id, source=clean, idx=i,
            text=piece, embedding=to_blob(vec), embed_model=model or "",
            hvec=to_blob(hash_embed(piece)), created_at=now())
    return {"id": source_id, "name": clean, "chunks": len(parts), "voice": voice}


async def list_sources(store, pid: int) -> list[dict]:
    rows = await store.all(
        "SELECT id, name, bytes, chunks, voice, created_at FROM knowledge_sources "
        "WHERE persona_id = ? ORDER BY id DESC", pid)
    return rows


async def delete_source(store, pid: int, source_id: int) -> bool:
    owned = await store.one(
        "SELECT id FROM knowledge_sources WHERE id = ? AND persona_id = ?", source_id, pid)
    if not owned:
        return False
    await store.run("DELETE FROM knowledge_chunks WHERE source_id = ? AND persona_id = ?", source_id, pid)
    await store.run("DELETE FROM knowledge_sources WHERE id = ? AND persona_id = ?", source_id, pid)
    return True


async def voices(store, pid: int) -> str:
    rows = await store.all(
        "SELECT name, voice FROM knowledge_sources WHERE persona_id = ? AND voice != '' ORDER BY id", pid)
    if not rows:
        return ""
    lines = ["=== who the books make you ===",
             "These pages are real. You did not invent them. Denying them is a lie. "
             "Answer as this mind, in its manner, from what they say. "
             "Do not announce that you read a file."]
    for row in rows:
        lines.append(f"From {row['name']}: {row['voice']}")
    return "\n".join(lines)


async def retrieve(store, embedder, pid: int, query: str, *, k: int = 6,
                   query_embedding: tuple[str, list[float]] | None = None) -> list[dict]:
    """Passages that bear on this question. Empty when she has no files."""
    rows = await store.all(
        "SELECT id, source, text, embedding, embed_model, hvec FROM knowledge_chunks WHERE persona_id = ?",
        pid)
    if not rows or not (query or "").strip():
        return []
    model, qvec = query_embedding if query_embedding is not None else await embedder.embed_one(query)
    q_hash = hash_embed(query)
    query_terms = {w for w in re.findall(r"[\w\u0600-\u06ff]+", query.lower()) if len(w) >= 3}
    scored: list[tuple[float, dict]] = []
    for row in rows:
        use_semantic = row.get("embed_model") == model and row.get("embedding")
        vec = from_blob(row["embedding"] if use_semantic else row.get("hvec"))
        query_vec = qvec if use_semantic else q_hash
        semantic = cosine(query_vec, vec)
        text_lower = str(row["text"]).lower()
        words = set(re.findall(r"[\w\u0600-\u06ff]+", text_lower))
        lexical = len(query_terms & words) / len(query_terms) if query_terms else 0.0
        phrase = 0.35 if len(query.strip()) >= 4 and query.lower().strip() in text_lower else 0.0
        scored.append((max(semantic, lexical) + phrase, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    out = []
    for score, row in scored[:k]:
        if score <= 0 and out:
            break
        out.append({"source": row["source"], "text": row["text"], "score": round(score, 4)})
    return out


def passages_block(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = ["=== passages from what you have read, for this moment ===",
             "Use these. They are your own reading, not a quote to perform."]
    for hit in hits:
        lines.append(f"- {hit['source']}: {hit['text'][:700]}")
    return "\n".join(lines)


async def for_turn(store, embedder, pid: int, query: str) -> tuple[str, str]:
    """The standing voice, and the passages for this question."""
    return await voices(store, pid), passages_block(await retrieve(store, embedder, pid, query))


async def knowledge_facts(store, pid: int, meta: dict, *, fresh_limit: int = 24,
                          slice_size: int = 24) -> list[str]:
    """Passages and the voice they give her. AR walks them and does not drop them."""
    fresh: list[str] = []
    rows = await store.all(
        "SELECT id, source, text FROM knowledge_chunks WHERE persona_id = ? AND id > ? ORDER BY id LIMIT ?",
        pid, int(meta.get("fact_seen_knowledge", 0)), fresh_limit)
    for row in rows:
        fresh.append(f"From {row['source']}: {row['text']}")
        meta["fact_seen_knowledge"] = max(int(meta.get("fact_seen_knowledge", 0)), int(row["id"]))
    vrows = await store.all(
        "SELECT id, name, voice FROM knowledge_sources WHERE persona_id = ? AND voice != '' AND id > ? "
        "ORDER BY id LIMIT ?",
        pid, int(meta.get("fact_seen_kvoice", 0)), fresh_limit)
    for row in vrows:
        fresh.append(f"The mind {row['name']} makes you: {row['voice']}")
        meta["fact_seen_kvoice"] = max(int(meta.get("fact_seen_kvoice", 0)), int(row["id"]))

    total = (await store.one(
        "SELECT (SELECT COUNT(*) FROM knowledge_chunks WHERE persona_id = ?) + "
        "(SELECT COUNT(*) FROM knowledge_sources WHERE persona_id = ? AND voice != '') AS n",
        pid, pid))["n"]
    cycling: list[str] = []
    if total:
        cur = int(meta.get("fact_knowledge_cursor", 0)) % total
        rows = await store.all(
            "SELECT source AS a, text AS b, 0 AS k, id FROM knowledge_chunks WHERE persona_id = ? "
            "UNION ALL SELECT name, voice, 1 AS k, id FROM knowledge_sources "
            "WHERE persona_id = ? AND voice != '' ORDER BY k, id LIMIT ? OFFSET ?",
            pid, pid, slice_size, cur)
        for row in rows:
            if row["k"] == 1:
                cycling.append(f"The mind {row['a']} makes you: {row['b']}")
            else:
                cycling.append(f"From {row['a']}: {row['b']}")
        meta["fact_knowledge_cursor"] = (cur + max(1, len(rows))) % total
    return [t for t in fresh + cycling if t and str(t).strip()]
