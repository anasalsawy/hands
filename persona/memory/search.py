"""One retrieval entrypoint: semantic + lexical long-term memory, document passages,
and verbatim conversation history. All searches stay scoped to one persona.
No reasoning-model hop, no hidden second memory tool.
"""
from __future__ import annotations

import asyncio
import re

from ..knowledge import retrieve
from .records import content_words


def _terms(query: str) -> list[str]:
    # Preserve multilingual word search; avoid SQL wildcard interpretation.
    terms = [w for w in content_words(query) if len(w) >= 3]
    if not terms:
        terms = [w for w in re.findall(r"[\w\u0600-\u06ff]+", query.lower()) if len(w) >= 2]
    return sorted(set(terms), key=lambda w: -len(w))[:8]


async def _journal(store, pid: int, query: str, *, exclude_turn_id: int | None = None,
                   limit: int = 6) -> list[dict]:
    """Find exact words/phrases in historical raw turns, including turns not distilled by B."""
    terms = _terms(query)
    if not terms:
        return []
    where = " OR ".join("(lower(user_text) LIKE ? ESCAPE '\\' OR lower(reply_text) LIKE ? ESCAPE '\\')" for _ in terms)
    params: list = [pid]
    for term in terms:
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.extend([f"%{escaped}%", f"%{escaped}%"])
    exclude = " AND id != ?" if exclude_turn_id is not None else ""
    if exclude_turn_id is not None:
        params.append(exclude_turn_id)
    rows = await store.all(
        f"SELECT id, user_text, reply_text, started_at FROM turns "
        f"WHERE persona_id = ? AND ({where}){exclude} ORDER BY id DESC LIMIT 250",
        *params)
    q = query.strip().lower()
    scored = []
    for row in rows:
        user = str(row.get("user_text") or "")
        reply = str(row.get("reply_text") or "")
        combined = (user + "\n" + reply).lower()
        matches = sum(1 for word in terms if word in combined)
        # Exact phrase beats scattered terms, but newer entries break ties.
        score = matches / len(terms) + (1.0 if len(q) >= 4 and q in combined else 0.0)
        scored.append((score, row["id"], row))
    scored.sort(key=lambda hit: (-hit[0], -hit[1]))
    return [row for _, _, row in scored[:limit]]


async def search_all(mind, persona: dict, query: str, *, exclude_turn_id: int | None = None,
                     budget: int | None = None) -> str:
    """Single search_memory implementation, across every indexed and exact-text source.

    One query embedding is shared by memory-layer and document searches. Keyword
    matches remain available when remote embeddings are unavailable.
    """
    q = (query or "").strip()[:2000]
    if not q:
        return '{"error":"missing memory query"}'
    pid = persona["id"]
    requested = budget if budget is not None else mind.s.recall_budget_chars
    cap = requested if requested is not None and requested > 0 else None
    query_embedding = await mind.embedder.embed_one(q)
    results = await asyncio.gather(
        mind.recall.recall(persona, q, budget=cap, exclude_turn_ids={exclude_turn_id}
                           if exclude_turn_id is not None else None,
                           query_embedding=query_embedding),
        retrieve(mind.store, mind.embedder, pid, q, k=6,
                 query_embedding=query_embedding),
        _journal(mind.store, pid, q, exclude_turn_id=exclude_turn_id),
    )
    memory, passages, turns = results
    sections = []
    if memory.text:
        sections.append("LONG-TERM MEMORIES (semantic + keyword + entity):\n" + memory.text)
    if passages:
        sections.append("DOCUMENT PASSAGES (semantic + exact words):\n" +
                        "\n".join(f"- [{row['source']}] {row['text'][:700]}"
                                  for row in passages))
    if turns:
        sections.append("ORIGINAL CONVERSATION MATCHES (verbatim, not inferred):\n" +
                        "\n".join(f"- [turn {row['id']}] user: {str(row.get('user_text') or '')[:550]}"
                                  + (f" | assistant: {str(row.get('reply_text') or '')[:350]}"
                                     if row.get('reply_text') else "")
                                  for row in turns))
    if not sections:
        return "No matching memory, passage, or conversation found."
    # Per-source budgets guarantee neither documents nor history crowd out memory.
    return "\n\n".join(sections)
