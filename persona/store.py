"""Durable storage: the single source of truth.

SQLite in WAL mode, accessed through one serialized connection on a worker
thread. Every table is scoped by ``persona_id``; nothing crosses personas.
The domain logic lives in ``memory/`` and ``life/``; this module owns the
schema and the plumbing only.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- A persona is an API key: same key from any device = same being.
CREATE TABLE IF NOT EXISTS personas (
    id            INTEGER PRIMARY KEY,
    key_hash      TEXT NOT NULL UNIQUE,
    key_hint      TEXT NOT NULL DEFAULT '',
    name          TEXT NOT NULL DEFAULT '',
    definition    TEXT NOT NULL DEFAULT '',       -- written by the user (preset or clone)
    preset        TEXT NOT NULL DEFAULT '',
    user_name     TEXT NOT NULL DEFAULT '',
    timezone      TEXT NOT NULL DEFAULT 'UTC',
    contact       TEXT NOT NULL DEFAULT '{}',     -- {"sms": "+1..", "whatsapp": .., "email": .., "telegram": chat_id}
    preferred_channel TEXT NOT NULL DEFAULT 'app',
    a_model_override TEXT NOT NULL DEFAULT '',    -- fine-tuned adapter, plug and play
    narrative     TEXT NOT NULL DEFAULT '',       -- A's own first-person story, rewritten at sleep
    narrative_at  REAL NOT NULL DEFAULT 0,
    psych_profile TEXT NOT NULL DEFAULT '',       -- C's longitudinal read of the user (hedged)
    psych_at      REAL NOT NULL DEFAULT 0,
    mood          TEXT NOT NULL DEFAULT '{}',     -- A's current chord (declared + drift)
    user_mood     TEXT NOT NULL DEFAULT '{}',     -- user's chord as C reads it (rolling)
    baseline      BLOB,                            -- subconscious baseline (EMA of ordinary)
    baseline_model TEXT NOT NULL DEFAULT '',
    last_user_at  REAL NOT NULL DEFAULT 0,
    last_reply_at REAL NOT NULL DEFAULT 0,
    last_sleep_at REAL NOT NULL DEFAULT 0,
    c_expect      TEXT NOT NULL DEFAULT '',       -- C's prediction of what they'll say/feel next
    habits        TEXT NOT NULL DEFAULT '',       -- C's mirror: A's own habits and ruts (sleep)
    calibration   TEXT NOT NULL DEFAULT '',       -- B's map of where A tends to be wrong (sleep)
    created_at    REAL NOT NULL
);

-- The verbatim journal. Raw text is never rewritten. Quote-or-die grounds here.
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    user_text   TEXT NOT NULL,
    reply_text  TEXT NOT NULL DEFAULT '',
    channel     TEXT NOT NULL DEFAULT 'app',
    surface     TEXT NOT NULL DEFAULT '',       -- device/app: provenance only, never memory content
    kind        TEXT NOT NULL DEFAULT 'chat',   -- chat | outreach (persona spoke first)
    started_at  REAL NOT NULL,
    ended_at    REAL NOT NULL DEFAULT 0,
    gap_seconds REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_turns_p ON turns(persona_id, id);

-- One memory record per event (turn); layers hang off it. Recall returns the
-- whole stitched record: fact (A) + collateral (B) + affect/readings (C).
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    turn_id     INTEGER REFERENCES turns(id) ON DELETE SET NULL,
    chord       TEXT NOT NULL DEFAULT '{}',     -- the moment's feeling (C)
    surface     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_p ON events(persona_id, id);

CREATE TABLE IF NOT EXISTS layers (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    event_id    INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    layer       TEXT NOT NULL CHECK (layer IN ('fact','collateral','note','reading','afterthought')),
    author      TEXT NOT NULL CHECK (author IN ('A','B','C')),
    kind        TEXT NOT NULL DEFAULT 'detail', -- preference|plan|detail|aside|joke|callback|...
    text        TEXT NOT NULL,                  -- the author's statement
    quote       TEXT NOT NULL DEFAULT '',       -- user's verbatim words (required for user claims)
    anchor      TEXT NOT NULL DEFAULT '',       -- topic/event it attaches to
    chord       TEXT NOT NULL DEFAULT '{}',
    salience    INTEGER NOT NULL DEFAULT 1,
    status      TEXT NOT NULL DEFAULT 'active', -- active | superseded | wiped
    superseded_by INTEGER,
    interpretations TEXT NOT NULL DEFAULT '[]', -- reconsolidation: later readings; quote stays carved
    embedding   BLOB,
    embed_model TEXT NOT NULL DEFAULT '',
    hvec        BLOB,                            -- local hash vector, always present
    recalled_count INTEGER NOT NULL DEFAULT 0,
    last_recalled_at REAL NOT NULL DEFAULT 0,
    tender      INTEGER NOT NULL DEFAULT 0,      -- C: painful; held back unless the person raises it
    fidelity    TEXT NOT NULL DEFAULT 'full',   -- full|gist|trace: the overflowing cup
    evaporated_at REAL NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_layers_p ON layers(persona_id, status, id);

-- C's psyche layer: hypotheses about the person, never facts.
CREATE TABLE IF NOT EXISTS hypotheses (
    id            INTEGER PRIMARY KEY,
    persona_id    INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    claim         TEXT NOT NULL,
    quotes        TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0.4,
    confirmations INTEGER NOT NULL DEFAULT 1,
    contradictions INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'active', -- active | revised | faded
    revision_of   INTEGER,
    embedding     BLOB,
    embed_model   TEXT NOT NULL DEFAULT '',
    hvec          BLOB,
    last_confirmed_at REAL NOT NULL,
    created_at    REAL NOT NULL
);

-- People / projects as first-class things.
CREATE TABLE IF NOT EXISTS entities (
    id           INTEGER PRIMARY KEY,
    persona_id   INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'person',
    account      TEXT NOT NULL DEFAULT '',
    last_quote   TEXT NOT NULL DEFAULT '',
    mention_count INTEGER NOT NULL DEFAULT 1,
    updated_at   REAL NOT NULL,
    UNIQUE (persona_id, name)
);

-- B's open loops: mutable working state (tasks, promises, reminders). Not history.
CREATE TABLE IF NOT EXISTS loops (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'task',   -- task|promise|reminder|schedule|follow_up
    text        TEXT NOT NULL,
    quote       TEXT NOT NULL DEFAULT '',
    due_at      REAL,
    status      TEXT NOT NULL DEFAULT 'open',   -- open|done|missed|cancelled
    notified_at REAL NOT NULL DEFAULT 0,
    turn_id     INTEGER,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

-- Standing requests the person made about how to be talked to ("keep it short").
-- B keeps them and checks A against them while A is still speaking.
CREATE TABLE IF NOT EXISTS rules (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    quote       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',  -- active | dropped
    created_at  REAL NOT NULL
);

-- B's verdicts: the integrity record A sees next turn.
CREATE TABLE IF NOT EXISTS verdicts (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    turn_id     INTEGER,
    level       TEXT NOT NULL,                  -- GREEN | YELLOW | RED
    rationale   TEXT NOT NULL DEFAULT '',
    findings    TEXT NOT NULL DEFAULT '[]',
    corrected   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

-- The thought stream: an inner monologue the companion has and mostly never says.
-- The anchor: C's wrongness dataset, replayed during sleep. This is the outside
-- the mind cannot forge - the seed of the gate that will refuse a night's
-- learning when the mind has drifted. One row per rehearsal, so the score can
-- be watched over weeks.
CREATE TABLE IF NOT EXISTS anchor_runs (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    cases       INTEGER NOT NULL DEFAULT 0,
    passed      INTEGER NOT NULL DEFAULT 0,
    score       REAL,
    detail      TEXT NOT NULL DEFAULT '[]',
    created_at  REAL NOT NULL
);

-- Aging: one row per weight update C's verdict asked for. This is the record of
-- the mind actually changing, turn by turn - as opposed to the nightly bake.
CREATE TABLE IF NOT EXISTS updates (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    turn_id     INTEGER,
    level       TEXT NOT NULL DEFAULT '',
    reward      REAL NOT NULL DEFAULT 0.0,
    status      TEXT NOT NULL DEFAULT 'applied',   -- applied|failed
    detail      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS thoughts (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'reflection', -- reflection|recollection|sudden|dream|afterthought|correction|dropped|deep
    score       REAL NOT NULL DEFAULT 0.0,
    status      TEXT NOT NULL DEFAULT 'pending',-- pending|voiced|inhibited|decayed
    ref         TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);

-- Outbox: everything the persona says first.
CREATE TABLE IF NOT EXISTS outbox (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    channel     TEXT NOT NULL DEFAULT 'app',
    text        TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'queued', -- queued|sent|failed|read
    error       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    sent_at     REAL NOT NULL DEFAULT 0
);

-- The subconscious buffer: raw high-variance "juice" for nightly fine-tuning.
CREATE TABLE IF NOT EXISTS juice (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    turn_id     INTEGER,
    raw_transcript TEXT NOT NULL DEFAULT '',     -- byte-exact user turn
    reply       TEXT NOT NULL DEFAULT '',        -- A's reply (training target)
    style_markers TEXT NOT NULL DEFAULT '[]',
    variance    REAL NOT NULL,
    sensory_vector BLOB,
    tier        TEXT NOT NULL DEFAULT 'raw',     -- raw|exported|baked|compressed
    compressed  TEXT NOT NULL DEFAULT '',
    raw_chars   INTEGER NOT NULL DEFAULT 0,
    exported_at REAL NOT NULL DEFAULT 0,
    baked_at    REAL NOT NULL DEFAULT 0,
    compressed_at REAL NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS finetune_jobs (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    provider    TEXT NOT NULL,
    job_ref     TEXT NOT NULL DEFAULT '',
    dataset     TEXT NOT NULL DEFAULT '',
    juice_ids   TEXT NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL DEFAULT 'pending', -- pending|running|succeeded|failed
    output_model TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

-- Character anchor: tonal corrections harvested from the user. Parallel
-- to the wrongness dataset (which lives in verdicts), but keyed on
-- character/tone rather than factual correctness. The data is: a prior
-- Maya reply that produced a tonal correction, plus the correction's
-- direction (softer, warmer, less-chipper, etc.). Replayed at night to
-- grade whether the model is drifting on tone — the gate on the night's
-- character-layer bake.
CREATE TABLE IF NOT EXISTS character_anchors (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    turn_id     INTEGER,                           -- the turn whose reply drew the correction
    user_prompt TEXT NOT NULL,                     -- what the user originally asked
    reply       TEXT NOT NULL DEFAULT '',          -- Maya's reply that was corrected
    correction  TEXT NOT NULL,                     -- the user's correction text
    direction   TEXT NOT NULL DEFAULT '',          -- softer | warmer | less-chipper | ...
    kind        TEXT NOT NULL DEFAULT 'negative',  -- negative (do less of this) | positive (do more)
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_character_anchors_persona ON character_anchors(persona_id, id);

CREATE TABLE IF NOT EXISTS character_anchor_runs (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    cases       INTEGER NOT NULL DEFAULT 0,
    passed      INTEGER NOT NULL DEFAULT 0,
    score       REAL NOT NULL DEFAULT 0,
    detail      TEXT NOT NULL DEFAULT '[]',
    created_at  REAL NOT NULL
);

-- Deep lane jobs (OpenHands / parallel workers).
CREATE TABLE IF NOT EXISTS deep_jobs (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    task        TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'auto',
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|failed
    result      TEXT NOT NULL DEFAULT '',
    trace       TEXT NOT NULL DEFAULT '[]',
    seen        INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    finished_at REAL NOT NULL DEFAULT 0
);

-- Books and notes the person uploaded. They are retrieved on the turn (RAG)
-- and walked slowly into the facts copy. The voice column is the mind those
-- pages make her.
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    bytes       INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    voice       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_knowledge_sources_p ON knowledge_sources(persona_id, id);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB,
    embed_model TEXT NOT NULL DEFAULT '',
    hvec        BLOB,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_p ON knowledge_chunks(persona_id, id);

-- ============================================================================
-- The scholar corpus (ARCHITECTURE.md section 11). A ruling is not a chunk.
-- It is a claim with a witness: who said it, where, when, and whether a later
-- saying of his overrides it. knowledge_chunks stays the general book carrier;
-- rulings is the authoritative record the answer-authority gate reads.
-- ============================================================================
CREATE TABLE IF NOT EXISTS rulings (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    source_id   INTEGER REFERENCES knowledge_sources(id) ON DELETE SET NULL,
    kind        TEXT NOT NULL DEFAULT 'fatwa',      -- fatwa|book_passage|lecture|written_qa
    topic       TEXT NOT NULL DEFAULT '',           -- fiqh chapter: tahara|salah|zakat|mirath|...
    question    TEXT NOT NULL DEFAULT '',           -- as it was put to him
    answer      TEXT NOT NULL,                      -- his words. Never paraphrased here.
    citation    TEXT NOT NULL DEFAULT '',           -- the form he would write: "al-Majmu' 7/158"
    cite_json   TEXT NOT NULL DEFAULT '{}',         -- structured coordinates (scholar/citations.py)
    issued_at   REAL NOT NULL DEFAULT 0,            -- when he said it; drives revision order
    revised_from INTEGER REFERENCES rulings(id) ON DELETE SET NULL,
    group_id    TEXT NOT NULL DEFAULT '',           -- content identity: same ruling, two witnesses
    derived_from TEXT NOT NULL DEFAULT '[]',        -- ids this ruling synthesises (section 11 DERIVED)
    norm        TEXT NOT NULL DEFAULT '',           -- normalized Arabic, for lexical reach
    embedding   BLOB,
    embed_model TEXT NOT NULL DEFAULT '',
    hvec        BLOB,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_rulings_p ON rulings(persona_id, id);
CREATE INDEX IF NOT EXISTS ix_rulings_topic ON rulings(persona_id, topic);
CREATE INDEX IF NOT EXISTS ix_rulings_group ON rulings(persona_id, group_id);

-- Corpus-scale lexical reach. The Python scan in knowledge.retrieve() is fine
-- for a shelf of books; it is not fine for a lifetime. FTS5 is the term index,
-- and it is kept in step by triggers so no writer can forget it.
CREATE VIRTUAL TABLE IF NOT EXISTS rulings_fts USING fts5(
    question, answer, topic, citation,
    content='rulings', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS rulings_ai AFTER INSERT ON rulings BEGIN
    INSERT INTO rulings_fts(rowid, question, answer, topic, citation)
    VALUES (new.id, new.question, new.answer, new.topic, new.citation);
END;
CREATE TRIGGER IF NOT EXISTS rulings_ad AFTER DELETE ON rulings BEGIN
    INSERT INTO rulings_fts(rulings_fts, rowid, question, answer, topic, citation)
    VALUES ('delete', old.id, old.question, old.answer, old.topic, old.citation);
END;
CREATE TRIGGER IF NOT EXISTS rulings_au AFTER UPDATE ON rulings BEGIN
    INSERT INTO rulings_fts(rulings_fts, rowid, question, answer, topic, citation)
    VALUES ('delete', old.id, old.question, old.answer, old.topic, old.citation);
    INSERT INTO rulings_fts(rowid, question, answer, topic, citation)
    VALUES (new.id, new.question, new.answer, new.topic, new.citation);
END;

-- NOT_ADDRESSED is harvest, not a dead end (ARCHITECTURE.md section 11).
-- Every question the corpus could not answer is queued to the scholar himself.
-- When he answers, that answer becomes a first-class ruling and the gap is
-- closed forever. No row is deleted when it expires - it stays as the record
-- of what he was never asked, which is itself knowledge.
CREATE TABLE IF NOT EXISTS escalations (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    turn_id     INTEGER,
    question    TEXT NOT NULL,
    asked_by    TEXT NOT NULL DEFAULT '',        -- who asked (contact/hash), for the reply-back
    attempted   TEXT NOT NULL DEFAULT '[]',      -- what retrieval touched, and why it was not enough
    majority    TEXT NOT NULL DEFAULT '',        -- the stand-in answer that was given meanwhile
    status      TEXT NOT NULL DEFAULT 'open',    -- open|sent|answered|ingested|declined|expired
    via         TEXT NOT NULL DEFAULT '',        -- sms|whatsapp|email|telegram|app|hand
    answered_by TEXT NOT NULL DEFAULT '',        -- his answer, verbatim, before ingestion
    ruling_id   INTEGER REFERENCES rulings(id) ON DELETE SET NULL,
    notified    INTEGER NOT NULL DEFAULT 0,      -- the original asker has been told the reply came
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    sent_at     REAL NOT NULL DEFAULT 0,
    answered_at REAL NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_escalations_open ON escalations(persona_id, status, id);
"""


# additive migrations for databases created by earlier versions
MIGRATIONS = [
    ("layers", "tender", "INTEGER NOT NULL DEFAULT 0"),
    ("personas", "c_expect", "TEXT NOT NULL DEFAULT ''"),
    ("personas", "habits", "TEXT NOT NULL DEFAULT ''"),
    ("personas", "calibration", "TEXT NOT NULL DEFAULT ''"),
    ("personas", "anchor_score", "REAL NOT NULL DEFAULT 0"),
    ("personas", "anchor_at", "REAL NOT NULL DEFAULT 0"),
    ("layers", "fidelity", "TEXT NOT NULL DEFAULT 'full'"),
    ("layers", "evaporated_at", "REAL NOT NULL DEFAULT 0"),
# Character anchor (tonal corrections) — the second gate on nightly
    # consolidation, parallel to the factual anchor.
    ("personas", "character_anchor_score", "REAL NOT NULL DEFAULT 0"),
    ("personas", "character_anchor_at", "REAL NOT NULL DEFAULT 0"),
    # Shaped-me on the verdict: when C sees an exchange that moved B's chord
    # significantly, this flag flips. Trained alongside the correctness
    # weight_gate but on a different adapter.
    ("verdicts", "shaped_me", "INTEGER NOT NULL DEFAULT 0"),
    # Maya's own last-turn declared chord (from A's ⟦feel: …⟧ suffix).
    # Carries her own tone across turns, same way last_chord carries her
    # read of the user. Not an instruction — background data A's input
    # assembly loads as "last turn you felt X" so the next beat starts
    # from the mood she ended on, not from a fresh zero.
    ("personas", "last_a_chord", "TEXT NOT NULL DEFAULT '{}'"),
    # Quick bake schedule: last time a quick consolidation was run (for 30-min cycle).
    ("personas", "last_quick_bake_at", "REAL NOT NULL DEFAULT 0"),
    # Fast-weight delta (L): per-turn volatile updates via divergence (decays per EMA).
    ("personas", "a_fast_weight_ref", "TEXT NOT NULL DEFAULT ''"),
    # Slow-weight delta (R): EMA consolidation from L (the lasting learned shape).
    ("personas", "a_slow_weight_ref", "TEXT NOT NULL DEFAULT ''"),
    # Chord-delta signal for character-change aging (see README.md
    # discussion 2026-10-21 — the second gate alongside C's verdict).
    ("personas", "last_chord", "TEXT NOT NULL DEFAULT '{}'"),  # the chord carried turn-to-turn
    ("juice", "chord_delta", "REAL NOT NULL DEFAULT 0"),       # how much this exchange moved B's read
    ("juice", "shaped_me", "INTEGER NOT NULL DEFAULT 0"),      # 1 when chord shifted beyond the floor
    ("juice", "chord_before", "TEXT NOT NULL DEFAULT '{}'"),   # B's chord before this exchange
    ("juice", "chord_after", "TEXT NOT NULL DEFAULT '{}'"),    # B's chord after this exchange
    # The picture of them: who they are, who is around them, what is heavy,
    # what is owed. Read on every turn from the CACHED prefix of the prompt, so
    # knowing all of it costs nothing per message. Rewritten rarely, by B,
    # additively - a line is replaced, never silently dropped.
    ("personas", "their_picture", "TEXT NOT NULL DEFAULT ''"),
    ("personas", "picture_at", "REAL NOT NULL DEFAULT 0"),
]


class Store:
    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            for table, col, ddl in MIGRATIONS:
                cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                if col not in cols:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    # -- sync core ---------------------------------------------------------
    def _run(self, sql: str, params: Iterable[Any] = (), *, many: bool = False):
        with self._lock:
            cur = self._conn.executemany(sql, params) if many else self._conn.execute(sql, tuple(params))
            return cur

    def _fetchall(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def _fetchone(self, sql, params=()):
        with self._lock:
            r = self._conn.execute(sql, tuple(params)).fetchone()
            return dict(r) if r else None

    def _insert(self, table: str, values: dict) -> int:
        cols = ", ".join(values)
        qs = ", ".join("?" for _ in values)
        with self._lock:
            cur = self._conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(values.values()))
            return int(cur.lastrowid)

    def _update(self, table: str, row_id: int, values: dict, *, persona_id: int | None = None) -> int:
        sets = ", ".join(f"{k} = ?" for k in values)
        sql = f"UPDATE {table} SET {sets} WHERE id = ?"
        params = [*values.values(), row_id]
        if persona_id is not None:
            sql += " AND persona_id = ?"
            params.append(persona_id)
        with self._lock:
            return self._conn.execute(sql, params).rowcount

    def _tx(self, fn):
        """Run ``fn(conn)`` inside one transaction."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                out = fn(self._conn)
                self._conn.execute("COMMIT")
                return out
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # -- async facade ------------------------------------------------------
    async def all(self, sql: str, *params) -> list[dict]:
        return await asyncio.to_thread(self._fetchall, sql, params)

    async def one(self, sql: str, *params) -> dict | None:
        return await asyncio.to_thread(self._fetchone, sql, params)

    async def run(self, sql: str, *params) -> int:
        cur = await asyncio.to_thread(self._run, sql, params)
        return cur.rowcount

    async def insert(self, table: str, **values) -> int:
        return await asyncio.to_thread(self._insert, table, values)

    async def update(self, table: str, row_id: int, persona_id: int | None = None, **values) -> int:
        return await asyncio.to_thread(self._update, table, row_id, values, persona_id=persona_id)

    async def tx(self, fn):
        return await asyncio.to_thread(self._tx, fn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def now() -> float:
    return time.time()


def jdump(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def jload(v: str | None, default: Any = None) -> Any:
    if not v:
        return default
    try:
        return json.loads(v)
    except (TypeError, json.JSONDecodeError):
        return default
