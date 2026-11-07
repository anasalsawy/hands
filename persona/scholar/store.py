"""The scholar's corpus tables, in the same sqlite file as everything else.

``store.Store`` already owns the file: personas, turns, memory layers - and, since
the corpus ruling in ARCHITECTURE.md section 11, ``rulings`` and ``escalations``.
This module does not restructure them. It opens the same file, brings those two
tables to the shape the harvest needs (additive columns, one UNIQUE witness key, a
FTS index over *normalised* text), and exposes the few calls the crawler writes
through.

Everything here is idempotent - ``CREATE ... IF NOT EXISTS``, ``ALTER`` only for a
column that is missing - because the crawl worker and the persona are separate
processes sharing one WAL database, and either may open it first.

Sync and pure sqlite on purpose: extraction is a single-threaded walk over bytes we
already own, so the async facade would only add a thread hop, and a ruling that
cannot be re-read from a plain file is a ruling nobody can audit.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Iterable

from .normalize import content_hash, normalize, stems, tokens

# Below this extraction score a page is a human's job before the persona quotes it
# (fatwas.parse_fatwa sets the same flag from the same rule).
REVIEW_BELOW = 0.8

# store.py's corpus DDL, restated so a corpus-only database can begin. Harmless when
# store.py got there first: every statement is IF NOT EXISTS and matches its version.
# No `personas` stub here on purpose. Creating a *looser* copy of a table another
# owner declares would block their real schema forever (CREATE IF NOT EXISTS skips),
# and this connection runs with foreign_keys off, so rulings.persona_id = 0 is legal
# with no parent row.
_BASE = """
CREATE TABLE IF NOT EXISTS rulings (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL DEFAULT 0,
    source_id   INTEGER,
    kind        TEXT NOT NULL DEFAULT 'fatwa',
    topic       TEXT NOT NULL DEFAULT '',
    question    TEXT NOT NULL DEFAULT '',
    answer      TEXT NOT NULL DEFAULT '',
    citation    TEXT NOT NULL DEFAULT '',
    cite_json   TEXT NOT NULL DEFAULT '{}',
    issued_at   REAL NOT NULL DEFAULT 0,
    revised_from INTEGER,
    group_id    TEXT NOT NULL DEFAULT '',
    derived_from TEXT NOT NULL DEFAULT '[]',
    norm        TEXT NOT NULL DEFAULT '',
    embedding   BLOB,
    embed_model TEXT NOT NULL DEFAULT '',
    hvec        BLOB,
    created_at  REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS escalations (
    id          INTEGER PRIMARY KEY,
    persona_id  INTEGER NOT NULL DEFAULT 0,
    turn_id     INTEGER,
    question    TEXT NOT NULL,
    asked_by    TEXT NOT NULL DEFAULT '',
    attempted   TEXT NOT NULL DEFAULT '[]',
    majority    TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'open',
    via         TEXT NOT NULL DEFAULT '',
    answered_by TEXT NOT NULL DEFAULT '',
    ruling_id   INTEGER,
    notified    INTEGER NOT NULL DEFAULT 0,
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL DEFAULT 0,
    sent_at     REAL NOT NULL DEFAULT 0,
    answered_at REAL NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0
);
"""

# What the harvest adds on top. Pairs of (table, column, ddl); a column that exists
# is never touched, so an edited-by-hand database survives the upgrade untouched.
_EXTRA: list[tuple[str, str, str]] = [
    ("rulings", "source_url", "TEXT NOT NULL DEFAULT ''"),
    ("rulings", "sha", "TEXT NOT NULL DEFAULT ''"),
    ("rulings", "fatwa_id", "TEXT NOT NULL DEFAULT ''"),
    ("rulings", "title", "TEXT NOT NULL DEFAULT ''"),
    # display text (`question`, `answer`) stays his letters; *_norm is the matching
    # key. Both stored, because a search that runs on display text loses every
    # ruling the moment one page is scanned without its tashkeel.
    ("rulings", "question_norm", "TEXT NOT NULL DEFAULT ''"),
    ("rulings", "answer_norm", "TEXT NOT NULL DEFAULT ''"),
    ("rulings", "score", "REAL NOT NULL DEFAULT 0"),
    ("rulings", "needs_review", "INTEGER NOT NULL DEFAULT 1"),
    ("rulings", "authority", "TEXT NOT NULL DEFAULT ''"),
    ("rulings", "dedup_group", "TEXT NOT NULL DEFAULT ''"),
    ("escalations", "question_norm", "TEXT NOT NULL DEFAULT ''"),
    ("escalations", "asked_via", "TEXT NOT NULL DEFAULT ''"),
    ("escalations", "ts", "REAL NOT NULL DEFAULT 0"),
    ("escalations", "answer", "TEXT NOT NULL DEFAULT ''"),
    ("escalations", "source_url", "TEXT NOT NULL DEFAULT ''"),
]

# Written AFTER the columns exist: a trigger body naming a missing column fails at
# CREATE time, not at insert time, which would look like a broken schema.
_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_rulings_p ON rulings(persona_id, id);
CREATE INDEX IF NOT EXISTS ix_rulings_group ON rulings(persona_id, group_id);
CREATE INDEX IF NOT EXISTS ix_rulings_review ON rulings(needs_review, score);

-- One witness per (bytes, content identity). The same page ingested twice collides
-- here and counts as a duplicate; the same *ruling* republished under a new url has
-- a new sha and the same dedup_group, so it joins the group instead of becoming a
-- second, competing answer. Partial: UNIQUE rejects a second ('','') pair, which is
-- what every column default is, so store.py's earlier witness-less rows would all
-- collide at the first insert. Constrained only rows that carry witness identity.
CREATE UNIQUE INDEX IF NOT EXISTS ux_rulings_witness ON rulings(sha, dedup_group)
    WHERE sha <> '' AND dedup_group <> '';
-- Non-unique twin, so the duplicate lookup in ingest() stays indexed even for the
-- blank-identity rows the partial guard lets through.
CREATE INDEX IF NOT EXISTS ix_rulings_witness ON rulings(sha, dedup_group);

-- Lexical reach over normalised text.
CREATE VIRTUAL TABLE IF NOT EXISTS rulings_norm_fts USING fts5(
    question_norm, answer_norm,
    content='rulings', content_rowid='id', tokenize='unicode61'
);

INSERT INTO rulings_norm_fts(rulings_norm_fts) VALUES ('rebuild');

CREATE TRIGGER IF NOT EXISTS rulings_norm_ai AFTER INSERT ON rulings BEGIN
    INSERT INTO rulings_norm_fts(rowid, question_norm, answer_norm)
    VALUES (new.id, new.question_norm, new.answer_norm);
END;
CREATE TRIGGER IF NOT EXISTS rulings_norm_ad AFTER DELETE ON rulings BEGIN
    INSERT INTO rulings_norm_fts(rulings_norm_fts, rowid, question_norm, answer_norm)
    VALUES ('delete', old.id, old.question_norm, old.answer_norm);
END;
CREATE TRIGGER IF NOT EXISTS rulings_norm_au AFTER UPDATE OF question_norm, answer_norm ON rulings BEGIN
    INSERT INTO rulings_norm_fts(rulings_norm_fts, rowid, question_norm, answer_norm)
    VALUES ('delete', old.id, old.question_norm, old.answer_norm);
    INSERT INTO rulings_norm_fts(rowid, question_norm, answer_norm)
    VALUES (new.id, new.question_norm, new.answer_norm);
END;

CREATE INDEX IF NOT EXISTS ix_escalations_p ON escalations(persona_id, status, id);
CREATE INDEX IF NOT EXISTS ix_escalations_norm ON escalations(question_norm, status);
"""
STATUS_OPEN, STATUS_ANSWERED = "open", "answered"


def _is_text_affinity(decl: str) -> bool:
    d = (decl or "").upper()
    return "TEXT" in d or "CHAR" in d or "CLOB" in d or d == ""


class ScholarStore:
    """The rulings/escalations face of the shared database.

    Open it with whatever path the persona uses (`config.db_path`, tests use
    ``tmp_path``); IDs come back as sqlite rowids, and every reader returns plain
    dicts so the crawler worker never has to know this class exists.
    """

    def __init__(self, path: str, *, persona_id: int = 0):
        self.path = path
        self.persona_id = int(persona_id or 0)
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        # foreign_keys is deliberately OFF on this connection: rulings.persona_id
        # references personas, and the harvest has no business inventing a persona
        # row just to file a fatwa.
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._apply()
        # store.py declares issued_at REAL; a corpus-only file leaves it TEXT. The
        # crawler writes whatever the page printed ("12/3/2024"), so convert per
        # column type rather than splitting one field across two tables.
        self._issued_is_text = _is_text_affinity(self._declared("rulings", "issued_at"))

    # -- plumbing ------------------------------------------------------------
    def _declared(self, table: str, column: str) -> str:
        for row in self._conn.execute(f"PRAGMA table_info({table})"):
            if row["name"] == column:
                return str(row["type"] or "")
        return ""

    def _apply(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.executescript(_BASE)
        fresh = False
        for table, col, ddl in _EXTRA:
            cols = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if col in cols:
                continue
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            fresh = fresh or table == "rulings"
        had_fts = self._one("SELECT COUNT(*) AS n FROM sqlite_master"
                            " WHERE type='table' AND name='rulings_norm_fts'") or {"n": 1}
        self._conn.executescript(_INDEXES)
        self._backfill(fresh or not int(had_fts["n"]))

    def _backfill(self, fresh: bool) -> None:
        """Derive *_norm/dedup_group on rows written before these columns existed.

        The triggers are already installed when this runs, so each UPDATE reindexes its
        row in FTS. The one global 'rebuild' happens only when the norm columns or the
        index were just added — that is when the index is empty while the table is not,
        and never pays for a corpus that is already current.
        """
        rows = self._all("SELECT id, question, answer FROM rulings WHERE answer_norm = ''")
        for row in rows:
            q = str(row.get("question") or "")
            a = str(row.get("answer") or "")
            self._conn.execute("UPDATE rulings SET question_norm = ?, answer_norm = ?,"
                               " dedup_group = COALESCE(NULLIF(dedup_group, ''), ?)"
                               " WHERE id = ?",
                               (normalize(q), normalize(a), content_hash(a), int(row["id"])))
        if fresh or rows:
            self._conn.execute("INSERT INTO rulings_norm_fts(rulings_norm_fts) VALUES ('rebuild')")

    def close(self) -> None:
        self._conn.close()

    def _tx(self, fn):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            out = fn(self._conn)
            self._conn.execute("COMMIT")
            return out
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _row(cur: sqlite3.Cursor) -> dict | None:
        r = cur.fetchone()
        return dict(r) if r else None

    def _one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        return self._row(self._conn.execute(sql, tuple(params)))

    def _all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    # -- writes --------------------------------------------------------------
    def ingest(self, records: Iterable[Any]) -> tuple[int, int]:
        """Write extracted fatwas. Returns (inserted, duplicates).

        Dedup happens twice on purpose: ``UNIQUE(sha, dedup_group)`` rejects the
        same page re-ingested from the same bytes, and `dedup_group` — the
        content hash of the normalised answer — is what recognises one ruling
        republished under a new url. A duplicate joins the group it already
        belongs to rather than overwriting a row a human may have edited.
        """
        inserted = duplicates = 0
        ts = time.time()

        def work(conn: sqlite3.Connection):
            nonlocal inserted, duplicates
            for rec in records or ():
                answer = str(getattr(rec, "answer", "") or "")
                question = str(getattr(rec, "question", "") or "")
                a_norm, q_norm = normalize(answer), normalize(question)
                group = content_hash(answer) or content_hash(question)
                values = {
                    # store.py's corpus columns keep their meaning; the harvest
                    # columns added by _EXTRA are what the crawler actually fills.
                    "persona_id": self.persona_id, "kind": "fatwa",
                    "topic": str(getattr(rec, "topic", "") or "")[:200],
                    "question": question, "answer": answer,
                    "citation": str(getattr(rec, "citation", "") or ""),
                    "issued_at": self._issued(getattr(rec, "issued_at", "")),
                    # group_id is store.py's content identity, dedup_group the job's
                    # name for it. Same value, both readable: authority.py may join on
                    # either without this module owning that decision.
                    "group_id": group, "norm": a_norm, "created_at": ts,
                    "source_url": str(getattr(rec, "url", "") or ""),
                    "sha": str(getattr(rec, "sha", "") or ""),
                    "fatwa_id": str(getattr(rec, "fatwa_id", "") or ""),
                    "title": str(getattr(rec, "title", "") or ""),
                    "question_norm": q_norm, "answer_norm": a_norm,
                    "score": float(getattr(rec, "score", 0.0) or 0.0),
                    "needs_review": 1 if _needs_review(rec, a_norm) else 0,
                    "authority": "", "dedup_group": group,
                }
                # Guard the insert with an explicit lookup, not only the UNIQUE index:
                # the witness index is partial (blank-identity rows slip it), and the
                # job's contract says re-ingesting the same page counts as a duplicate,
                # never a second row.
                if conn.execute("SELECT 1 FROM rulings WHERE sha = ? AND dedup_group = ? LIMIT 1",
                                (values["sha"], group)).fetchone():
                    duplicates += 1
                    continue
                try:
                    cur = conn.execute(
                        "INSERT INTO rulings (" + ", ".join(values) + ") VALUES ("
                        + ", ".join(":" + k for k in values) + ")", values)
                    inserted += cur.rowcount
                except sqlite3.IntegrityError:
                    duplicates += 1
            return None

        self._tx(work)
        return inserted, duplicates

    def _issued(self, raw: Any) -> Any:
        """The page's own date string, or its epoch when the column demands a number."""
        text = str(raw or "").strip()
        if self._issued_is_text:
            return text[:64]
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return time.mktime(time.strptime(text, fmt))
            except ValueError:
                continue
        try:
            return float(text)
        except ValueError:
            return 0.0

    def mark_authority(self, ruling_id: int, tier: str) -> int:
        """Record the answer-authority verdict (DIRECT|DERIVED|NOT_ADDRESSED|'')."""
        cur = self._conn.execute("UPDATE rulings SET authority = ? WHERE id = ?",
                                 (str(tier or "").strip()[:40], int(ruling_id)))
        return cur.rowcount

    def escalate(self, question: str, asked_via: str = "", *, ts: float | None = None,
                 source_url: str = "") -> int:
        """Queue a question the corpus could not answer — mirrored from the JSONL journal.

        Idempotent while a row for the same normalised question is still open, so a
        replayed journal never double-asks the scholar. Once it is answered the question
        may be raised again: a new follow-up is a new piece of harvest.
        """
        q_norm = normalize(question)
        if not q_norm:
            return 0
        at = float(ts if ts is not None else time.time())
        found = self._conn.execute(
            "SELECT id FROM escalations WHERE question_norm = ? AND status != ? "
            "ORDER BY id DESC LIMIT 1", (q_norm, STATUS_ANSWERED)).fetchone()
        if found:
            self._conn.execute("UPDATE escalations SET attempts = attempts + 1, updated_at = ? WHERE id = ?",
                               (at, int(found["id"])))
            return int(found["id"])
        values = {
            # JSONL queue fields (asked_via/ts/answer/source_url) are the ones the
            # escalation worker owns; asked_by/via stay filled so store.py's readers
            # see one story rather than half a row per schema.
            "persona_id": self.persona_id, "question": str(question or ""), "asked_by": "",
            "status": STATUS_OPEN, "via": str(asked_via or "")[:40], "answered_by": "",
            "notified": 0, "attempts": 0, "created_at": at, "sent_at": 0.0, "answered_at": 0.0,
            "updated_at": at, "question_norm": q_norm, "asked_via": str(asked_via or "")[:40],
            "ts": at, "answer": "", "source_url": source_url or "",
        }
        cur = self._conn.execute(
            "INSERT INTO escalations (" + ", ".join(values) + ") VALUES ("
            + ", ".join(":" + k for k in values) + ")", values)
        return int(cur.lastrowid)

    def record_answer(self, escalation_id: int, answer: str, *, source_url: str = "",
                      ruling_id: int | None = None, status: str = STATUS_ANSWERED) -> int:
        """Close the loop: his reply becomes a first-class ruling, the gap is gone."""
        cur = self._conn.execute(
            "UPDATE escalations SET status = ?, answer = ?, answered_by = ?, source_url = ?, ruling_id = ?,"
            " answered_at = ?, updated_at = ? WHERE id = ?",
            (status, str(answer or ""), str(answer or ""), source_url or "", ruling_id,
             time.time(), time.time(), int(escalation_id)))
        return cur.rowcount

    # -- reads ---------------------------------------------------------------
    def search(self, query: str, limit: int = 8) -> list[dict]:
        """FTS over normalized text, so tashkeel and hamza spellings find the same ruling.

        Every query term is also widened through ``stems()`` so the ال/و/ب/ك
        clitics the reader types (or drops) cannot hide a hit.
        """
        match = _match_terms(query)
        if not match:
            return []
        limit = max(1, min(int(limit or 8), 200))
        sql = ("SELECT rulings.*, rank FROM rulings_norm_fts"
               + " JOIN rulings ON rulings.id = rulings_norm_fts.rowid"
               + " WHERE rulings_norm_fts MATCH ?")
        try:
            rows = self._all(sql + " ORDER BY rank LIMIT ?", (match, limit))
        except sqlite3.OperationalError:
            rows = []
        return rows or self._like_fallback(query, limit)

    def _like_fallback(self, query: str, limit: int) -> list[dict]:
        """One leg for long queries fts5 caps out on; correctness over ranking."""
        needle = normalize(query)[:200]
        if not needle:
            return []
        like = "%" + needle[:90] + "%"
        return self._all(
            "SELECT * FROM rulings WHERE answer_norm LIKE ? OR question_norm LIKE ?"
            " ORDER BY score DESC, id DESC LIMIT ?", (like, like, limit))

    def get(self, ruling_id: int) -> dict | None:
        return self._one("SELECT * FROM rulings WHERE id = ?", int(ruling_id))

    def needs_review(self, limit: int = 100) -> list[dict]:
        """Low-score pages, worst first: the list a human works down.

        `needs_review` is recomputed from the record's score, so a page re-extracted
        with better patterns drops out of this queue by itself.
        """
        return self._all(
            "SELECT * FROM rulings WHERE needs_review = 1 AND score < ?"
            " ORDER BY score ASC, id DESC LIMIT ?", (REVIEW_BELOW, max(1, int(limit))))

    def pending_escalations(self, limit: int = 100) -> list[dict]:
        return self._all("SELECT * FROM escalations WHERE status != ? ORDER BY id LIMIT ?",
                         (STATUS_ANSWERED, max(1, int(limit))))

    def stats(self) -> dict:
        r = self._one("SELECT COUNT(*) AS rulings, SUM(needs_review) AS flagged,"
                      " COUNT(DISTINCT dedup_group) AS distinct_rulings FROM rulings") or {}
        desc = self._one("SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name='rulings_norm_fts'") or {}
        e = self._one("SELECT COUNT(*) AS escalations, SUM(status = ?) AS answered FROM escalations",
                      (STATUS_ANSWERED,)) or {}
        return {"rulings": int(r.get("rulings") or 0),
                "distinct_rulings": int(r.get("distinct_rulings") or 0),
                "needs_review": int(r.get("flagged") or 0),
                "escalations": int(e.get("escalations") or 0),
                "escalations_answered": int(e.get("answered") or 0),
                "fts": bool(desc.get("n")),
                "path": self.path}


def _needs_review(rec: Any, a_norm: str) -> bool:
    """Flag from the extractor or the score, and always when there is no answer text."""
    if not a_norm:
        return True
    if bool(getattr(rec, "needs_review", False)):
        return True
    return float(getattr(rec, "score", 0.0) or 0.0) < REVIEW_BELOW


def _match_terms(query: str) -> str:
    """fts5 MATCH string from free text: normalized, stemmed, ANDed quoted terms."""
    parts: list[str] = []
    for tok in tokens(query):
        alts = [t for t in stems(tok) if t]
        if not alts:
            continue
        quoted = " OR ".join('"%s"' % a.replace('"', "") for a in dict.fromkeys(alts))
        parts.append("(" + quoted + ")")
    return " AND ".join(dict.fromkeys(parts))


def open_store(path: str, **kw) -> ScholarStore:
    """One-call constructor for the crawler: it may hand us either a db path or a
    data root, and this decides which without importing config."""
    if os.path.isdir(path):
        path = os.path.join(path, "persona.db")
    return ScholarStore(path, **kw)

