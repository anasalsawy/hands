"""Authoritative, append-only, persona-scoped conversation and tool event timeline.

Every lane reads the same ordered event stream; events retain their producer,
task ownership, causality, and supersession metadata. A checkpoint is an
optimization, never a substitute for retrievable source events.
"""
from __future__ import annotations
import asyncio
import json
import time
from typing import Any

DDL = """
CREATE TABLE IF NOT EXISTS shared_timeline (
 seq INTEGER PRIMARY KEY AUTOINCREMENT,
 persona_id INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
 session_id TEXT NOT NULL DEFAULT '',
 task_id TEXT NOT NULL DEFAULT '',
 producer TEXT NOT NULL,
 kind TEXT NOT NULL,
 payload TEXT NOT NULL,
 parent_seq INTEGER,
 supersedes_seq INTEGER,
 created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS shared_timeline_scope ON shared_timeline(persona_id,seq);
CREATE INDEX IF NOT EXISTS shared_timeline_task ON shared_timeline(persona_id,task_id,seq);
"""

class SharedTimeline:
    def __init__(self, store, bus=None):
        self.store=store
        self.bus=bus
        self._ready=False
        self._lock=asyncio.Lock()

    async def initialize(self):
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            def ddl(conn):
                for statement in DDL.split(';'):
                    if statement.strip():
                        conn.execute(statement)
            await self.store.tx(ddl)
            self._ready=True

    async def append(self, pid:int, *, producer:str, kind:str,
                     payload:dict[str,Any], session_id:str='',task_id:str='',
                     parent_seq:int|None=None,supersedes_seq:int|None=None)->dict:
        await self.initialize()
        if not producer or not kind or not isinstance(payload,dict):
            raise ValueError('producer, kind and dictionary payload required')
        now=time.time()
        def write(conn):
            for pointer in (parent_seq,supersedes_seq):
                if pointer is not None and not conn.execute(
                    'SELECT 1 FROM shared_timeline WHERE seq=? AND persona_id=?',
                    (pointer,pid)).fetchone():
                    raise ValueError('causal event must belong to same persona')
            cursor=conn.execute(
                'INSERT INTO shared_timeline(persona_id,session_id,task_id,producer,kind,'
                'payload,parent_seq,supersedes_seq,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                (pid,session_id,task_id,producer,kind,json.dumps(payload,ensure_ascii=False),
                 parent_seq,supersedes_seq,now))
            return cursor.lastrowid
        seq=await self.store.tx(write)
        event={'seq':seq,'persona_id':pid,'session_id':session_id,
               'task_id':task_id,'producer':producer,'kind':kind,
               'payload':payload,'parent_seq':parent_seq,
               'supersedes_seq':supersedes_seq,'created_at':now}
        if self.bus:
            # The payload rides along: a watcher must be able to show the lane
            # row (tool, topic, state) from this event alone, without a refetch.
            self.bus.publish(pid,{'type':'shared_timeline','seq':seq,'event_kind':kind,
                                  'producer':producer,'task_id':task_id,
                                  'payload':payload,'created_at':now})
        return event

    async def read(self,pid:int,*,after:int=0,limit:int=200,
                   through:int|None=None)->list[dict]:
        await self.initialize()
        limit=min(1000,max(1,int(limit)))
        if through is None:
            rows=await self.store.all(
                'SELECT * FROM shared_timeline WHERE persona_id=? AND seq>? '
                'ORDER BY seq LIMIT ?',pid,after,limit)
        else:
            rows=await self.store.all(
                'SELECT * FROM shared_timeline WHERE persona_id=? AND seq>? AND seq<=? '
                'ORDER BY seq LIMIT ?',pid,after,through,limit)
        return [{**row,'payload':json.loads(row['payload'])} for row in rows]

    async def snapshot(self, pid: int, *, tail: int = 120,
                       exclude_kinds: tuple[str, ...] = ()) -> dict:
        """Read chronological evidence, optionally excluding cached board revisions.

        Exclusion occurs in SQL, before payload JSON is loaded.
        """
        await self.initialize()
        row = await self.store.one(
            'SELECT COALESCE(MAX(seq),0) AS seq FROM shared_timeline WHERE persona_id=?', pid)
        watermark = row["seq"]
        tail = min(1000, max(1, int(tail)))
        where = ""
        args: list = [pid, watermark]
        if exclude_kinds:
            where = " AND kind NOT IN (" + ",".join("?" for _ in exclude_kinds) + ")"
            args.extend(exclude_kinds)
        args.append(tail + 1)
        rows = await self.store.all(
            "SELECT * FROM shared_timeline WHERE persona_id=? AND seq<=?" + where +
            " ORDER BY seq DESC LIMIT ?", *args)
        has_older = len(rows) > tail
        return {
            "through": watermark,
            "events": [{**r, "payload": json.loads(r["payload"])}
                       for r in reversed(rows[:tail])],
            "has_older_events": has_older,
        }
