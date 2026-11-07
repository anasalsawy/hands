"""Durable, concurrent live-command control plane for Sawii execution lanes.

The receipt path is independent of inference and long-running tools. A 202
means the command is durably accepted, NOT that a remote tool was interrupted.
Every transition is persisted and client-visible; a command is only "applied"
after the target worker actually incorporates the instruction.

The control plane owns delivery, not irreversible actions. Never synthesize
a tool result or silently reexecute an unknown action after a crash.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)
SCHEMA = """
CREATE TABLE IF NOT EXISTS live_commands (
  id TEXT PRIMARY KEY,
  persona_id INTEGER NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
  idempotency_key TEXT NOT NULL,
  session_id TEXT NOT NULL DEFAULT '',
  target_type TEXT NOT NULL CHECK(target_type IN ('auto','goal','lane','host')),
  target_id TEXT NOT NULL DEFAULT '',
  action TEXT NOT NULL CHECK(action IN ('steer','cancel')),
  instruction TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'accepted'
      CHECK(state IN ('accepted','delivering','offered','applied','cancel_requested',
                      'awaiting_target','failed','cancelled')),
  version INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  detail TEXT NOT NULL DEFAULT '',
  received_at REAL NOT NULL,
  offered_at REAL NOT NULL DEFAULT 0,
  applied_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL,
  UNIQUE(persona_id,idempotency_key)
);
CREATE INDEX IF NOT EXISTS ix_live_commands_pending
 ON live_commands(persona_id,state,received_at);
CREATE INDEX IF NOT EXISTS ix_live_commands_target
 ON live_commands(persona_id,target_type,target_id,received_at);
"""


class CommandConflict(ValueError):
    pass


class CommandValidation(ValueError):
    pass


def _public(row: dict) -> dict:
    return {k: row[k] for k in (
        'id','idempotency_key','session_id','target_type','target_id',
        'action','instruction','state','version','attempts','detail',
        'received_at','offered_at','applied_at','updated_at')}


class LiveCommands:
    """Persona-scoped durable ingress, idempotency, worker dispatch and recovery."""

    def __init__(self, store, bus, goal_agent, lanes=None):
        """`lanes` is the LaneBook (inference_lanes): the lines of work running
        right now. It replaces the old single background worker, which was one
        queued job addressable only as 'director'."""
        self.store = store
        self.bus = bus
        self.goal_agent = goal_agent
        self.lanes = lanes
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._tasks: set[asyncio.Task] = set()

    async def initialize(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            # Store.tx runs a DB transaction off the event loop, so command
            # ingress never blocks on the SQLite mutex synchronously.
            def ddl(conn):
                # 'director' meant the old single background worker. Old
                # databases carry it in their CHECK constraint and lack 'lane';
                # a CHECK cannot be widened in place, so rebuild once.
                row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                                   "AND name='live_commands'").fetchone()
                if row and row[0] and "'lane'" not in row[0]:
                    conn.execute('DROP TABLE live_commands')
                for statement in SCHEMA.split(';'):
                    if statement.strip():
                        conn.execute(statement)
            await self.store.tx(ddl)
            self._initialized = True

    async def _get(self, pid: int, command_id: str) -> dict | None:
        await self.initialize()
        return await self.store.one(
            'SELECT * FROM live_commands WHERE persona_id=? AND id=?',
            pid, command_id)

    async def get(self, pid: int, command_id: str) -> dict | None:
        row = await self._get(pid, command_id)
        return _public(row) if row else None

    async def list(self, pid: int, *, limit: int = 50) -> list[dict]:
        await self.initialize()
        rows = await self.store.all(
            'SELECT * FROM live_commands WHERE persona_id=? '
            'ORDER BY received_at DESC LIMIT ?', pid, min(max(1,limit),200))
        return [_public(r) for r in rows]

    def _lane_instruction(self, pid: int, line_id: str) -> str:
        """What this line was told to do, for the address router's prompt."""
        if self.lanes is None:
            return ''
        for line in self.lanes.active(pid):
            if line.id == line_id:
                return (line.instruction or line.topic)[:300]
        return ''

    async def active_lanes(self, pid: int) -> list[dict]:
        """Actual active lanes, never imaginary synthetic R workers."""
        active = []
        for task_id in self.goal_agent._active.get(pid, set()):
            job = self.goal_agent.runner._running.get(task_id)
            if job is not None and not job.done():
                active.append({'target_type':'goal','target_id':task_id,'state':'running'})
        # The lines of work from inference_lanes: each is one line the persona
        # is running at this moment, and mid-work guidance lands on it.
        if self.lanes is not None:
            for line in self.lanes.active(pid):
                active.append({'target_type':'lane','target_id':line.id,
                               'state':line.status})
        return active

    async def submit(self, pid: int, *, instruction: str, action: str = 'steer',
                     target_type: str = 'auto', target_id: str = '',
                     session_id: str = '', idempotency_key: str = '') -> dict:
        await self.initialize()
        if action not in ('steer','cancel'):
            raise CommandValidation('action must be steer or cancel')
        if target_type not in ('auto','goal','lane','host'):
            raise CommandValidation('invalid target_type')
        if action=='steer' and not (instruction or '').strip():
            raise CommandValidation('steer instruction cannot be empty')
        if len(instruction or '') > 6000:
            raise CommandValidation('instruction exceeds 6000 characters')
        if len(session_id)>128 or len(target_id)>180:
            raise CommandValidation('session/target id too long')
        if target_type == 'host' and not target_id:
            raise CommandValidation('host target_id must name an actual host child')
        key = idempotency_key or uuid.uuid4().hex
        if not isinstance(key,str) or len(key)>128 or not key.strip():
            raise CommandValidation('invalid idempotency_key')
        instruction=(instruction or '').strip()
        now=time.time()
        command_id='lc_'+uuid.uuid4().hex
        def commit(conn):
            existing = conn.execute(
                'SELECT * FROM live_commands WHERE persona_id=? AND idempotency_key=?',
                (pid,key)).fetchone()
            if existing:
                row=dict(existing)
                if any((row[k] != expected) for k, expected in (
                    ('instruction',instruction),('action',action),
                    ('target_type',target_type),('target_id',target_id),
                    ('session_id',session_id))):
                    raise CommandConflict('idempotency key was used with different command contents')
                return row, False
            conn.execute(
                'INSERT INTO live_commands'
                '(id,persona_id,idempotency_key,session_id,target_type,target_id,'
                'action,instruction,received_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (command_id,pid,key,session_id,target_type,target_id,
                 action,instruction,now,now))
            return dict(conn.execute(
                'SELECT * FROM live_commands WHERE id=?',(command_id,)).fetchone()),True
        row,is_new=await self.store.tx(commit)
        if is_new:
            if getattr(self,'timeline',None):
                await self.timeline.append(pid,producer='L',kind='command_accepted',
                    payload={'command_id':row['id'],'action':action,
                             'instruction':instruction,'target_type':target_type,
                             'target_id':target_id},session_id=session_id,
                    task_id=target_id if target_type=='goal' else '')
            self.bus.publish(pid, {'type':'live_command','id':row['id'],'state':'accepted',
                                   'received_at':now})
        if row['state'] in ('accepted','awaiting_target'):
            self._spawn(pid,row['id'])
        return _public(row)

    def _spawn(self, pid: int, command_id: str):
        task=asyncio.create_task(self._dispatch(pid,command_id),
                                 name='live-command-'+command_id[-10:])
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transition(self, pid: int, command_id: str, state: str, *,
                          detail: str = '', version: int | None = None):
        await self.initialize()
        now=time.time()
        def txn(conn):
            row=conn.execute(
                'SELECT * FROM live_commands WHERE id=? AND persona_id=?',
                (command_id,pid)).fetchone()
            if row is None:
                return None
            # Never demote a command already applied by a racing worker.
            if row['state'] in ('applied','cancelled') and state not in ('applied','cancelled'):
                return dict(row)
            version_val=row['version'] if version is None else version
            conn.execute(
                'UPDATE live_commands SET state=?,detail=?,version=?,'
                'attempts=attempts+?,offered_at=CASE WHEN ? IN (\'offered\',\'cancel_requested\') '
                'THEN ? ELSE offered_at END,'
                'applied_at=CASE WHEN ?=\'applied\' THEN ? ELSE applied_at END,'
                'updated_at=? WHERE id=?',
                (state,detail[:600],version_val,1 if state=='delivering' else 0,
                 state,now,state,now,now,command_id))
            return dict(conn.execute(
                'SELECT * FROM live_commands WHERE id=?',(command_id,)).fetchone())
        row=await self.store.tx(txn)
        if row:
            if getattr(self,'timeline',None):
                await self.timeline.append(pid,producer=row['target_type'],
                    kind='command_'+row['state'],
                    payload={'command_id':command_id,'state':row['state'],
                             'detail':row['detail'],'target_id':row['target_id']},
                    session_id=row['session_id'],
                    task_id=row['target_id'] if row['target_type']=='goal' else '')
            self.bus.publish(pid, {'type':'live_command','id':command_id,
                                   'target_type':row['target_type'],
                                   'target_id':row['target_id'],
                                   'state':row['state'],'detail':row['detail'],
                                   'updated_at':row['updated_at']})
        return row

    def mark_applied(self, pid: int, command_id: str):
        """Called only when a worker folds the command into a real model turn.

        Fire-and-track, never block the tool dispatcher on DB writes.
        """
        if not command_id:
            return
        t=asyncio.create_task(self._transition(pid,command_id,'applied',
                             detail='injected into execution context'))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _classify_targets(self, pid: int, instruction: str,
                                current: list[dict]) -> list[dict]:
        """L chooses affected live R addresses while conversation continues.

        This call never runs a tool. Only ids returned by active_lanes are
        valid; errors leave the durable command unresolved, never lost.
        """
        candidate=[]
        for lane in current:
            title=''
            if lane['target_type']=='goal':
                try:
                    task=await self.goal_agent.ledger.aget(pid,lane['target_id'])
                    title=str(getattr(task,'goal','') or '')[:300]
                except Exception:
                    pass
            elif lane['target_type']=='lane':
                title=self._lane_instruction(pid, lane['target_id'])
            candidate.append({'type':lane['target_type'],'id':lane['target_id'],
                              'current_task':title})
        model=getattr(getattr(self.goal_agent,'m',None),'a',None)
        if model is None:
            return []
        try:
            reply=await model.chat([
                {'role':'system','content':(
                    'You are the live command address router. Given the NEW user '
                    'message and live execution agents, choose exactly those '
                    'existing tasks that the message corrects or changes. '
                    'Never create or execute a task. Unrelated conversation '
                    'must have empty targets. If the message affects two '
                    'active tasks select both. Reply only JSON '
                    '{"targets":["exact-id",...]} with IDs from the list.'
                )},
                {'role':'user','content':json.dumps({
                    'message':instruction,'active_agents':candidate
                },ensure_ascii=False)}
            ],max_tokens=250,temperature=0.0,json_mode=True)
            obj=json.loads(reply.text)
            ids=obj.get('targets',[]) if isinstance(obj,dict) else []
            allow={x['target_id']:x for x in current}
            if not isinstance(ids,list):
                return []
            return [allow[i] for i in dict.fromkeys(ids)
                    if isinstance(i,str) and i in allow]
        except Exception:
            log.exception('Live L command target selection failed')
            return []

    async def _dispatch(self, pid: int, command_id: str):
        await self.initialize()
        # Atomic claim prevents a retrying HTTP client or concurrent workers
        # from delivering the same guidance twice in one process.
        now=time.time()
        def claim(conn):
            cur=conn.execute(
                "UPDATE live_commands SET state='delivering', attempts=attempts+1, "
                "updated_at=? WHERE id=? AND persona_id=? AND "
                "state IN ('accepted','awaiting_target')",
                (now,command_id,pid))
            if cur.rowcount != 1:
                return None
            return dict(conn.execute(
                'SELECT * FROM live_commands WHERE id=?',(command_id,)).fetchone())
        row=await self.store.tx(claim)
        if row is None:
            return
        try:
            current=await self.active_lanes(pid)
            typ=row['target_type']
            ident=row['target_id']
            if typ=='auto':
                if len(current)>=1:
                    selected=await self._classify_targets(pid,row['instruction'],current)
                    if not selected:
                        await self._transition(pid,command_id,'awaiting_target',
                            detail='No active R task selected by L; message remains in shared timeline')
                        return
                    children=[]
                    for target in selected:
                        child=await self.submit(pid,instruction=row['instruction'],
                            action=row['action'],target_type=target['target_type'],
                            target_id=target['target_id'],session_id=row['session_id'],
                            idempotency_key='route:'+hashlib.sha256(
                                (command_id+':'+target['target_type']+':'+
                                 target['target_id']).encode()).hexdigest())
                        children.append(child['id'])
                    await self._transition(pid,command_id,'offered',
                        detail='addressed to '+str(len(children))+
                               ' execution lanes; child command ids: '+','.join(children))
                    return
                else:
                    await self._transition(pid,command_id,'awaiting_target',
                        detail='No active R lane; instruction retained for L/explicit target')
                    return
            elif typ=='host':
                # The host must expose an authenticated transport supporting
                # child steer/stop. Sending a *model prompt* to Hermes is NOT a
                # reliable low-latency steering control plane.
                await self._transition(pid,command_id,'awaiting_target',
                    detail='Host child is not directly controllable by this Persona server; '
                           'a host-native authenticated steer/stop bridge is required')
                return
            elif not any(x['target_type']==typ and
                         (not ident or x['target_id']==ident) for x in current):
                await self._transition(pid,command_id,'awaiting_target',
                    detail='Target R lane not active (awaiting resume or registration)')
                return
            if typ=='goal':
                if row['action']=='steer':
                    result=self.goal_agent.note_user_input(
                        pid,row['instruction'],task_id=ident,command_id=command_id)
                    if not result:
                        await self._transition(pid,command_id,'awaiting_target',
                                               detail='Goal no longer accepting input')
                        return
                    # New instructions invalidate not-yet-started proposals,
                    # without cancelling external tools already in flight.
                    await self.goal_agent.runner.conflict_coordinator(pid).supersede(ident)
                    await self._transition(pid,command_id,'offered',
                                           detail='delivered to goal mailbox; pending proposals superseded',
                                           version=result)
                    return
                result=await self.goal_agent.cancel_live(pid,ident)
                await self._transition(pid,command_id,
                    'cancel_requested' if result else 'awaiting_target',
                    detail='stop requested; in-flight external tool may still run' if result
                           else 'goal not found')
                return
            if typ=='lane':
                # A line of work from inference_lanes. Steering is a note it
                # reads before its next step; cancelling marks it stopped so it
                # takes no further step (a tool already handed to the host is
                # the host's to run or drop).
                if row['action']=='cancel':
                    stopped = await self.lanes.stop(pid, ident) if self.lanes else False
                    await self._transition(pid,command_id,
                        'applied' if stopped else 'awaiting_target',
                        detail='line stopped' if stopped else 'line not running')
                    return
                noted = await self.lanes.note(pid, ident, row['instruction']) if self.lanes else False
                await self._transition(pid,command_id,
                    'applied' if noted else 'awaiting_target',
                    detail='note delivered to the line' if noted else 'line not running')
                return
        except Exception as exc:
            log.exception('live command %s delivery failed',command_id)
            await self._transition(pid,command_id,'failed',
                                   detail=type(exc).__name__)

    async def retry_pending(self, pid: int | None = None):
        """Recovery-safe replay of *unapplied guidance*, not external tools."""
        await self.initialize()
        # An offered command is not an applied command. Recover only
        # incomplete guidance, never re-execute side-effectful host tools.
        if pid is None:
            rows=await self.store.all(
                "SELECT persona_id,id FROM live_commands "
                "WHERE state IN ('accepted','awaiting_target') ORDER BY received_at LIMIT 500")
        else:
            rows=await self.store.all(
                "SELECT persona_id,id FROM live_commands "
                "WHERE persona_id=? AND state IN ('accepted','awaiting_target') "
                "ORDER BY received_at LIMIT 500",pid)
        for row in rows:
            self._spawn(row['persona_id'],row['id'])
        return len(rows)

    async def recover_on_startup(self):
        """Reset incomplete *guidance* after a process death.

        This is a startup-only operation, before in-process delivery begins.
        Outstanding external actions are NEVER automatically re-run.
        """
        await self.initialize()
        await self.store.run(
            "UPDATE live_commands SET state='accepted', updated_at=? "
            "WHERE state IN ('delivering','offered')",
            time.time())
        return await self.retry_pending()

    async def shutdown(self):
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks),return_exceptions=True)

