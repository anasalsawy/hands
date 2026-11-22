"""Persona-scoped background browser tasks, run by the Agent TARS loop on the persona's own
browser engine with Persona as the mind. L chat never awaits their run()."""
from __future__ import annotations
import asyncio
import json
import uuid


def task_schemas():
    specs = [
        ('browser_run_task', 'Run a browser goal with Agent TARS (the execution engine) while Persona thinks for it. L observes and converses while R executes. Returns immediately with task status.',
         {'goal':{'type':'string'}}, ['goal']),
        ('browser_task_status', 'Read the current browser task status. A started task is not a completed task.', {}, []),
        ('browser_stop_task', 'Stop the current browser task.', {}, []),
        ('browser_snapshot', 'Read the current browser page.', {}, []),
    ]
    return [{'type':'function','function':{'name':name,'description':description,
            'parameters':{'type':'object','properties':props,'required':required}}}
            for name, description, props, required in specs]


class BrowserTasks:
    def __init__(self, hub, mind, *, infer_factory=None):
        self.hub, self.mind = hub, mind
        self.infer_factory = infer_factory
        self.jobs = {}

    def status(self, pid):
        job = self.jobs.get(pid)
        return {k:v for k,v in job.items() if k not in ('task','agent')} if job else {'state':'idle'}

    def active(self, pid):
        job = self.jobs.get(pid)
        return bool(job and not job['task'].done())

    def start(self, pid, goal):
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 20000:
            raise ValueError('Browser goal must contain 1..20000 characters')
        if self.active(pid):
            raise ValueError('A browser task already owns this persona browser')
        job = {'id':uuid.uuid4().hex, 'state':'starting', 'result':None, 'error':None}
        job['task'] = asyncio.create_task(self._run(pid, goal, job), name=f'browser-{pid}-{job["id"]}')
        self.jobs[pid] = job
        return self.status(pid)

    async def _run(self, pid, goal, job):
        viewer = None
        async def publish(state):
            if self.mind:
                self.mind.bus.publish(pid, {'type':'browser_task', **self.status(pid)})
            if pid in self.hub._ones:
                await self.hub._ones[pid].broadcast.push_action('task', {'name':'browser_task','result':state})
        try:
            one = await self.hub._get(pid)
            engine = getattr(one.session, 'engine', None)
            if engine is None:
                raise RuntimeError('Browser tasks require the Agent TARS engine')
            from ...tars_loop import run_tars_loop
            tools = {'schemas': []}
            lobe = getattr(self.mind, 'a', None)

            async def infer(messages):
                if self.infer_factory is not None:
                    return await self.infer_factory(messages, tools['schemas'])
                reply = await lobe.chat(messages, tools=tools['schemas'] or None, max_tokens=4096, temperature=0.2)
                calls = [{'id': tc.id, 'name': tc.name, 'arguments': json.loads(tc.arguments or '{}')}
                         for tc in getattr(reply, 'tool_calls', None) or []]
                return {'text': getattr(reply, 'text', '') or '', 'tool_calls': calls,
                        'finish_reason': getattr(reply, 'finish_reason', 'stop')}

            async def observe():
                return {'text': ''}

            async def execute(name, args):
                raise RuntimeError('no Persona-side tools in a browser task: ' + name)

            job['state'] = 'running'
            await publish('running')

            async def refresh():
                while True:
                    # Read-only frames: TARS's own screenshot tool, never a selector-map change.
                    await one.broadcast.push_snapshot()
                    await asyncio.sleep(.75)
            viewer = asyncio.create_task(refresh())
            job['result'] = await run_tars_loop(
                goal=goal, instructions='You are Sawii\'s R line driving her own browser with Agent TARS tools. '
                'Do the task with real tool calls, then say in one short line what you actually did.',
                tools=[], infer=infer, execute=execute, observe=observe, engine=engine,
                on_tools=lambda specs: tools.__setitem__('schemas', specs),
                on_receipt=getattr(self, 'on_receipt', None),
                max_steps=100, timeout=180.0)
            job['state'] = 'finished'
        except asyncio.CancelledError:
            job['state'] = 'cancelled'
            raise
        except Exception as exc:
            job['state'], job['error'] = 'failed', f'{type(exc).__name__}: {exc}'[:500]
        finally:
            if viewer:
                viewer.cancel()
                await asyncio.gather(viewer, return_exceptions=True)
            # run() cleans up the edited SDK observer/prefetch even on cancellation.
            await publish(job['state'])
            loop = getattr(self.mind, '_active_loops', {}).get(pid) if self.mind else None
            if loop is not None:
                loop.hear_inner('[Browser task status, not independently verified] ' + json.dumps(self.status(pid)))

    async def stop(self, pid):
        job = self.jobs.get(pid)
        if job and not job['task'].done():
            job['state'] = 'cancelled'
            job['task'].cancel()
            await asyncio.gather(job['task'], return_exceptions=True)
        return self.status(pid)

    async def close(self):
        await asyncio.gather(*(self.stop(pid) for pid in list(self.jobs)))
