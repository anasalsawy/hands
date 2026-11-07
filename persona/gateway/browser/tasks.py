"""Persona-scoped background Browser-Use tasks; L chat never awaits their run()."""
from __future__ import annotations
import asyncio
import json
import uuid
from types import SimpleNamespace


def task_schemas():
    specs = [
        ('browser_run_task', 'Run a browser goal with the edited Browser-Use SDK. L observes and converses while R executes. Returns immediately with task status.',
         {'goal':{'type':'string'}}, ['goal']),
        ('browser_task_status', 'Read the current browser task status. A started task is not a completed task.', {}, []),
        ('browser_stop_task', 'Stop the current browser task.', {}, []),
        ('browser_snapshot', 'Read the current browser page.', {}, []),
    ]
    return [{'type':'function','function':{'name':name,'description':description,
            'parameters':{'type':'object','properties':props,'required':required}}}
            for name, description, props, required in specs]


class MindBrowserLanes:
    """Same mind, session and lane contract as PersonaLanes, without self-HTTP.

    SDK owns the tool loop. No recursive Companion tools are offered back to it.
    """
    def __init__(self, mind, pid, session_id):
        self.mind, self.pid, self.session_id = mind, pid, session_id
        self.config = SimpleNamespace(model='sawii-persona')

    async def _stream(self, messages, lane, purpose='conversation', response_format=None):
        system = '\n'.join(str(m.get('content','')) for m in messages if m.get('role') == 'system')
        transcript = [m for m in messages if m.get('role') != 'system']
        text = next((m.get('content') for m in reversed(transcript) if m.get('role') == 'user'), '')
        if not isinstance(text, str):
            text = '\n'.join(p.get('text','') for p in text or [] if p.get('type') == 'text')
        if not text:
            text = 'Continue the browser task using the attached transcript.'
        if response_format:
            system += '\nReturn only JSON matching this response format, without markdown:\n' + json.dumps(response_format)
        async for event in self.mind.respond(self.pid, text, home=False, tools=False,
                host_system=system, host_messages=transcript, lane=lane,
                session_id=self.session_id, purpose=purpose, channel='browser-sdk'):
            if event['type'] == 'error':
                raise RuntimeError('Persona browser inference failed')
            if event['type'] == 'client_tool_calls':
                raise RuntimeError('Browser SDK lane unexpectedly returned host tool calls')
            if event['type'] == 'delta':
                yield event['text']

    async def execute(self, messages, **options):
        pieces = [p async for p in self._stream(messages, 'R', response_format=options.get('response_format'))]
        return {'choices':[{'message':{'role':'assistant','content':''.join(pieces)}}]}

    async def conversation(self, messages, *, purpose='conversation', **options):
        if options.get('tools'):
            raise ValueError('L cannot execute tools')
        async for piece in self._stream(messages, 'L', purpose=purpose):
            yield piece


class BrowserTasks:
    def __init__(self, hub, mind, *, agent_factory=None, lanes_factory=MindBrowserLanes):
        self.hub, self.mind = hub, mind
        self.agent_factory, self.lanes_factory = agent_factory, lanes_factory
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
            if not getattr(one.session, 'native', None):
                raise RuntimeError('Browser tasks require the edited Browser-Use SDK')
            factory = self.agent_factory
            if factory is None:
                from browser_use.persona.agent import PersonaBrowserAgent
                factory = PersonaBrowserAgent
            lanes = self.lanes_factory(self.mind, pid, job['id'])
            agent = factory(task=goal, lanes=lanes, browser_session=one.session.native,
                            max_actions_per_step=1, generate_gif=False)
            job['agent'] = agent
            job['state'] = 'running'
            await publish('running')
            async def refresh():
                while True:
                    # Read-only snapshots; never change the SDK selector map.
                    await one.broadcast.push_snapshot()
                    await asyncio.sleep(.75)
            viewer = asyncio.create_task(refresh())
            history = await agent.run(max_steps=100)
            job['result'] = history.final_result()
            job['state'] = 'finished' if history.is_done() and history.is_successful() is True else 'needs_attention'
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
            if job.get('agent'):
                job['agent'].stop()
            job['state'] = 'cancelled'
            job['task'].cancel()
            await asyncio.gather(job['task'], return_exceptions=True)
        return self.status(pid)

    async def close(self):
        await asyncio.gather(*(self.stop(pid) for pid in list(self.jobs)))
