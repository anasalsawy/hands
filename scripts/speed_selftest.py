"""Offline safety/latency regressions: python scripts/speed_selftest.py."""
from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona.hands import HandsHub
from persona.link import PossibilitySpeculator, SpeculativeEngine
from persona.perception import screen_frame


async def result(value):
    return value


class SpeculationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_and_sync_callbacks(self):
        engine = SpeculativeEngine()
        step = await engine.step(0.5, lambda: result('ok'))
        self.assertEqual(step['result'], 'ok')
        step = await engine.step(0.5, lambda: result('ok'), lambda ctx: {'ok': {'name': 'read'}})
        self.assertEqual(step['next_action'], {'name': 'read'})

    async def test_action_and_precompute_overlap(self):
        action_started, compute_started = asyncio.Event(), asyncio.Event()

        async def action():
            action_started.set()
            await compute_started.wait()
            return 'ok'

        async def compute(ctx):
            compute_started.set()
            await action_started.wait()
            return {'ok': {'name': 'read'}}

        got = await asyncio.wait_for(SpeculativeEngine().step(.5, action, compute), 1)
        self.assertEqual(got['next_action'], {'name': 'read'})

    async def test_slow_precompute_does_not_block_and_is_cancelled(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def compute(ctx):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def action():
            await started.wait()
            return 'ok'

        spec = PossibilitySpeculator(precompute_grace=0)
        got = await asyncio.wait_for(spec.decide_and_act(action, compute), 1)
        self.assertEqual(got, ('ok', None))
        self.assertTrue(cancelled.is_set())

    async def test_failed_or_malformed_precompute_preserves_actual_result(self):
        async def failed(ctx):
            raise RuntimeError('prediction failed')

        for compute in (failed, lambda ctx: []):
            got = await SpeculativeEngine().step(.5, lambda: result('ok'), compute)
            self.assertEqual(got['result'], 'ok')
            self.assertIsNone(got['next_action'])

    async def test_failed_action_cancels_precompute(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def compute(ctx):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def action():
            await started.wait()
            raise RuntimeError('actual failure')

        with self.assertRaisesRegex(RuntimeError, 'actual failure'):
            await SpeculativeEngine().step(.5, action, compute)
        self.assertTrue(cancelled.is_set())

    async def test_parent_cancellation_stops_both_tasks(self):
        action_started, compute_started = asyncio.Event(), asyncio.Event()
        ended = []

        async def blocked(label, started):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.append(label)

        task = asyncio.create_task(SpeculativeEngine().step(.5,
            lambda: blocked('action', action_started),
            lambda ctx: blocked('compute', compute_started)))
        await action_started.wait()
        await compute_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertCountEqual(ended, ['action', 'compute'])

    async def test_contradicted_or_unguarded_batch_never_selected(self):
        for prediction in ({'name': 'click', '_expected_outcome': 'ok'}, {'name': 'click'}):
            got = await SpeculativeEngine().step(.99, lambda: result('failed'),
                batch_actions=[prediction])
            self.assertFalse(got['batch_fired'])
            self.assertIsNone(got['next_action'])

    async def test_misprediction_dispatches_observed_alternative(self):
        got = await SpeculativeEngine().step(.99, lambda: result('failed'),
            lambda ctx: {'failed': {'name': 'recover'}},
            batch_actions=[{'name': 'click', '_expected_outcome': 'ok'}])
        self.assertFalse(got['batch_fired'])
        self.assertEqual(got['next_action'], {'name': 'recover'})
        self.assertEqual(got['batch_remaining'], [])

    async def test_matching_batch_handles_structured_and_json_status(self):
        for value in ({'status': 'ok'}, '{"status":"ok"}', 'ok'):
            got = await SpeculativeEngine().step(.99, lambda: result(value),
                batch_actions=[{'name': 'read', '_expected_outcome': 'ok'}])
            self.assertTrue(got['batch_fired'])
            self.assertEqual(got['next_action']['name'], 'read')

    async def test_status_dispatch_and_exact_result_precedence(self):
        spec = PossibilitySpeculator()
        self.assertEqual(spec.dispatch('{"status":"active"}', {'active': {'name': 'monitor'}}),
                         {'name': 'monitor'})
        raw = '{"status":"active"}'
        self.assertEqual(spec.dispatch(raw, {raw: {'name': 'exact'}, 'active': {}}), {'name': 'exact'})

    async def test_invalid_probabilities_rejected_before_execution(self):
        for confidence in (-1, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                await SpeculativeEngine().step(confidence, lambda: self.fail('action ran'))

    async def test_last_step_skips_precompute_and_batch(self):
        got = await SpeculativeEngine().step(.99, lambda: result('ok'),
            lambda ctx: self.fail('precompute ran'),
            batch_actions=[{'name': 'click', '_expected_outcome': 'ok'}], is_last_step=True)
        self.assertEqual(got['result'], 'ok')
        self.assertFalse(got['batch_fired'])

    async def test_engine_uses_its_configured_possibility_budget(self):
        engine = SpeculativeEngine(dial=10)
        got = await engine.step(.1, lambda: result('none'), lambda ctx: {str(i): {} for i in range(20)})
        self.assertEqual(len(got['possibilities']), engine.max_possibilities)


class PerceptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_sources_capture_concurrently_in_stable_order(self):
        started = set()
        all_started = asyncio.Event()

        async def capture(source):
            started.add(source)
            if len(started) == 3:
                all_started.set()
            await all_started.wait()

        class Desktop:
            driver = object()

            async def call(self, name, args):
                await capture('desktop')
                return json.dumps({'image_b64': 'aW1hZ2U='})

        class Hands:
            def names(self, pid):
                return {'computer_ocu_get_app_state', 'cloud_ocu_get_app_state'}

            async def observe(self, pid, source):
                await capture(source)
                return {'source': source, 'text': 'same screen'}

        mind = types.SimpleNamespace(desktop=Desktop(), hands=Hands())
        got = await screen_frame(mind, 7, timeout=1)
        self.assertEqual(got['errors'], [])
        self.assertEqual([s['source'] for s in got['screens']], ['desktop', 'computer', 'cloud'])

    async def test_slow_surface_times_out_without_losing_other_surfaces(self):
        cancelled = asyncio.Event()

        class Hands:
            def names(self, pid):
                return {'computer_ocu_get_app_state', 'cloud_ocu_get_app_state'}

            async def observe(self, pid, source):
                if source == 'computer':
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                return {'source': source, 'text': 'available'}

        got = await screen_frame(types.SimpleNamespace(hands=Hands()), 7, timeout=.01)
        self.assertEqual(got['errors'], [{'source': 'computer', 'error': 'TimeoutError'}])
        self.assertEqual(got['screens'][0]['source'], 'cloud')
        self.assertTrue(cancelled.is_set())

    async def test_image_only_changes_invalidate_digest(self):
        class Hands:
            image = 'first'

            def names(self, pid):
                return {'computer_ocu_get_app_state'}

            async def observe(self, pid, source):
                return {'source': source, 'text': 'same', 'image_url': self.image}

        hands = Hands()
        mind = types.SimpleNamespace(hands=hands)
        first = await screen_frame(mind, 7)
        self.assertEqual(first['state_digest'], (await screen_frame(mind, 7))['state_digest'])
        hands.image = 'second'
        self.assertNotEqual(first['state_digest'], (await screen_frame(mind, 7))['state_digest'])

    async def test_empty_mind_is_valid(self):
        got = await screen_frame(types.SimpleNamespace(), 7)
        self.assertEqual(got['screens'], [])
        self.assertEqual(got['errors'], [])

    async def test_l_vision_collects_every_r_lane_without_cross_persona_leaks(self):
        hub = HandsHub()

        async def send(frame):
            self.fail('cached observations must not mutate the screen')

        schemas = [{'type':'function','function':{'name':'ocu_get_app_state','parameters':{}}}]
        for pid, lane in [(7, 'R2'), (8, 'private'), (7, 'R1')]:
            runner = hub.connect(pid, send, schemas, lane=lane)
            runner.observation = {'source':'computer', 'text':f'only-{lane}'}
        frame = await screen_frame(types.SimpleNamespace(hands=hub), 7)
        self.assertEqual([s['lane'] for s in frame['screens']], ['R1', 'R2'])
        self.assertEqual([s['text'] for s in frame['screens']], ['only-R1', 'only-R2'])
        self.assertEqual(frame['errors'], [])

    async def test_observation_stays_on_selected_lane_even_if_another_hand_is_idle(self):
        hub = HandsHub()
        schemas = [{'type': 'function', 'function': {'name': 'ocu_get_app_state'}}]
        sent = []
        chosen = None

        async def selected_send(frame):
            sent.append(('selected', frame))
            hub.result(chosen, {'id': frame['id'], 'output': 'ok'})

        async def wrong_send(frame):
            self.fail('observation crossed into the idle lane')

        chosen = hub.connect(7, selected_send, schemas, lane='R1')
        hub.connect(7, wrong_send, schemas, lane='R2')
        chosen.pending['existing-action'] = object()
        chosen.last_app = 'Editor'
        await hub.observe(7, 'computer')
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1]['arguments'], {'app': 'Editor'})
        self.assertIsNone(await hub.observe(7, 'computer', lane='missing'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
