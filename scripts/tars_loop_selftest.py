"""Exercise the real vendored Omni-TARS loop through Persona's private bridge.

    npm ci --prefix engine
    python scripts/tars_loop_selftest.py

Only inference and OS tools are scripted; the TypeScript agent/composer,
native call parser, history and lifecycle are the production implementation.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona.court import HandsDeck
from persona.llm import ChatResult, ToolCall
from persona.tars_loop import LanePlan, PLAN_NAME, PLAN_TOOL, run_tars_loop

TOOL = {"type": "function", "function": {
    "name": "computer_click", "description": "Click this lane's surface",
    "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
}}


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_loop(self, infer, execute=None, observe=None, **kwargs):
        async def call(name, args):
            return "receipt"

        async def frame():
            return None

        return await run_tars_loop(goal="test goal", instructions="Persona owns this lane",
                                   tools=[TOOL], infer=infer, execute=execute or call,
                                   observe=observe or frame, **kwargs)

    async def test_real_loop_preserves_vision_receipts_and_ordered_batch(self):
        calls, seen, observations = [], [], []
        running = 0

        async def frame():
            observations.append(len(calls))
            return {"text": f"frame-{len(calls)}", "image_url": "data:image/png;base64,dGVzdA=="}

        async def infer(messages):
            seen.append(messages)
            if len(seen) == 1:
                return {"text": "", "tool_calls": [
                    {"id": f"c{i}", "name": "computer_click", "arguments": {"x": i}}
                    for i in range(2)]}
            return {"text": "clicked twice", "tool_calls": []}

        async def call(name, args):
            nonlocal running
            self.assertEqual(running, 0, "same-surface mutations must not overlap")
            running += 1
            await asyncio.sleep(0)
            calls.append(args["x"])
            running -= 1
            return f"receipt-{args['x']}"

        result = await self.run_loop(infer, call, frame)
        self.assertEqual(result, "clicked twice")
        self.assertEqual(calls, [0, 1])
        self.assertEqual(observations, [0, 2])
        self.assertIn("frame-0", str(seen[0][-1]))
        self.assertIn("data:image/png;base64,dGVzdA==", str(seen[0][-1]))
        self.assertIn("frame-2", str(seen[1][-1]))
        self.assertEqual([m["content"] for m in seen[1] if m["role"] == "tool"],
                         ["receipt-0", "receipt-1"])
        self.assertEqual(sum("frame-0" in str(m) for m in seen[1]), 0,
                         "old screenshots must not accumulate in model history")

    async def test_provider_credentials_and_node_injection_not_inherited(self):
        create = asyncio.create_subprocess_exec
        children = []

        async def spawn(*args, **kwargs):
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            self.assertNotIn("NODE_OPTIONS", kwargs["env"])
            proc = await create(*args, **kwargs)
            children.append(proc)
            return proc

        async def infer(messages):
            return {"text": "done"}

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-not-a-key", "NODE_OPTIONS": "--bad-option"}), \
                patch("asyncio.create_subprocess_exec", side_effect=spawn):
            self.assertEqual(await self.run_loop(infer), "done")
        self.assertIsNotNone(children[0].returncode)

    async def test_cancellation_reaps_child_and_pending_inference(self):
        create = asyncio.create_subprocess_exec
        children = []
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def spawn(*args, **kwargs):
            child = await create(*args, **kwargs)
            children.append(child)
            return child

        async def infer(messages):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        with patch("asyncio.create_subprocess_exec", side_effect=spawn):
            task = asyncio.create_task(self.run_loop(infer))
            await asyncio.wait_for(entered.wait(), 10)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stopped.is_set())
        self.assertIsNotNone(children[0].returncode)

    async def test_timeout_cancels_pending_inference(self):
        stopped = asyncio.Event()

        async def infer(messages):
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        with self.assertRaises(asyncio.TimeoutError):
            await self.run_loop(infer, timeout=0.05)
        self.assertTrue(stopped.is_set())

    async def test_step_limit_never_reports_done(self):
        async def infer(messages):
            return {"text": "", "tool_calls": [
                {"id": "again", "name": "computer_click", "arguments": {}}]}

        with self.assertRaisesRegex(RuntimeError, "step limit"):
            await self.run_loop(infer, max_steps=1)

    async def test_empty_answer_never_reports_done(self):
        async def infer(messages):
            return {"text": ""}

        with self.assertRaisesRegex(RuntimeError, "completed answer"):
            await self.run_loop(infer)

    async def test_truncated_response_never_reports_done(self):
        async def infer(messages):
            return {"text": "I have", "finish_reason": "length"}

        with self.assertRaisesRegex(RuntimeError, "truncated"):
            await self.run_loop(infer)

    async def test_missing_node_has_actionable_setup_error(self):
        async def infer(messages):
            self.fail("inference must not run without the loop")

        with patch.dict(os.environ, {"SAWII_NODE": "/not-installed/node"}):
            with self.assertRaisesRegex(RuntimeError, "npm ci --prefix engine"):
                await self.run_loop(infer)

    async def test_inference_failure_propagates(self):
        async def infer(messages):
            raise ValueError("test lobe unavailable")

        with self.assertRaisesRegex(ValueError, "lobe unavailable"):
            await self.run_loop(infer)

    async def test_prepared_batch_is_advice_not_a_speculative_click(self):
        seen, calls = [], []

        async def infer(messages):
            seen.append(messages)
            if len(seen) == 1:
                return {"text": "", "tool_calls": [
                    {"id": "plan", "name": PLAN_NAME, "arguments": {
                        "confidence": 0.99,
                        "possibilities": {"error": {"name": "computer_click", "arguments": {"x": 2}}},
                        "batch": [{"name": "computer_click", "arguments": {"x": 99},
                                   "_expected_outcome": "ok"}]}},
                    {"id": "action", "name": "computer_click", "arguments": {"x": 1}}]}
            self.assertIn('"x": 2', str(messages[-1]))
            self.assertNotIn('"x": 99', str(messages[-1]))
            self.assertIn("NOT executed", str(messages[-1]))
            return {"text": "checked the failure"}

        async def execute(name, args):
            calls.append(args["x"])
            return '{"status":"error"}'

        result = await run_tars_loop(goal="test", instructions="test", tools=[TOOL, PLAN_TOOL],
                                     infer=infer, execute=execute, observe=lambda: asyncio.sleep(0))
        self.assertEqual(result, "checked the failure")
        self.assertEqual(calls, [1], "neither the wrong batch nor the advisory may auto-execute")


class PlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_match_is_one_use_and_always_requires_r_confirmation(self):
        plan = LanePlan([TOOL])
        proposed = {"name": "computer_click", "arguments": {"x": 2}, "_expected_outcome": "ok"}
        plan.prepare({"confidence": 0.9, "possibilities": {}, "batch": [proposed]})
        result = await plan.execute(lambda: asyncio.sleep(0, result='{"status":"ok"}'))
        self.assertEqual(result, '{"status":"ok"}')
        self.assertEqual(plan.advice, proposed)
        self.assertIn("NOT executed", plan.with_advice([])[0]["content"])
        self.assertEqual(plan.with_advice([]), [])
        await plan.execute(lambda: asyncio.sleep(0, result='{"status":"ok"}'))
        self.assertIsNone(plan.advice, "an old batch must not be replayed on the next call")

    async def test_invalid_confidence_actions_and_unguarded_batches_rejected(self):
        for arguments in [
            {"confidence": value, "possibilities": {}} for value in [-1, 2, float("nan"), True]
        ] + [
            {"confidence": 0.9, "possibilities": {"ok": {"name": "not-advertised", "arguments": {}}}},
            {"confidence": 0.9, "possibilities": {}, "batch": [{"name": "computer_click", "arguments": {}}]},
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                LanePlan([TOOL]).prepare(arguments)


class CourtTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, replies, outputs=("receipt",), *, lane="r-one"):
        seen, calls, frames, records = [], [], [], []
        scripted = iter(replies)
        results = iter(outputs)

        async def chat(messages, **kwargs):
            seen.append(messages)
            self.assertEqual(kwargs["tools"], [TOOL, PLAN_TOOL])
            return next(scripted)

        async def call(pid, name, args, **kwargs):
            self.assertEqual(kwargs["lane"], lane)
            calls.append((pid, name, args, kwargs["lane"]))
            return next(results)

        async def observe(pid, place, **kwargs):
            self.assertEqual(kwargs["lane"], lane)
            frames.append(kwargs["lane"])
            return {"text": f"only-{lane}"}

        async def append(pid, **kwargs):
            self.assertEqual(pid, 7)
            self.assertEqual(kwargs['task_id'], lane)
            records.append(kwargs)

        hub = SimpleNamespace(_key=lambda pid, place, lane: (pid, place, lane),
                              runners={(7, "computer", lane): SimpleNamespace(tools=[TOOL])},
                              call=call, observe=observe)
        mind = SimpleNamespace(a=SimpleNamespace(chat=chat), hands=hub, links=None, bus=None,
                               shared_timeline=SimpleNamespace(append=append))
        deck = HandsDeck(mind)
        court = deck.create(7, title="test")
        court.lane, court.status = lane, "running"
        await deck._drive(7, court.id, lane, "click")
        self.assertEqual([record['kind'] for record in records],
                         ['tool_call', 'tool_result'] * len(calls),
                         "L and R must share the real call/result receipts")
        return court, calls, seen, frames

    async def test_court_executes_on_explicit_lane_and_returns_real_receipt(self):
        court, calls, seen, frames = await self.drive([
            ChatResult("", [ToolCall("c1", "computer_click", '{"x":1}')]), ChatResult("clicked")])
        self.assertEqual(court.status, "done")
        self.assertEqual(court.meta["agent_loop"], "omni-tars")
        self.assertEqual(court.meta["result"], "clicked")
        self.assertEqual(calls, [(7, "computer_click", {"x": 1}, "r-one")])
        self.assertIn("only-r-one", str(seen[0]))
        self.assertIn("receipt", str(seen[1]))
        self.assertEqual(frames, ["r-one", "r-one"])

    async def test_dead_hand_cannot_be_hidden_by_final_answer(self):
        court, _, _, _ = await self.drive([
            ChatResult("", [ToolCall("c1", "computer_click", "{}")]), ChatResult("could not click")],
            outputs=('{"error":"hand disconnected"}',))
        self.assertEqual(court.status, "failed")
        self.assertIn("hand disconnected", court.error)

    async def test_successful_recovery_clears_tool_error(self):
        court, calls, _, _ = await self.drive([
            ChatResult("", [ToolCall("c1", "computer_click", "{}")]),
            ChatResult("", [ToolCall("c2", "computer_click", "{}")]), ChatResult("recovered")],
            outputs=('{"error":"temporary"}', "receipt"))
        self.assertEqual(court.status, "done")
        self.assertEqual(len(calls), 2)

    async def test_unknown_tool_and_malformed_arguments_fail_before_dispatch(self):
        for name, args in [("unadvertised", "{}"), ("computer_click", "[]"), ("computer_click", "bad")]:
            with self.subTest(name=name, args=args):
                court, calls, _, _ = await self.drive([ChatResult("", [ToolCall("c1", name, args)])])
                self.assertEqual(court.status, "failed")
                self.assertEqual(calls, [])

    async def test_two_courts_never_cross_lanes(self):
        results = await asyncio.gather(*[
            self.drive([ChatResult("", [ToolCall("c1", "computer_click", "{}")]),
                        ChatResult("done")], lane=lane)
            for lane in ["r-one", "r-two"]])
        self.assertEqual([result[0].status for result in results], ["done", "done"])
        self.assertEqual([result[1][0][-1] for result in results], ["r-one", "r-two"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
