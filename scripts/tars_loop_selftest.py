"""Exercise the real vendored Omni-TARS loop through Persona's private bridge.

    npm ci --prefix engine
    python scripts/tars_loop_selftest.py

Only inference and OS tools are scripted; the TypeScript agent/composer,
native call parser, history and lifecycle are the production implementation.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
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
            with self.assertRaisesRegex(RuntimeError, "npm run build:tars --prefix engine"):
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

    async def spec_run(self, spec, autofire, receipt="receipt"):
        calls, thinks, stats = [], [], {}

        async def infer(messages):
            thinks.append(messages)
            if len(thinks) == 1:
                return {"text": "", "tool_calls": [
                    {"id": "p1", "name": PLAN_NAME, "arguments": spec},
                    {"id": "c1", "name": "computer_click", "arguments": {"x": 1}}]}
            return {"text": "done", "tool_calls": []}

        async def call(name, args):
            calls.append(args["x"])
            return receipt

        async def frame():
            return None

        result = await run_tars_loop(goal="test", instructions="test", tools=[TOOL, PLAN_TOOL],
                                     infer=infer, execute=call, observe=frame,
                                     autofire=autofire, stats=stats)
        return result, calls, thinks, stats

    async def test_persona_spec_is_advice_when_autofire_is_off(self):
        spec = {"confidence": 0.95, "possibilities": {
            "ok": {"name": "computer_click", "arguments": {"x": 2}}}}
        result, calls, thinks, stats = await self.spec_run(spec, autofire=False)
        self.assertEqual((result, calls), ("done", [1]))
        self.assertIn('"x": 2', str(thinks[1][-1]))
        self.assertEqual(stats.get("matched"), 1)
        self.assertNotIn("fired", stats)

    async def test_confident_matching_spec_fires_without_another_think(self):
        spec = {"confidence": 0.95, "possibilities": {
            "ok": {"name": "computer_click", "arguments": {"x": 2}}}}
        result, calls, thinks, stats = await self.spec_run(spec, autofire=True)
        self.assertEqual((result, calls, len(thinks)), ("done", [1, 2], 2))
        self.assertEqual(stats.get("fired"), 1)
        self.assertIn("prepared step fired", str(thinks[1]))

    async def test_batch_fires_in_order_only_while_each_outcome_matches(self):
        spec = {"confidence": 0.95, "possibilities": {}, "batch": [
            {"name": "computer_click", "arguments": {"x": 2}, "_expected_outcome": "ok"},
            {"name": "computer_click", "arguments": {"x": 3}, "_expected_outcome": "ok"}]}
        result, calls, thinks, stats = await self.spec_run(spec, autofire=True)
        self.assertEqual((result, calls, len(thinks), stats.get("fired")), ("done", [1, 2, 3], 2, 2))
        result, calls, thinks, stats = await self.spec_run(spec, True, '{"error": "gone"}')
        self.assertEqual((result, calls), ("done", [1]), "a failed step stops the batch")

    async def test_spec_never_fires_on_wrong_outcome_or_low_confidence(self):
        for spec, receipt in (
            ({"confidence": 0.95, "possibilities": {
                "ok": {"name": "computer_click", "arguments": {"x": 2}}}}, '{"error": "no window"}'),
            ({"confidence": 0.6, "possibilities": {
                "ok": {"name": "computer_click", "arguments": {"x": 2}}}}, "receipt"),
        ):
            result, calls, thinks, stats = await self.spec_run(spec, True, receipt)
            self.assertEqual((result, calls), ("done", [1]), spec)
            self.assertNotIn("fired", stats)

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
    """Court tests run the REAL Agent TARS engine: its own filesystem, terminal and
    browser tools execute; only Persona's inference (the lobe) is scripted."""

    @classmethod
    def setUpClass(cls):
        import http.server
        import threading

        class Page(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"<html><body><h1>COURT-PAGE-OK</h1><a href='/x'>Learn more</a></body></html>"
                self.send_response(200)
                self.send_header("content-type", "text/html")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Page)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/"
        cls.work = tempfile.mkdtemp(prefix="hands-court-")
        cls.env = mock.patch.dict(os.environ, {
            "HANDS_ROOT": cls.work, "HANDS_HEADLESS": "1", "SAWII_CHROME": os.getenv("SAWII_CHROME", "/usr/bin/google-chrome"),
            "SAWII_HANDS_APPROVE": "auto"})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.server.shutdown()

    async def drive(self, replies, *, kind="browser", lane="r-one", l_read="", l_seen=None, goal="work", deck_out=None):
        seen, advertised, records = [], [], []
        scripted = iter(replies)

        async def chat(messages, **kwargs):
            if kwargs["tools"] is None:
                if l_seen is None:
                    raise AssertionError("L read without a frame")
                l_seen.append(messages)
                if isinstance(l_read, Exception):
                    raise l_read
                return ChatResult(l_read)
            advertised.append({t["function"]["name"] for t in kwargs["tools"]})
            seen.append(messages)
            return next(scripted)

        async def append(pid, **kwargs):
            records.append(kwargs)

        async def push_screen(*args, **kwargs):
            return False   # this brain has no V feed: L reads the frame itself

        mind = SimpleNamespace(a=SimpleNamespace(chat=chat, push_screen=push_screen), hands=None, links=None,
                               bus=None, shared_timeline=SimpleNamespace(append=append))
        deck = HandsDeck(mind)
        if deck_out is not None:
            deck_out.append(deck)
        court = deck.create(7, title="test", kind=kind)
        court.lane, court.status = lane, "running"
        await deck._drive(7, court.id, lane, goal)
        return court, seen, advertised, records

    async def test_agent_tars_executes_real_tools_and_persona_sees_real_receipts(self):
        target = os.path.join(self.work, "court.txt")
        court, seen, advertised, records = await self.drive([
            ChatResult("", [ToolCall("c1", "write_file", json.dumps({"path": target, "content": "from-tars"}))]),
            ChatResult("", [ToolCall("c2", "run_command", json.dumps({"command": f"cat {target}"}))]),
            ChatResult("wrote and read the file")])
        self.assertEqual(court.status, "done", court.error)
        self.assertEqual(court.meta["agent_loop"], "agent-tars")
        self.assertEqual(open(target).read(), "from-tars", "Agent TARS itself wrote the file")
        self.assertIn("from-tars", str(seen[2]), "the terminal receipt reached Persona's next decision")
        # TARS's own tool catalog is what the lobe is offered (browser, files, terminal, search).
        self.assertTrue({"write_file", "run_command", "browser_navigate", "browser_get_markdown",
                         "persona_prepare_next_actions"} <= advertised[0])
        self.assertEqual([r["kind"] for r in records if r["kind"] in ("tool_call", "tool_result")],
                         ["tool_call", "tool_result"] * 2, "L and R share the real call/result receipts")

    async def test_dom_browser_runs_inside_agent_tars_and_card_frame_comes_from_it(self):
        l_seen = []
        court, seen, _, _ = await self.drive([
            ChatResult("", [ToolCall("b1", "browser_navigate", json.dumps({"url": self.url}))]),
            ChatResult("", [ToolCall("b2", "browser_get_markdown", "{}")]),
            ChatResult("saw the page")], l_read="Page titled COURT-PAGE-OK", l_seen=l_seen)
        self.assertEqual(court.status, "done", court.error)
        self.assertIn("COURT-PAGE-OK", str(seen[2]), "DOM markdown extraction by TARS reached Persona")
        self.assertTrue(l_seen, "the card's frame (a TARS screenshot) was read by L/V")
        self.assertIn("data:image/png;base64,", str(l_seen[0]))

    async def test_tool_error_receipt_fails_the_card_and_recovery_clears_it(self):
        missing = os.path.join(self.work, "nope", "missing.txt")
        court, _, _, _ = await self.drive([
            ChatResult("", [ToolCall("c1", "read_file", json.dumps({"path": missing}))]),
            ChatResult("could not read it")])
        self.assertEqual(court.status, "failed", "a failed tool cannot be hidden by a polite final answer")
        good = os.path.join(self.work, "ok.txt")
        court, _, _, _ = await self.drive([
            ChatResult("", [ToolCall("c1", "read_file", json.dumps({"path": missing}))]),
            ChatResult("", [ToolCall("c2", "write_file", json.dumps({"path": good, "content": "x"}))]),
            ChatResult("recovered")])
        self.assertEqual(court.status, "done", court.error)

    async def test_unknown_tool_and_malformed_arguments_fail_before_dispatch(self):
        for name, args in [("unadvertised", "{}"), ("write_file", "[]"), ("write_file", "bad")]:
            with self.subTest(name=name, args=args):
                court, _, _, _ = await self.drive([ChatResult("", [ToolCall("c1", name, args)])])
                self.assertEqual(court.status, "failed")

    async def test_person_can_decline_an_action_and_agent_tars_does_not_run_it(self):
        target = os.path.join(self.work, "declined.txt")
        decks = []
        with mock.patch.dict(os.environ, {"SAWII_HANDS_APPROVE": "ask"}):
            task = asyncio.create_task(self.drive([
                ChatResult("", [ToolCall("c1", "write_file", json.dumps({"path": target, "content": "no"}))]),
                ChatResult("it was declined")], deck_out=decks))
            for _ in range(1200):
                await asyncio.sleep(0.05)
                if decks and decks[0]._approvals:
                    break
            self.assertTrue(decks and decks[0]._approvals, "the card must be waiting for the person")
            cid = next(iter(decks[0]._approvals))
            self.assertEqual(decks[0].get(7, cid).status, "waiting")
            self.assertTrue(decks[0].resolve_approval(7, cid, False, "not now"))
            await task
        self.assertFalse(os.path.exists(target), "a declined action must never execute")

    async def test_two_courts_never_cross_lanes(self):
        results = await asyncio.gather(*[
            self.drive([ChatResult("", [ToolCall("c1", "list_directory", json.dumps({"path": self.work}))]),
                        ChatResult("done")], lane=lane)
            for lane in ["r-one", "r-two"]])
        self.assertEqual([r[0].status for r in results], ["done", "done"])
        for r, lane in zip(results, ["r-one", "r-two"]):
            self.assertTrue(all(rec["task_id"] == lane for rec in r[3] if rec["kind"] in ("tool_call", "tool_result")))


PAGE = ("data:text/html,<title>Form</title><h1>Sign</h1><input id=n placeholder=name>"
        "<button onclick=\"document.title='Hello '+document.getElementById('n').value\">Go</button>")


class HubTests(unittest.IsolatedAsyncioTestCase):
    """Persona's browser hub is Agent TARS: real Chrome, real TARS tools, no other browser stack."""

    async def asyncSetUp(self):
        from persona.gateway.browser.routes import BrowserHub
        self.hub = BrowserHub()

    async def asyncTearDown(self):
        await self.hub.shutdown()

    async def test_hub_drives_real_browser_through_tars_tools(self):
        await self.hub.run(1, "browser_navigate", {"url": PAGE})
        snap = json.loads(await self.hub.run(1, "browser_snapshot", {}))
        self.assertEqual(snap["title"], "Form")
        self.assertEqual([e["tag"] for e in snap["elements"]], ["input", "button"])
        await self.hub.run(1, "browser_type", {"index": 0, "text": "Anas"})
        await self.hub.run(1, "browser_click", {"index": 1})
        self.assertEqual(json.loads(await self.hub.run(1, "browser_snapshot", {}))["title"], "Hello Anas")

    async def test_tars_loop_runs_on_the_hubs_engine(self):
        from persona.gateway.browser.tasks import BrowserTasks
        steps = [
            ("browser_navigate", {"url": PAGE}),
            ("browser_get_markdown", {}),  # let the page settle, as a model reading the page would
            ("browser_get_clickable_elements", {}),
            ("browser_form_input_fill", {"index": 0, "value": "Loop"}),
            ("browser_click", {"index": 1}),
        ]
        seen_tools, position, attempts = [], [0], [0]

        async def infer(messages, tools):
            # A model reads the last receipt and retries a failed step: so does this one.
            seen_tools.append({t["function"]["name"] for t in tools}); tool_msgs = [m for m in messages if m.get("role") == "tool"]
            last = str(tool_msgs[-1].get("content", "")) if tool_msgs else ""
            if "Failed" in last[:80] and attempts[0] < 5:
                attempts[0] += 1
                await asyncio.sleep(0.5)
            elif seen_tools[1:]:
                position[0] += 1
            if position[0] >= len(steps):
                return {"text": "opened the form and said hello", "tool_calls": [], "finish_reason": "stop"}
            name, args = steps[position[0]]
            return {"text": "", "tool_calls": [{"id": f"c{len(seen_tools)}", "name": name, "arguments": args}],
                    "finish_reason": "stop"}
        published = []
        mind = SimpleNamespace(bus=SimpleNamespace(publish=lambda pid, ev: published.append(ev)), a=None)
        tasks = self.hub.tasks = BrowserTasks(self.hub, mind, infer_factory=infer)
        receipts = []

        async def on_receipt(name, args, result):
            receipts.append((name, str(result)[:100]))
        tasks.on_receipt = on_receipt
        tasks.start(1, "open the form and say hello")
        await asyncio.wait_for(tasks.jobs[1]["task"], 120)
        self.assertEqual(tasks.status(1)["state"], "finished", tasks.status(1))
        self.assertIn("browser_vision_control", seen_tools[0])  # the full TARS tool set, vision included
        # The loop and the hub share ONE browser: the hub sees what the TARS loop did.
        snap = json.loads(await self.hub.run(1, "browser_snapshot", {}))
        self.assertEqual(snap["title"], "Hello Loop", receipts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
