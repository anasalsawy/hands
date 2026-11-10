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
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona.court import HandsDeck
from persona.llm import ChatResult, ToolCall
from persona.link import AgentProfile, Link, LinkedLane
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


class MotorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_kills_descendants_before_delayed_mutation_without_stopping_other_lane(self):
        profile = AgentProfile(name="test", strategy="spawn")
        link = Link(profile)
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "must-not-exist"
            child = ("import signal,time; from pathlib import Path; "
                     "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                     "print('ready',flush=True); time.sleep(5); "
                     f"Path({str(sentinel)!r}).write_text('SURVIVED')")
            parent = ("import subprocess,sys,time; "
                      f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE); "
                      "print(p.stdout.readline().decode().strip(),flush=True); time.sleep(60)")
            async def motor(lane, code):
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, start_new_session=os.name != "nt")
                linked = LinkedLane(lane_id=lane, agent=profile, process=proc, status="active",
                                    _process_group=os.name != "nt")
                link._lanes[lane] = linked
                return proc
            a = await motor("a", parent)
            b = await motor("b", "import time; time.sleep(60)")
            try:
                ready = await asyncio.wait_for(a.stdout.readline(), 10)
                self.assertEqual(ready.strip(), b"ready")
                await link.stop("a")
                self.assertIsNotNone(a.returncode)
                self.assertIsNone(b.returncode)
                await asyncio.sleep(5.2)
                self.assertFalse(sentinel.exists(), "a stopped lane's child performed a later write")
                self.assertEqual(link._lanes["a"].status, "stopped")
            finally:
                if a.returncode is None:
                    await link.stop("a")
                await link.stop("b")

    async def test_intentional_sigterm_remains_stopped_not_failed(self):
        profile = AgentProfile(name="test", strategy="spawn")
        link = Link(profile)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        linked = LinkedLane(lane_id="test", agent=profile, process=process,
                            status="active", connected=True)
        exits = []
        linked.on_exit = lambda rc, error: exits.append((rc, error))
        link._lanes[linked.lane_id] = linked
        monitor = asyncio.create_task(link._monitor_process(linked))
        try:
            self.assertTrue(await link.stop(linked.lane_id))
            await monitor
            self.assertIsNotNone(process.returncode)
            self.assertEqual(linked.status, "stopped")
            self.assertEqual(linked.error, "")
            self.assertEqual(len(exits), 1)
            self.assertEqual(exits[0][1], "")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def test_unexpected_exit_still_reports_failure(self):
        profile = AgentProfile(name="test", strategy="spawn")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "raise SystemExit(3)",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        linked = LinkedLane(lane_id="test", agent=profile, process=process, status="active")
        await Link(profile)._monitor_process(linked)
        self.assertEqual(linked.status, "failed")
        self.assertIn("code 3", linked.error)


class CourtTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_goal_on_listening_motor_starts_r_and_steering_reuses_loop(self):
        entered, release = asyncio.Event(), asyncio.Event()
        seen, calls, steered = [], [], []

        async def chat(messages, **kwargs):
            seen.append(messages)
            if len(seen) == 1:
                entered.set()
                await release.wait()
                return ChatResult("", [ToolCall("c1", "computer_click", '{"x":1}')])
            return ChatResult("clicked with guidance")

        async def call(pid, name, args, **kwargs):
            calls.append((name, args))
            return "receipt"

        async def steer(lane, goal):
            steered.append((lane, goal))

        hub = SimpleNamespace(_key=lambda pid, place, lane: (pid, place, lane),
                              runners={(7, "computer", "ready"): SimpleNamespace(tools=[TOOL])},
                              call=call, observe=lambda *args, **kw: asyncio.sleep(0))
        link = SimpleNamespace(agent=SimpleNamespace(place="computer"), steer=steer)
        mind = SimpleNamespace(a=SimpleNamespace(chat=chat), hands=hub,
                               links=SimpleNamespace(get=lambda name: link), bus=None)
        deck = HandsDeck(mind)
        court = deck.create(7, title="already started")
        court.lane, court.link, court.status = "ready", "hands-engine", "running"
        self.assertEqual(court.goal, "")
        await deck.send(7, court.id, "click once")
        self.assertEqual(len(deck._work_tasks), 1)
        await asyncio.wait_for(entered.wait(), 10)
        original = next(iter(deck._work_tasks))
        await deck.send(7, court.id, "then report briefly")
        self.assertEqual(list(deck._work_tasks), [original], "steering must not start a second loop")
        self.assertEqual(steered, [("ready", "then report briefly")])
        release.set()
        await asyncio.wait_for(original, 10)
        self.assertEqual(court.status, "done", court.error)
        self.assertEqual(calls, [("computer_click", {"x": 1})])
        self.assertIn("Updated user guidance for this lane: then report briefly", str(seen[1]))

    async def test_stop_cancels_only_own_r_task_and_never_emits_failure(self):
        stopped = asyncio.Event()
        link = SimpleNamespace()
        mind = SimpleNamespace(bus=None, links=SimpleNamespace(get=lambda name: link))
        deck = HandsDeck(mind)
        court = deck.create(7, title="cancel")
        court.status, court.lane, court.link = "running", "r-cancel", "hands-engine"
        court.error = "old error"
        events = deck.subscribe(7)

        async def wait_for_cancel():
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def stop(lane):
            self.assertEqual(lane, "r-cancel")
            self.assertEqual(court.status, "stopped")
            self.assertTrue(stopped.is_set(), "R must stop before killing its motor")
            deck._motor_died(7, court.id)(-15, "intentional SIGTERM")

        link.stop = stop
        own = asyncio.create_task(wait_for_cancel(), name=f"court-work-{court.id}")
        other = asyncio.create_task(asyncio.sleep(10), name="court-work-other")
        deck._work_tasks.update([own, other])
        await asyncio.sleep(0)
        try:
            await deck.cancel(7, court.id)
            self.assertTrue(own.cancelled())
            self.assertFalse(other.done())
            self.assertEqual(court.status, "stopped")
            self.assertEqual(court.error, "")
            recorded = []
            while not events.empty():
                recorded.append(events.get_nowait())
            self.assertFalse(any(ev.get('status') == 'failed' for ev in recorded))
        finally:
            other.cancel()
            await asyncio.gather(other, return_exceptions=True)

    async def drive(self, replies, outputs=("receipt",), *, lane="r-one", image=None, l_read="",
                    l_seen=None):
        seen, calls, frames, records = [], [], [], []
        scripted = iter(replies)
        results = iter(outputs)

        async def chat(messages, **kwargs):
            if kwargs["tools"] is None:
                if l_seen is None:
                    raise AssertionError("L read without a frame")
                l_seen.append(messages)
                if isinstance(l_read, Exception):
                    raise l_read
                return ChatResult(l_read)
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
            frame = {"text": f"only-{lane}"}
            shot = image.pop(0) if isinstance(image, list) else image
            return {**frame, "image_url": shot} if shot else frame

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
        self.assertEqual([record['kind'] for record in records if record['kind'] != 'l_vision'],
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

    async def test_screen_goes_through_l_and_r_gets_ls_words_with_pixels(self):
        l_seen, img = [], "data:image/png;base64,AAAA"
        court, calls, seen, _ = await self.drive([
            ChatResult("", [ToolCall("c1", "computer_click", '{"x":1}')]), ChatResult("clicked")],
            image=img, l_read="Login form; Submit button bottom right", l_seen=l_seen)
        self.assertEqual(court.status, "done")
        self.assertEqual(len(l_seen), 1, "an unchanged screen reuses L's read")
        self.assertEqual(court.meta["l_vision_reused"], 1)
        self.assertIn(img, str(l_seen[0]))
        self.assertIn("only-r-one", str(l_seen[0]), "L reads only its own lane")
        for messages in seen:
            self.assertIn("L sees: Login form; Submit button bottom right", str(messages))
            self.assertIn(img, str(messages), "R still gets the raw pixels")
        self.assertEqual(len(calls), 1, "L never dispatches tools")
        self.assertIn("l_vision_ms", court.meta)

    async def test_v_rereads_changed_screen_and_never_lets_l_act(self):
        l_seen, shots = [], ["data:image/png;base64,ONE", "data:image/png;base64,TWO"]
        court, calls, seen, _ = await self.drive([
            ChatResult("", [ToolCall("c1", "computer_click", '{"x":1}')]), ChatResult("clicked")],
            image=shots, l_read="screen", l_seen=l_seen)
        self.assertEqual(court.status, "done")
        self.assertEqual(len(l_seen), 2)
        self.assertIn("ONE", str(l_seen[0]))
        self.assertIn("TWO", str(l_seen[1]))
        self.assertIn("TWO", str(seen[1]), "R gets the frame captured after its action")
        self.assertEqual(len(calls), 1)

    async def test_browser_card_drives_its_own_lane_browser_and_l_reads_it(self):
        import dataclasses
        from persona.gateway.browser.routes import BrowserHub
        from persona.gateway.browser.session import DryRunSession
        original = DryRunSession.snapshot

        async def snapshot(self, **kwargs):
            return dataclasses.replace(await original(self, **kwargs), screenshot_png=b"PNG-" + self.current_url.encode())

        r_seen, l_seen, advertised = [], [], []
        replies = iter([ChatResult("", [ToolCall("b1", "browser_navigate", '{"url":"https://example.com"}')]),
                        ChatResult("Learn more")])

        async def chat(messages, **kwargs):
            if kwargs["tools"] is None:
                l_seen.append(messages)
                return ChatResult("Example Domain page with a Learn more link")
            advertised.append([t["function"]["name"] for t in kwargs["tools"]])
            r_seen.append(messages)
            return next(replies)

        async def append(pid, **kwargs):
            pass

        browser = BrowserHub(dry_run=True)
        hub = SimpleNamespace(_key=lambda pid, place, lane: (pid, place, lane), runners={})
        mind = SimpleNamespace(a=SimpleNamespace(chat=chat), hands=hub, links=None, bus=None,
                               browser_hub=browser, shared_timeline=SimpleNamespace(append=append))
        deck = HandsDeck(mind)
        court = deck.create(7, title="web", kind="browser")
        court.lane, court.status = "r-web", "running"
        with patch.object(DryRunSession, "snapshot", snapshot):
            await deck._drive(7, court.id, "r-web", "open example.com")
        self.assertEqual(court.status, "done", court.error)
        self.assertIn("browser_navigate", advertised[0])
        self.assertNotIn("computer_shell", advertised[0])
        lane_browser = browser._ones[(7, "r-web")].session
        self.assertEqual(lane_browser.current_url, "https://example.com")
        self.assertNotIn(7, browser._ones, "the persona's own pane browser is untouched")
        self.assertIn("example.com", str(l_seen[0]))
        self.assertIn("L sees: Example Domain page", str(r_seen[1]))
        self.assertIn("persona_prepare_next_actions", str(r_seen[0]), "R is told to plan ahead")
        deck.remove(7, court.id)
        for _ in range(20):
            if (7, "r-web") not in browser._ones:
                break
            await asyncio.sleep(0.01)
        self.assertNotIn((7, "r-web"), browser._ones, "removing the card closes its lane browser")
        await browser.shutdown()

    async def test_l_vision_failure_falls_back_to_raw_frame(self):
        l_seen, img = [], "data:image/png;base64,BBBB"
        court, _, seen, _ = await self.drive([ChatResult("nothing to do")], image=img,
                                             l_read=RuntimeError("L down"), l_seen=l_seen)
        self.assertEqual(court.status, "done")
        self.assertIn(img, str(seen[0]))
        self.assertNotIn("L sees:", str(seen[0]))
        self.assertIn("L down", court.meta["l_vision_error"])

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
