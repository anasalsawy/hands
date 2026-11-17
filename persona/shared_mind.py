"""Hands as the body of Persona's shared mind (sawii-persona ``shared_mind`` server).

One card = one task (Rn). The server decodes all of that task's hands (rn) and L in one
shared attention, so the hands split the goal between themselves. Hands only executes
what they ask for on the card's own browser tab and streams the results back:

- page-changing actions on one tab run in order; reads (``find``) overlap;
- every task gets its own tab, so several cards browse at the same time;
- steering goes to the live task instead of restarting it.

Enabled by ``SAWII_SHARED_MIND_URL``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

TOOLS = """goto {"url": "..."}: open a URL in your own browser page
click {"id": N}: click element [N] from your last page view
type {"id": N, "text": "...", "enter": true}: type into element [N]
find {"text": "..."}: search your page for text; returns the matching passages (fastest way to read a fact)
scroll {}: scroll down one screen
back {}: go back"""

FIND_JS = """(needle) => {
  const text = (document.body ? document.body.innerText : '').replace(/\\s+/g, ' ');
  const low = text.toLowerCase(), q = needle.toLowerCase(), out = [];
  let i = low.indexOf(q);
  while (i >= 0 && out.length < 6) {
    out.push(text.slice(Math.max(0, i - 160), i + q.length + 200));
    i = low.indexOf(q, i + q.length + 200);
  }
  if (out.length) return out;
  const words = q.split(/\\W+/).filter(w => w.length > 2);
  if (!words.length) return out;
  const stem = w => w.replace(/(ion|ed|ing|es|s)$/, '');
  const sents = text.split(/(?<=[.!?])\\s+/);
  const scored = sents.map((s, k) => {
    const l = s.toLowerCase();
    return [words.filter(w => l.includes(stem(w))).length, k];
  }).filter(([n]) => n >= Math.max(1, Math.ceil(words.length / 2))).sort((x, y) => y[0] - x[0] || x[1] - y[1]);
  for (const [, k] of scored.slice(0, 4)) out.push('(closest) ' + sents[k].slice(0, 360));
  return out;
}"""

READS = {"find"}


def enabled() -> bool:
    return bool(os.getenv("SAWII_SHARED_MIND_URL"))


class SharedMindClient:
    """HTTP client for the shared-mind server. One event poller feeds every task's queue."""

    def __init__(self, base: str, http=None):
        import httpx
        self.base = base.rstrip("/")
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30))
        self._queues: dict[str, asyncio.Queue] = {}
        self._early: dict[str, list[dict]] = {}
        self._since = 0
        self._poller: asyncio.Task | None = None

    async def _post(self, path: str, body: dict) -> dict:
        r = await self.http.post(self.base + path, json=body)
        r.raise_for_status()
        return r.json()

    async def reset(self, tools: str = TOOLS, **kw) -> dict:
        self._since = 0
        return await self._post("/reset", dict(tools=tools, **kw))

    async def chat(self, text: str) -> dict:
        return await self._post("/chat", dict(text=text))

    async def start(self, goal: str, hands: int = 2) -> tuple[str, asyncio.Queue]:
        if self._poller is None or self._poller.done():
            await self._catch_up()
            self._poller = asyncio.create_task(self._poll(), name="shared-mind-events")
        t = await self._post("/tasks", dict(goal=goal, hands=hands))
        q = self._queues.setdefault(t["tid"], asyncio.Queue())
        for ev in self._early.pop(t["tid"], []):
            q.put_nowait(ev)
        return t["tid"], q

    async def steer(self, tid: str, text: str) -> None:
        await self._post(f"/tasks/{tid}/steer", dict(text=text))

    async def cancel(self, tid: str) -> None:
        await self._post(f"/tasks/{tid}/cancel", {})

    async def observe(self, slot: int, text: str) -> None:
        await self._post(f"/hands/{slot}/observe", dict(text=text))

    def forget(self, tid: str) -> None:
        self._queues.pop(tid, None)

    async def _catch_up(self) -> None:
        """Skip history from before this client so old tasks never replay."""
        r = await self.http.get(self.base + "/events", params=dict(since=self._since, timeout=0))
        evs = r.json()
        if evs:
            self._since = evs[-1]["seq"] + 1

    async def _poll(self) -> None:
        while self._queues or self._poller is not None:
            try:
                r = await self.http.get(self.base + "/events", params=dict(since=self._since, timeout=20))
                evs = r.json()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a blip must not kill every card
                log.debug("shared mind poll failed", exc_info=True)
                await asyncio.sleep(0.5)
                continue
            for ev in evs:
                self._since = ev["seq"] + 1
                tid = ev.get("tid")
                if tid is None:
                    continue
                q = self._queues.get(tid)
                if q is not None:
                    q.put_nowait(ev)
                else:   # the event can beat POST /tasks' reply
                    self._early.setdefault(tid, []).append(ev)
                    if len(self._early) > 64:
                        self._early.pop(next(iter(self._early)))

    async def close(self) -> None:
        if self._poller is not None:
            self._poller.cancel()
        await self.http.aclose()


VIEW_JS = """() => {
  const els = [...document.querySelectorAll('a[href],button,input,select,textarea,[role=button]')]
    .filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight * 2; })
    .slice(0, 40);
  window.__ids = els;
  const items = els.map((e, i) => `[${i}] ${e.tagName.toLowerCase()} ${(e.innerText || e.value || e.placeholder || e.getAttribute('aria-label') || e.name || '').trim().replace(/\\s+/g, ' ').slice(0, 60)}`);
  const parts = []; let size = 0;
  const walk = document.createTreeWalker(document.body || document, NodeFilter.SHOW_TEXT);
  while (walk.nextNode() && size < 1500) {
    const n = walk.currentNode, t = n.textContent.trim(), p = n.parentElement;
    if (!t || !p || ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(p.tagName)) continue;
    const r = p.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight || r.width === 0) continue;
    parts.push(t); size += t.length + 1;
  }
  const text = parts.join(' ').replace(/\\s+/g, ' ').slice(0, 1500);
  return `URL: ${location.href}\\nTitle: ${document.title}\\nText: ${text}\\nElements:\\n${items.join('\\n')}`;
}"""

ACT_JS = """([tool, id, text]) => {
  const e = (window.__ids || [])[id];
  if (!e || !e.isConnected) return 'missing';
  window.__leaving = false;
  window.addEventListener('beforeunload', () => { window.__leaving = true; }, {once: true});
  e.scrollIntoView({block: 'center'});
  if (tool === 'click') { e.click(); return 'ok'; }
  e.focus();
  const proto = e instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const set = Object.getOwnPropertyDescriptor(proto, 'value');
  set && e instanceof HTMLElement && 'value' in e ? set.set.call(e, text) : (e.textContent = text);
  e.dispatchEvent(new Event('input', {bubbles: true}));
  e.dispatchEvent(new Event('change', {bubbles: true}));
  return 'ok';
}"""

SETTLE_JS = """new Promise(done => { const t0 = Date.now(); (function wait() {
  if (document.readyState !== 'loading' || Date.now() - t0 > 8000) done(location.href); else setTimeout(wait, 50); })(); })"""


def render_view(snap) -> str:
    els = "\n".join(f"[{e.get('index', i)}] {e.get('tag', '')} {str(e.get('text') or '')[:60]}".rstrip()
                    for i, e in enumerate((snap.elements or [])[:40]))
    text = " ".join(str(snap.visible_text or "").split())[:1500]
    return f"URL: {snap.url}\nTitle: {snap.title}\nText: {text}\nElements:\n{els}"


class TabHands:
    """Runs one task's actions on its own browser tab."""

    def __init__(self, run_tool: Callable[[str, dict], Awaitable[str]],
                 read_page: Callable[[], Awaitable[object]],
                 find: Callable[[str], Awaitable[list[str]]] | None = None,
                 press_enter: Callable[[], Awaitable[None]] | None = None,
                 evaluate: Callable[[str], Awaitable[object]] | None = None):
        self.run_tool, self.read_page, self.find = run_tool, read_page, find
        self.press_enter, self.evaluate = press_enter, evaluate
        self.lock = asyncio.Lock()   # page-changing actions on one tab stay ordered

    async def _view(self) -> str:
        if self.evaluate is not None:
            return str(await self._eval(f"({VIEW_JS})()"))
        return render_view(await self.read_page())

    async def _eval(self, expr: str):
        """Evaluate in the tab, riding out a navigation that replaces the page mid-call."""
        for attempt in range(20):
            try:
                return await self.evaluate(expr)
            except Exception:  # noqa: BLE001 - the old document went away; the new one is loading
                if attempt == 19:
                    raise
                await asyncio.sleep(0.1)

    async def _left(self, before: str, wait: float = 0.4) -> None:
        """Wait out a navigation a click or submit started; return at once if it started none."""
        end, leaving = time.time() + wait, False
        while time.time() < end:
            state = str(await self._eval("(window.__leaving ? 'L ' : 'S ') + document.readyState + ' ' + location.href"))
            if not state.endswith(" " + before) or " loading " in state:
                return
            if state.startswith("L ") and not leaving:
                leaving, end = True, time.time() + 10.0
            await asyncio.sleep(0.05)

    async def _settle(self) -> None:
        await asyncio.sleep(0.15)
        await self._eval(SETTLE_JS)

    async def act(self, tool: str, args: dict) -> str:
        if self.evaluate is not None and tool != "goto":
            return await self._act_in_page(tool, args)
        if tool == "find":
            needle = str(args.get("text", "")).strip()
            if self.find is not None:
                hits = await self.find(needle) if needle else []
            else:
                snap = await self.read_page()
                text = " ".join(str(snap.visible_text or "").split())
                i = text.lower().find(needle.lower()) if needle else -1
                hits = [text[max(0, i - 160): i + len(needle) + 200]] if i >= 0 else []
            snap = await self.read_page()
            return (f"URL: {snap.url}\nTitle: {snap.title}\nMatches for {needle!r}:\n"
                    + ("\n".join(f"- {h}" for h in hits) or "none"))
        async with self.lock:
            if tool == "goto":
                url = str(args.get("url", ""))
                out = await self.run_tool("browser_navigate", {"url": url if "://" in url else "https://" + url})
            elif tool == "click":
                out = await self.run_tool("browser_click", {"index": int(args.get("id", -1))})
            elif tool == "type":
                out = await self.run_tool("browser_type", {"index": int(args.get("id", -1)),
                                                           "text": str(args.get("text", ""))})
                if args.get("enter") and not _failed(out) and self.press_enter is not None:
                    await self.press_enter()
                await self._left(before)
            elif tool == "scroll":
                out = await self.run_tool("browser_scroll", {"direction": "down"})
            elif tool == "back":
                out = await self.run_tool("browser_back", {})
            else:
                return f"error: unknown tool {tool!r}"
            if _failed(out):
                return f"error: {str(out)[:300]}"
            return await self._view()

    async def _act_in_page(self, tool: str, args: dict) -> str:
        """Page actions run as one script in the tab: no full-DOM snapshot per action."""
        if tool == "find":
            needle = str(args.get("text", "")).strip()
            hits = await self._eval(f"({FIND_JS})({json.dumps(needle)})") if needle else []
            head = await self._eval("`URL: ${location.href}\\nTitle: ${document.title}`")
            return f"{head}\nMatches for {needle!r}:\n" + ("\n".join(f"- {h}" for h in hits or []) or "none")
        async with self.lock:
            if tool in ("click", "type"):
                ident = int(args.get("id", -1))
                before = await self._eval("location.href")
                res = await self._eval(f"({ACT_JS})({json.dumps([tool, ident, str(args.get('text', ''))])})")
                if res != "ok":
                    return f"error: no element [{ident}] in the last view"
                if tool == "type" and args.get("enter") and self.press_enter is not None:
                    await self.press_enter()
            elif tool == "scroll":
                await self._eval("window.scrollBy(0, innerHeight * 0.9)")
            elif tool == "back":
                before = await self._eval("location.href")
                await self._eval("history.back()")
                await self._left(before, wait=1.0)
            else:
                return f"error: unknown tool {tool!r}"
            await self._settle()
            return await self._view()


def _failed(out) -> bool:
    try:
        parsed = json.loads(str(out))
    except ValueError:
        return False
    return isinstance(parsed, dict) and bool(parsed.get("error") or parsed.get("isError"))


async def drive(client: SharedMindClient, goal: str, tab: TabHands, *, hands: int = 2,
                on_event: Callable[[dict], None] | None = None,
                on_task: Callable[[str], None] | None = None,
                timeout: float = 600.0) -> dict:
    """Run one goal as a shared-mind task on ``tab`` until it is done. Returns the done event
    plus per-action timings."""
    tid, q = await client.start(goal, hands)
    if on_task is not None:
        on_task(tid)
    timings: list[dict] = []
    last: dict[int, float] = {}
    pending: set[asyncio.Task] = set()

    async def run(ev: dict) -> None:
        t0 = time.time()
        try:
            text = await tab.act(ev["tool"], ev.get("args") or {})
        except Exception as exc:  # noqa: BLE001 - the hand reads its failure and adapts
            text = f"error: {type(exc).__name__}: {str(exc)[:200]}"
        slot = ev["slot"]
        prev = last.get(slot)
        last[slot] = ev["t"]
        timings.append(dict(slot=slot, tool=ev["tool"], think_secs=ev.get("think_secs"),
                            tool_secs=round(time.time() - t0, 3),
                            act_to_act=round(ev["t"] - prev, 3) if prev else None))
        await client.observe(slot, text)

    deadline = time.time() + timeout
    try:
        while True:
            ev = await asyncio.wait_for(q.get(), max(1.0, deadline - time.time()))
            if on_event is not None:
                on_event(ev)
            if ev["type"] == "act":
                task = asyncio.create_task(run(ev))
                pending.add(task)
                task.add_done_callback(pending.discard)
            elif ev["type"] == "done":
                return dict(ev, tid=tid, timings=timings)
    except BaseException:
        for task in pending:
            task.cancel()
        try:
            await asyncio.shield(client.cancel(tid))
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        client.forget(tid)
