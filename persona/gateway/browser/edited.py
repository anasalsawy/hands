"""Companion session backed by the exact edited Browser-Use SDK from Railway."""
from __future__ import annotations
import asyncio
import base64
import json
from .session import Snapshot, _chrome_path


class EditedBrowserSession:
    def __init__(self, *, headless=True, cdp_url=None):
        self.headless, self.cdp_url = headless, cdp_url
        self.native = None
        self.action_lock = asyncio.Lock()

    async def start(self):
        # No stock SDK / Playwright fallback: missing patches are a setup failure.
        from browser_use import BrowserSession, BrowserProfile
        from browser_use.persona.agent import PersonaBrowserAgent
        if not hasattr(PersonaBrowserAgent, '_start_persona_prefetch'):
            raise RuntimeError('Install the tested edited SDK with scripts/install_browser_sdk.py')
        profile = BrowserProfile(headless=self.headless, keep_alive=True, user_data_dir=None,
                                 executable_path=_chrome_path())
        self.native = BrowserSession(browser_profile=profile, cdp_url=self.cdp_url)
        await self.native.start()

    async def close(self):
        if self.native:
            await self.native.kill()
            self.native = None

    async def _event(self, event):
        if self.native is None:
            raise RuntimeError('Browser is not started')
        dispatched = self.native.event_bus.dispatch(event)
        await dispatched
        return await dispatched.event_result(raise_if_any=True, raise_if_none=False)

    async def navigate(self, url):
        from urllib.parse import urlsplit
        from browser_use.browser.events import NavigateToUrlEvent
        if urlsplit(url).scheme not in ('http', 'https'):
            raise ValueError('Only HTTP(S) URLs are supported')
        await self._event(NavigateToUrlEvent(url=url))
        return f'navigated to {url}'

    async def _node(self, index):
        node = await self.native.get_dom_element_by_index(index)
        if node is None:
            raise LookupError('Element is absent; read a fresh browser snapshot')
        return node

    async def click(self, index):
        from browser_use.browser.events import ClickElementEvent
        await self._event(ClickElementEvent(node=await self._node(index)))
        return f'clicked element {index}'

    async def type_text(self, index, text):
        from browser_use.browser.events import TypeTextEvent
        await self._event(TypeTextEvent(node=await self._node(index), text=text))
        return f'typed into element {index}'

    async def scroll(self, direction):
        from browser_use.browser.events import ScrollEvent
        await self._event(ScrollEvent(direction=direction, amount=500))
        return f'scrolled {direction}'

    async def go_back(self):
        from browser_use.browser.events import GoBackEvent
        await self._event(GoBackEvent())
        return 'went back'

    async def wait_for_selector(self, selector, timeout_ms=5000):
        # Evaluate a JSON-quoted selector, never concatenate executable selector text.
        deadline = asyncio.get_running_loop().time() + max(0, min(timeout_ms, 30000)) / 1000
        while True:
            cdp = await self.native.get_or_create_cdp_session(focus=False)
            result = await cdp.cdp_client.send.Runtime.evaluate(
                params={'expression':f'!!document.querySelector({json.dumps(selector)})', 'returnByValue':True},
                session_id=cdp.session_id)
            if result.get('exceptionDetails'):
                raise ValueError('Invalid CSS selector')
            if result.get('result', {}).get('value'):
                return f'found {selector}'
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError('Selector did not appear')
            await asyncio.sleep(.1)

    async def snapshot(self, *, screenshot=True, ground=True):
        if self.native is None:
            raise RuntimeError('Browser is not started')
        elements = []
        if ground:
            state = await self.native.get_browser_state_summary(include_screenshot=False)
            dom = state.dom_state
            text = dom.llm_representation() if dom else ''
            for index, node in (dom.selector_map if dom else {}).items():
                elements.append({'index':index, 'tag':node.tag_name,
                                 'text':node.get_all_children_text(max_depth=2)[:120],
                                 'attrs':{k:v for k,v in node.attributes.items() if k in ('aria-label','href','name','placeholder')}})
            url, title = state.url, state.title
        else:
            # The viewing/observation path never overwrites the SDK selector cache.
            from browser_use.persona.observer import _READ_STATE
            cdp = await self.native.get_or_create_cdp_session(focus=False)
            raw = await cdp.cdp_client.send.Runtime.evaluate(
                params={'expression':_READ_STATE, 'returnByValue':True}, session_id=cdp.session_id)
            if raw.get('exceptionDetails'):
                raise RuntimeError('Browser observation failed')
            data = json.loads(raw['result']['value'])
            url, title, text = data['url'], data['title'], data.get('text','')
        png = None
        if screenshot:
            cdp = await self.native.get_or_create_cdp_session(focus=False)
            shot = await cdp.cdp_client.send.Page.captureScreenshot(
                params={'format':'png'}, session_id=cdp.session_id)
            png = base64.b64decode(shot['data'])
        return Snapshot(url=url, title=title, visible_text=text, elements=elements, screenshot_png=png)
