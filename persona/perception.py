"""Read-only per-persona browser/computer evidence for Companion's L lane."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json


async def screen_frame(mind, pid, *, timeout=3.0):
    """Read independent surfaces concurrently; one slow hand cannot stall L."""
    sources = []

    async def browser(one):
        from .gateway.browser.session import read_page
        snap = await read_page(one.session, screenshot=True, ground=False)
        item = {'source':'browser','url':snap.url,'title':snap.title,
                'text':snap.visible_text[:14000],'elements':snap.elements[:80]}
        if snap.screenshot_png:
            item['image_url'] = 'data:image/png;base64,' + base64.b64encode(snap.screenshot_png).decode()
        return item

    async def desktop_frame(desktop):
        raw = json.loads(await desktop.call('desktop_screenshot', {}))
        item = {'source':'desktop',**{k:v for k,v in raw.items() if k != 'image_b64'}}
        if raw.get('image_b64'):
            item['image_url'] = 'data:image/png;base64,' + raw['image_b64']
        return item

    hands = getattr(mind, 'hands', None)
    names = hands.names(pid) if hands else set()

    async def hand(source, lane='', offered=None):
        offered = names if offered is None else offered
        route = {'lane':lane} if lane else {}
        if source + '_ocu_get_app_state' in offered:
            item = await hands.observe(pid, source, **route)
            return {**item, 'lane':lane} if item and lane else item
        text = str(await hands.call(pid, source + '_screen_text', {}, **route))[:14000]
        item = {'source':source,'text':text}
        if lane:
            item['lane'] = lane
        if source + '_screenshot' in offered:
            raw = json.loads(await hands.call(pid, source + '_screenshot', {}, **route))
            if raw.get('image_b64'):
                item['image_url'] = 'data:image/png;base64,' + raw['image_b64']
        return item

    hub = getattr(mind, 'browser_hub', None)
    one = getattr(hub, '_ones', {}).get(pid)
    if one is not None:
        sources.append(('browser', browser(one)))
    desktop = getattr(mind, 'desktop', None)
    if getattr(desktop, 'driver', None) is not None:
        sources.append(('desktop', desktop_frame(desktop)))
    for source in ('computer','cloud'):
        runners = sorted((lane, runner) for (owner, place, lane), runner in
                         getattr(hands, 'runners', {}).items() if owner == pid and place == source)
        if runners:
            for lane, runner in runners:
                offered = getattr(runner, 'names', {})
                if source + '_ocu_get_app_state' in offered or source + '_screen_text' in offered:
                    label = f'{source}:{lane}' if lane else source
                    sources.append((label, hand(source, lane, offered)))
        elif source + '_ocu_get_app_state' in names or source + '_screen_text' in names:
            sources.append((source, hand(source)))

    results = await asyncio.gather(*(asyncio.wait_for(read, timeout=timeout)
                                     for _, read in sources), return_exceptions=True)
    screens, errors = [], []
    for (source, _), result in zip(sources, results):
        if isinstance(result, BaseException):
            errors.append({'source':source,'error':type(result).__name__})
        elif result:
            screens.append(result)
    stable = [{**{k:v for k,v in row.items() if k != 'image_url'},
               'image_digest':hashlib.sha256(row.get('image_url', '').encode()).hexdigest()}
              for row in screens]
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()
    return {'state_digest':digest,'screens':screens,'errors':errors}
