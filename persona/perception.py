"""Read-only per-persona browser/computer evidence for Companion's L lane."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json


async def screen_frame(mind, pid):
    screens = []
    errors = []
    hub = getattr(mind, 'browser_hub', None)
    one = getattr(hub, '_ones', {}).get(pid)
    if one is not None:
        try:
            from .gateway.browser.session import read_page
            snap = await read_page(one.session, screenshot=True, ground=False)
            item = {'source':'browser','url':snap.url,'title':snap.title,
                    'text':snap.visible_text[:14000],'elements':snap.elements[:80]}
            if snap.screenshot_png:
                item['image_url'] = 'data:image/png;base64,' + base64.b64encode(snap.screenshot_png).decode()
            screens.append(item)
        except Exception as exc:
            errors.append({'source':'browser','error':type(exc).__name__})
    desktop = getattr(mind, 'desktop', None)
    if getattr(desktop, 'driver', None) is not None:
        try:
            raw = json.loads(await desktop.call('desktop_screenshot', {}))
            item = {'source':'desktop',**{k:v for k,v in raw.items() if k != 'image_b64'}}
            if raw.get('image_b64'):
                item['image_url'] = 'data:image/png;base64,' + raw['image_b64']
            screens.append(item)
        except Exception as exc:
            errors.append({'source':'desktop','error':type(exc).__name__})
    hands = getattr(mind, 'hands', None)
    names = hands.names(pid) if hands else set()
    for source in ('computer','cloud'):
        if source + '_ocu_get_app_state' in names:
            try:
                observed = await hands.observe(pid, source)
                if observed:
                    screens.append(observed)
            except Exception as exc:
                errors.append({'source':source,'error':type(exc).__name__})
            continue
        text_tool = source + '_screen_text'
        if text_tool not in names:
            continue
        try:
            text = str(await hands.call(pid, text_tool, {}))[:14000]
            item = {'source':source,'text':text}
            shot_tool = source + '_screenshot'
            if shot_tool in names:
                raw = json.loads(await hands.call(pid, shot_tool, {}))
                image = raw.get('image_b64')
                if image:
                    item['image_url'] = 'data:image/png;base64,' + image
            screens.append(item)
        except Exception as exc:
            errors.append({'source':source,'error':type(exc).__name__})
    stable = [{k:v for k,v in row.items() if k != 'image_url'} for row in screens]
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()
    return {'state_digest':digest,'screens':screens,'errors':errors}

