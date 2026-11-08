"""Talk the /v1/hands handshake by hand: does the server answer at all?

Usage: python scripts/_ws_probe.py <port> <key> [lane]
"""
import asyncio
import json
import sys

import websockets

port = sys.argv[1] if len(sys.argv) > 1 else "8802"
key = sys.argv[2] if len(sys.argv) > 2 else ""
lane = sys.argv[3] if len(sys.argv) > 3 else "probe"


async def main():
    url = f"ws://127.0.0.1:{port}/v1/hands"
    print("connecting", url, flush=True)
    try:
        async with asyncio.timeout(20):
            async with websockets.connect(url, max_size=2 ** 24) as ws:
                print("socket open", flush=True)
                await ws.send(json.dumps({"type": "hello", "key": key, "tools": [],
                                          "place": "computer", "lane": lane, "host": "probe"}))
                print("hello sent", flush=True)
                raw = await ws.recv()
                print("reply:", str(raw)[:300], flush=True)
    except SystemExit as exc:
        print("refused:", exc)
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {type(exc).__name__}: {exc}", flush=True)


asyncio.run(main())