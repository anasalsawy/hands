"""Mint a persona against the SAME store stitch.py uses, and print the FULL key.

stitch.py only logs `key=sp_XXXX…YYYY`, so the operator cannot copy a usable
key. This creates one and prints it in full so the deck can be tested.

Usage: python scripts/_mint_key.py [data_dir]
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def main() -> None:
    default = Path(__file__).resolve().parents[2] / "data" / "sawii.db"
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    from persona.identity import create_persona
    from persona.store import Store

    store = Store(str(db))
    pass  # Store opens lazily
    pid, key = await create_persona(store, name="deck test", definition="deck test", preset="default")
    print(f"PID={pid}")
    print(f"KEY={key}")
    print(f"DB={db}")
    pass


asyncio.run(main())
