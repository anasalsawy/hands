"""Desktop GUI moved to Open Computer Use in the authenticated desktop shell.

No shared server-local Windows/PyAutoGUI driver is offered to Companion.
"""
import json

class Desktop:
    enabled = False
    driver = None
    def __init__(self, *args, **kwargs): pass
    def schemas(self): return []
    def names(self): return set()
    async def call(self, name, args):
        return json.dumps({"error":"Desktop controls use Open Computer Use in the Companion desktop app"})
