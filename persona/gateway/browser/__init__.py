"""Browser gateway — Shape B wrap of browser_use for Sawii's companion app.

See README.md for the Shape A vs Shape B distinction and
why this module exists.

Public surface:
    - BrowserTools      : ToolsCap over a browser session (for the kernel)
    - BrowserPaneInsight: InsightCap(mode='L') reading the page
    - BrowserBroadcast  : screenshot streaming to UI viewers
    - register_browser_routes: mounts the browser pane routes onto the app

Session abstraction (session.py) imports browser_use lazily. If browser_use
is not installed, the module still imports and tests pass against the
DryRunSession stub. For real browser work, add `browser-use` to the
environment.
"""
from .insight import BrowserPaneInsight
from .routes import BrowserHub, register_browser_routes
from .session import BrowserSessionProto, DryRunSession, RealSession, Snapshot, create_browser_session
from .stream import BrowserBroadcast
from .tools import BrowserTools

__all__ = [
    "BrowserBroadcast",
    "BrowserHub",
    "BrowserPaneInsight",
    "BrowserSessionProto",
    "BrowserTools",
    "DryRunSession",
    "RealSession",
    "Snapshot",
    "create_browser_session",
    "register_browser_routes",
]
