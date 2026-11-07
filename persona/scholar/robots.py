"""Robots.txt, read honestly.

Which URLs a crawler may fetch, at what pace. When a host has no robots.txt we
fetch freely; when it disallows us we skip the path and record *why* in the
manifest — the scholar's reputation outranks one missing page of fatwas.

Implements the widely-used subset: User-agent groups, Disallow/Allow with `*`
wildcards and trailing `$`, longest-match wins (Allow beats an equal-length
Disallow), `Crawl-delay` honoured. Comments and unknown fields ignored.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

_RULE = re.compile(r"^\s*(user-agent|disallow|allow|crawl-delay)\s*:\s*(.*)$", re.I)


def _compile(pattern: str) -> re.Pattern:
    """Translate a robots path pattern: * is .* , a trailing $ anchors."""
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    parts = re.split(r"(\*)", body)
    regex = "".join(".*" if p == "*" else re.escape(p) for p in parts)
    return re.compile(regex + ("$" if anchored else ""))


class Robots:
    """One host's rules. `allowed(url)` is the only judgment callers need."""

    def __init__(self, agent: str = "*") -> None:
        self.agent = (agent or "*").lower()
        self.rules: list[tuple[re.Pattern, bool, int]] = []   # (pattern, is_allow, len)
        self.crawl_delay: float = 0.0
        self.present = False        # did the host actually serve a robots.txt?

    def parse(self, text: str) -> "Robots":
        self.present = True
        groups: list[tuple[set[str], list[tuple[str, bool]], float]] = []
        agents: set[str] = set()
        fields: list[tuple[str, bool] | str] = []      # rules or ("delay", value)
        delay = 0.0

        def flush() -> None:
            nonlocal agents, fields, delay
            if agents and (fields or delay):
                rules = [f for f in fields if not isinstance(f, str)]
                groups.append((agents, rules, delay))
            agents, fields, delay = set(), [], 0.0

        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            m = _RULE.match(line)
            if not m:
                continue
            key, value = m.group(1).lower(), m.group(2).strip()
            if key == "user-agent":
                if fields or delay:
                    flush()
                agents.add(value.lower())
                continue
            if not agents:
                continue
            if key == "crawl-delay":
                try:
                    delay = max(0.0, float(value))
                except ValueError:
                    pass
            elif value:
                fields.append((value, key == "allow"))
        flush()

        chosen = [g for g in groups if self.agent in g[0]] or [g for g in groups if "*" in g[0]]
        seen: set[tuple[str, bool]] = set()
        for _ags, rules, _d in chosen:
            for pattern, is_allow in rules:
                if (pattern, is_allow) not in seen:
                    seen.add((pattern, is_allow))
                    self.rules.append((_compile(pattern), is_allow, len(pattern)))
        self.rules.sort(key=lambda r: (-r[2], not r[1]))   # longest first, Allow ties ahead
        self.crawl_delay = max((d for _a, _r, d in chosen), default=0.0)
        return self

    def allowed(self, url: str) -> bool:
        """The spec compares against the path; we keep query visible too — sites
        that hide content behind ?include= must not be fetched on a technicality."""
        if not self.rules:
            return True
        p = urlparse(url)
        path = p.path or "/"
        if p.query:
            path = f"{path}?{p.query}"
        for pattern, is_allow, _ in self.rules:
            if pattern.search(path):
                return is_allow
        return True
