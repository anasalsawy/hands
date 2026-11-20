"""Web reading for the engine: search and fetch, with no API key and no browser.

Fetching refuses private and link-local addresses, so a hosted engine cannot be
pointed at the host's own network (metadata endpoints, other services).
"""
from __future__ import annotations

import html as htmllib
import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (compatible; HandsEngine/1.0)"
MAX_BYTES = 1_500_000

WEB_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web. Returns the top results (title, url, snippet). Follow up with web_fetch to read a page.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "max_results": {"type": "integer", "description": "default 8, max 15"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Fetch a web page or text file and return its readable text (no scripts). Use for reading sources and docs.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "max_chars": {"type": "integer", "description": "default 4000, max 40000"}},
            "required": ["url"]}}},
]
WEB_NAMES = {t["function"]["name"] for t in WEB_TOOLS}


def _public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


class _NoPrivateRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlparse(newurl).hostname or ""
        if not _public(host):
            raise urllib.error.URLError("redirect to a private address refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get(url: str, timeout: float = 25) -> tuple[bytes, str]:
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("only http(s) URLs")
    if not _public(p.hostname):
        raise ValueError("that address is private or unreachable; only public websites can be fetched")
    opener = urllib.request.build_opener(_NoPrivateRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,text/plain,*/*;q=0.5"})
    with opener.open(req, timeout=timeout) as r:
        data = r.read(MAX_BYTES)
        return data, r.headers.get("content-type", "")


def _text(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", markup)
    markup = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)[^>]*>", "\n", markup)
    markup = re.sub(r"(?s)<[^>]+>", " ", markup)
    markup = htmllib.unescape(markup)
    markup = re.sub(r"[ \t\r\f\v]+", " ", markup)
    return re.sub(r"\n\s*\n+", "\n\n", markup).strip()


def web_fetch(url: str, max_chars: int = 4000) -> str:
    max_chars = max(500, min(int(max_chars or 4000), 40000))
    try:
        data, ctype = _get(url)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    enc = re.search(r"charset=([\w-]+)", ctype or "")
    body = data.decode(enc.group(1) if enc else "utf-8", errors="replace")
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
    text = _text(body) if "html" in (ctype or "").lower() or body.lstrip().startswith("<") else body
    return json.dumps({"url": url, "title": htmllib.unescape(title.group(1)).strip() if title else "",
                       "text": text[:max_chars], "truncated": len(text) > max_chars})


def web_search(query: str, max_results: int = 8) -> str:
    n = max(1, min(int(max_results or 8), 15))
    try:
        data, _ = _get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}))
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"search failed: {type(exc).__name__}: {exc}"})
    page = data.decode("utf-8", errors="replace")
    out = []
    for m in re.finditer(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)', page):
        href, title, rest = m.group(1), _text(m.group(2)), m.group(3)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
        url = q[0] if q else href
        if url.startswith("//"):
            url = "https:" + url
        snip = re.search(r'(?is)class="result__snippet"[^>]*>(.*?)</a>', rest)
        out.append({"title": title, "url": url, "snippet": _text(snip.group(1)) if snip else ""})
        if len(out) >= n:
            break
    if not out:
        return json.dumps({"error": "no results (the search page may have changed or blocked this request)"})
    return json.dumps({"query": query, "results": out})
