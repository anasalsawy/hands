"""A local site that offers every structured surface, for testing API-first execution:
UCP profile + REST checkout, OpenAPI, GraphQL, an authorized endpoint, a WebMCP page and an
accessibility-friendly form. Also a plain page with only DOM. Usage: from scripts.web_fixture_server import serve"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ITEMS = [{"id": 1, "name": "tea"}]
SESSIONS: list[dict] = []
TOKEN = "Bearer sekrit"

WEBMCP_PAGE = """<!doctype html><title>todo</title><h1>Todos</h1><ul id=list></ul>
<script>
const mc = document.modelContext || navigator.modelContext;
mc.registerTool({name:'add-todo', description:'Add a todo item', inputSchema:{type:'object',properties:{text:{type:'string'}},required:['text']},
  execute: ({text}) => { const li=document.createElement('li'); li.textContent=text; document.getElementById('list').appendChild(li);
    return {content:[{type:'text', text:'added '+text}]}; }});
</script>"""

A11Y_PAGE = """<!doctype html><title>a11y</title>
<label for=n>Full name</label><input id=n>
<button onclick="document.title='Saved '+document.getElementById('n').value">Save profile</button>"""

SHARED_PAGE = """<!doctype html><title>shared</title><p id=fact>The tower opened in 1889.</p>
<button id=a onclick="this.textContent='A done'">A</button><button id=b onclick="this.textContent='B done'">B</button>"""

PLAIN_PAGE = "<!doctype html><title>plain</title><button>Only DOM here</button>"


def make_handler(port: int):
    base = f"http://127.0.0.1:{port}"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("content-length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/.well-known/ucp":
                return self._send(200, {"ucp": {"version": "2026-04-08", "services": {"dev.ucp.shopping": [
                    {"version": "2026-04-08", "transport": "rest", "endpoint": base + "/ucp/v1",
                     "spec": "https://ucp.dev/2026-04-08/specification/overview"}]},
                    "capabilities": {"dev.ucp.shopping.checkout": [{"version": "2026-04-08"}]}, "payment_handlers": {}},
                    "signing_keys": []})
            if p == "/openapi.json":
                return self._send(200, {"openapi": "3.0.0", "info": {"title": "Shop", "version": "1"}, "servers": [{"url": base + "/api"}],
                    "paths": {"/items": {"get": {"operationId": "listItems", "summary": "List"},
                                         "post": {"operationId": "createItem", "requestBody": {}}},
                              "/items/{id}": {"get": {"operationId": "getItem", "parameters": [{"name": "id", "in": "path", "required": True}]}}}})
            if p == "/api/items":
                return self._send(200, ITEMS)
            if p.startswith("/api/items/"):
                item = next((i for i in ITEMS if str(i["id"]) == p.rsplit("/", 1)[1]), None)
                return self._send(200 if item else 404, item or {"error": "not found"})
            if p == "/secure":
                ok = self.headers.get("authorization") == TOKEN
                return self._send(200 if ok else 401, {"secret": 42} if ok else {"error": "unauthorized"})
            page = {"/webmcp": WEBMCP_PAGE, "/a11y": A11Y_PAGE, "/plain": PLAIN_PAGE, "/shared": SHARED_PAGE,
                    "/one": "<title>one</title><h1>one</h1>", "/two": "<title>two</title><h1>two</h1>"}.get(p)
            if page:
                return self._send(200, page, "text/html")
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?")[0]
            body = self._body()
            if p == "/api/items":
                ITEMS.append({"id": len(ITEMS) + 1, "name": body.get("name", "")})
                return self._send(201, ITEMS[-1])
            if p == "/graphql":
                q = body.get("query", "")
                if "__typename" in q:
                    return self._send(200, {"data": {"__typename": "Query"}})
                if "mutation" in q:
                    ITEMS.append({"id": len(ITEMS) + 1, "name": (body.get("variables") or {}).get("name", "")})
                    return self._send(200, {"data": {"addItem": ITEMS[-1]}})
                if "items" in q:
                    return self._send(200, {"data": {"items": ITEMS}})
                return self._send(200, {"data": {"__schema": {"queryType": {"name": "Query"}, "types": [{"name": "Query", "kind": "OBJECT", "fields": [{"name": "items"}]}]}}})
            if p == "/ucp/v1/checkout-sessions":
                if not self.headers.get("UCP-Agent"):
                    return self._send(400, {"error": "UCP-Agent header required"})
                SESSIONS.append({"id": f"cs_{len(SESSIONS) + 1}", "agent": self.headers.get("UCP-Agent"), "line_items": body.get("line_items", [])})
                return self._send(201, SESSIONS[-1])
            return self._send(404, {"error": "not found"})
    return H


def serve(port: int = 0):
    """Start the site on a free port; returns (server, base_url)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(0))
    port = server.server_address[1]
    server.RequestHandlerClass = make_handler(port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"
