// Research-driven execution capabilities, added to Agent TARS as native tools.
//
// Upstream TARS executes the web through the DOM and through vision. The research on
// agents (API-first execution, WebMCP, UCP) says a web task should first ask whether the site
// already offers a structured surface and only then fall back to pixels. These tools extend
// TARS's own operator architecture (same Tool class, same approval/receipt path, same event
// stream) with those surfaces. They are NOT upstream; see docs/ARCHITECTURE_AUDIT.md section 9.
//
//   api_request            authorized REST call (allowlisted hosts, credentials from config only)
//   openapi_discover/call  read an OpenAPI document and call its operations
//   graphql_query          GraphQL query / introspection
//   ucp_discover/call      Universal Commerce Protocol: /.well-known/ucp, REST transport
//   webmcp_list_tools/call WebMCP tools a page registers on document|navigator.modelContext
//   browser_accessibility_snapshot / browser_accessibility_click / _fill   accessibility tree
//   web_plan               picks the execution surface: ucp > webmcp > openapi > graphql > dom > visual

type Cfg = { api_hosts?: Record<string, { headers?: Record<string, string> }>; ucp_agent_profile?: string };

const text = (value: unknown) => (typeof value === 'string' ? value : JSON.stringify(value));

// WebMCP shim: installed before any page script runs. It implements the page-facing half of the
// API (registerTool / unregisterTool / provideContext / clearContext) on document.modelContext
// and navigator.modelContext, and records tools so the engine can list and execute them.
const WEBMCP_SHIM = `(() => {
  if (window.__webmcp) return;
  const tools = new Map();
  const ctx = {
    registerTool(tool, options) {
      if (!tool || !tool.name || !tool.description) throw new TypeError('tool needs name and description');
      if (tools.has(tool.name)) throw new Error('duplicate tool name: ' + tool.name);
      tools.set(tool.name, tool);
      if (options && options.signal) options.signal.addEventListener('abort', () => tools.delete(tool.name));
    },
    unregisterTool(name) { tools.delete(name); },
    provideContext(options) { tools.clear(); for (const t of (options && options.tools) || []) ctx.registerTool(t); },
    clearContext() { tools.clear(); },
  };
  const native = (typeof document !== 'undefined' && document.modelContext) || (navigator && navigator.modelContext);
  if (!native) {
    try { Object.defineProperty(document, 'modelContext', { value: ctx, configurable: true }); } catch (e) {}
    try { Object.defineProperty(navigator, 'modelContext', { value: ctx, configurable: true }); } catch (e) {}
  } else {
    // The browser ships WebMCP itself (recent Chrome): keep its behaviour, record what the page registers.
    for (const name of ['registerTool', 'unregisterTool', 'provideContext', 'clearContext']) {
      const original = native[name] && native[name].bind(native);
      if (!original) continue;
      native[name] = (...args) => { const r = original(...args); try { ctx[name](...args); } catch (e) {} return r; };
    }
  }
  window.__webmcp = {
    list: () => [...tools.values()].map(t => ({ name: t.name, description: t.description, inputSchema: t.inputSchema || { type: 'object', properties: {} },
                                                annotations: t.annotations || null })),
    call: async (name, input) => {
      const t = tools.get(name);
      if (!t) throw new Error('no such WebMCP tool: ' + name);
      return await t.execute(input || {}, { requestUserInteraction: async () => { throw new Error('user interaction unavailable'); } });
    },
    native: !!native,
  };
})();`;

export function researchTools(Tool: any, getAgent: () => any, cfg: Cfg): any[] {
  const page = async () => {
    const browser = getAgent().environment.getBrowserManager().getBrowser();
    return browser.getActivePage();
  };
  const allowed = (url: string) => {
    const host = new URL(url).host;
    const entry = cfg.api_hosts?.[host];
    if (!entry) throw new Error(`host ${host} is not an authorized API host (configure tars.api_hosts)`);
    return entry.headers || {};
  };
  const fetchJson = async (url: string, init: any = {}, authorize = true) => {
    const headers = { accept: 'application/json', ...(authorize ? allowed(url) : {}), ...(init.headers || {}) };
    const response = await fetch(url, { ...init, headers, signal: AbortSignal.timeout(20000) });
    const body = await response.text();
    let parsed: unknown = body;
    try { parsed = JSON.parse(body); } catch { /* keep text */ }
    return { status: response.status, ok: response.ok, body: typeof parsed === 'string' ? parsed.slice(0, 20000) : parsed };
  };
  // Public discovery documents (/.well-known/ucp, OpenAPI) are read without credentials.
  const discover = async (url: string) => {
    try { return await fetchJson(url, {}, false); } catch (error: any) { return { status: 0, ok: false, body: String(error?.message || error) }; }
  };

  const tools: any[] = [];
  const add = (id: string, description: string, properties: Record<string, unknown>, required: string[], fn: (a: any) => Promise<unknown>) =>
    tools.push(new Tool({ id, description, parameters: { type: 'object', properties, required }, function: async (a: any) => {
      try { return await fn(a); } catch (error: any) { return { status: 'error', message: String(error?.message || error) }; }
    } }));
  const str = { type: 'string' };
  const obj = { type: 'object' };

  add('api_request', '[api] Authorized REST call to an allowlisted API host. Credentials come from the engine configuration, never from the model.',
    { method: str, url: str, headers: obj, body: obj }, ['method', 'url'],
    async (a) => {
      const method = String(a.method || 'GET').toUpperCase();
      return fetchJson(a.url, { method, headers: { 'content-type': 'application/json', ...(a.headers || {}) },
        ...(a.body && !['GET', 'HEAD'].includes(method) ? { body: JSON.stringify(a.body) } : {}) });
    });

  const operations = (spec: any) => {
    const out: any[] = [];
    for (const [path, item] of Object.entries<any>(spec.paths || {})) {
      for (const method of ['get', 'post', 'put', 'patch', 'delete']) {
        const op = item?.[method];
        if (!op) continue;
        out.push({ operationId: op.operationId || `${method} ${path}`, method: method.toUpperCase(), path, summary: op.summary || '',
          parameters: [...(item.parameters || []), ...(op.parameters || [])].map((p: any) => ({ name: p.name, in: p.in, required: !!p.required })),
          hasBody: !!op.requestBody });
      }
    }
    return out;
  };
  add('openapi_discover', '[api] Read an OpenAPI document and list its operations.', { spec_url: str }, ['spec_url'],
    async (a) => {
      const doc = await discover(a.spec_url);
      if (!doc.ok || typeof doc.body !== 'object') return { status: 'error', message: `no OpenAPI document (HTTP ${doc.status})` };
      const spec: any = doc.body;
      return { status: 'success', title: spec.info?.title, servers: (spec.servers || []).map((s: any) => s.url), operations: operations(spec) };
    });
  add('openapi_call', '[api] Call one operation of an OpenAPI document by operationId.',
    { spec_url: str, operation_id: str, path_params: obj, query: obj, body: obj }, ['spec_url', 'operation_id'],
    async (a) => {
      const doc = await discover(a.spec_url);
      const spec: any = doc.body;
      const op = operations(spec).find((o) => o.operationId === a.operation_id);
      if (!op) return { status: 'error', message: `unknown operation ${a.operation_id}` };
      const base = new URL(spec.servers?.[0]?.url || '/', a.spec_url).toString().replace(/\/$/, '');
      let path = op.path as string;
      for (const [k, v] of Object.entries<any>(a.path_params || {})) path = path.replace(`{${k}}`, encodeURIComponent(String(v)));
      const url = new URL(base + path);
      for (const [k, v] of Object.entries<any>(a.query || {})) url.searchParams.set(k, String(v));
      return fetchJson(url.toString(), { method: op.method, headers: { 'content-type': 'application/json' },
        ...(a.body && op.hasBody ? { body: JSON.stringify(a.body) } : {}) });
    });

  add('graphql_query', '[api] Run a GraphQL query (or introspect the schema) against an authorized endpoint.',
    { endpoint: str, query: str, variables: obj, introspect: { type: 'boolean' } }, ['endpoint'],
    async (a) => {
      const query = a.introspect ? '{ __schema { queryType { name } types { name kind fields { name } } } }' : String(a.query || '');
      const result: any = await fetchJson(a.endpoint, { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ query, variables: a.variables || {} }) });
      return { ...result, http: result.status, status: result.ok && !(result.body as any)?.errors ? 'success' : 'error' };
    });

  const ucpProfile = async (origin: string) => {
    const doc = await discover(new URL('/.well-known/ucp', origin).toString());
    if (!doc.ok || typeof doc.body !== 'object' || !(doc.body as any).ucp) return null;
    return (doc.body as any).ucp;
  };
  add('ucp_discover', '[commerce] Read a merchant\'s Universal Commerce Protocol profile at /.well-known/ucp: services (transports) and capabilities.',
    { origin: str }, ['origin'],
    async (a) => {
      const ucp = await ucpProfile(a.origin);
      if (!ucp) return { status: 'error', message: 'no UCP profile at /.well-known/ucp' };
      const services = Object.entries<any>(ucp.services || {}).flatMap(([name, list]) =>
        (list as any[]).map((s) => ({ service: name, transport: s.transport, endpoint: s.endpoint, version: s.version, schema: s.schema })));
      return { status: 'success', version: ucp.version, services, capabilities: Object.keys(ucp.capabilities || {}),
        payment_handlers: Object.keys(ucp.payment_handlers || {}) };
    });
  add('ucp_call', '[commerce] Call a merchant\'s UCP REST service (e.g. POST /checkout-sessions). The agent profile is sent in the UCP-Agent header.',
    { origin: str, service: str, method: str, path: str, body: obj }, ['origin', 'method', 'path'],
    async (a) => {
      const ucp = await ucpProfile(a.origin);
      if (!ucp) return { status: 'error', message: 'no UCP profile at /.well-known/ucp' };
      const rest = Object.entries<any>(ucp.services || {})
        .filter(([name]) => !a.service || name === a.service)
        .flatMap(([, list]) => list as any[]).find((s) => s.transport === 'rest');
      if (!rest) return { status: 'error', message: 'merchant offers no UCP REST transport' };
      const headers: Record<string, string> = { 'content-type': 'application/json' };
      if (cfg.ucp_agent_profile) headers['UCP-Agent'] = `profile="${cfg.ucp_agent_profile}"`;
      const method = String(a.method).toUpperCase();
      return fetchJson(rest.endpoint.replace(/\/$/, '') + a.path, { method, headers,
        ...(a.body && !['GET', 'HEAD'].includes(method) ? { body: JSON.stringify(a.body) } : {}) }, false);
    });

  add('webmcp_list_tools', '[webmcp] List the WebMCP tools the current page registered (document.modelContext / navigator.modelContext).',
    {}, [],
    async () => {
      const p = await page();
      const list = await p.evaluate('window.__webmcp ? window.__webmcp.list() : null');
      if (list === null) return { status: 'error', message: 'WebMCP shim not installed on this page' };
      return { status: 'success', url: p.url(), tools: list };
    });
  add('webmcp_call_tool', '[webmcp] Execute a WebMCP tool of the current page (the page runs its own function; no clicking).',
    { name: str, input: obj }, ['name'],
    async (a) => {
      const p = await page();
      const result = await p.evaluate(`window.__webmcp.call(${JSON.stringify(a.name)}, ${JSON.stringify(a.input || {})})`);
      return { status: 'success', result };
    });

  // Accessibility tree: what assistive technology (and a model) sees, by role and name.
  add('browser_accessibility_snapshot', '[a11y] The page\'s accessibility tree (role, name, value) - stable across layout changes.',
    {}, [],
    async () => {
      const p = await page();
      const walk = (n: any, depth = 0): string[] => !n ? [] : [`${'  '.repeat(depth)}${n.role}${n.name ? ` "${n.name}"` : ''}${n.value ? ` =${n.value}` : ''}`,
        ...(n.children || []).flatMap((c: any) => walk(c, depth + 1))];
      const snapshot = await (p as any).accessibility.snapshot({ interestingOnly: true });
      return { status: 'success', url: p.url(), tree: walk(snapshot).join('\n') };
    });
  add('browser_accessibility_click', '[a11y] Click an element by accessibility role and name.', { role: str, name: str }, ['role', 'name'],
    async (a) => {
      const p = await page();
      await p.locator(`::-p-aria([role="${a.role}"][name="${a.name}"])`).click();
      return { status: 'success', clicked: `${a.role} "${a.name}"` };
    });
  add('browser_accessibility_fill', '[a11y] Fill a field by accessibility role and name.', { role: str, name: str, value: str }, ['role', 'name', 'value'],
    async (a) => {
      const p = await page();
      await p.locator(`::-p-aria([role="${a.role}"][name="${a.name}"])`).fill(String(a.value));
      return { status: 'success', filled: `${a.role} "${a.name}"` };
    });

  // Dynamic selection: probe what the site offers, in order of preference, and say which to use.
  add('web_plan', '[plan] Decide HOW to execute a web task on a site: probes UCP, WebMCP, OpenAPI, GraphQL, then falls back to DOM or visual. Returns surfaces ranked best-first with evidence.',
    { url: str, goal: str }, ['url'],
    async (a) => {
      const origin = new URL(a.url).origin;
      const surfaces: Array<{ surface: string; evidence: unknown }> = [];
      const ucp = await ucpProfile(origin);
      if (ucp) surfaces.push({ surface: 'ucp', evidence: { version: ucp.version, services: Object.keys(ucp.services || {}), capabilities: Object.keys(ucp.capabilities || {}) } });
      try {
        const p = await page();
        if (p.url() !== a.url) await p.goto(a.url, { waitUntil: 'load' });
        const tools = await p.evaluate('window.__webmcp ? window.__webmcp.list() : []') as any[];
        if (tools.length) surfaces.push({ surface: 'webmcp', evidence: tools.map((t) => t.name) });
      } catch { /* no browser page available: skip */ }
      for (const path of ['/openapi.json', '/swagger.json', '/.well-known/openapi.json']) {
        const doc = await discover(origin + path);
        if (doc.ok && typeof doc.body === 'object' && ((doc.body as any).openapi || (doc.body as any).swagger)) {
          surfaces.push({ surface: 'openapi', evidence: { spec_url: origin + path, operations: operations(doc.body).length } });
          break;
        }
      }
      try {
        const g = await fetchJson(origin + '/graphql', { method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ query: '{ __typename }' }) }, false);
        if (g.ok && (g.body as any)?.data?.__typename) surfaces.push({ surface: 'graphql', evidence: { endpoint: origin + '/graphql' } });
      } catch { /* no graphql */ }
      surfaces.push({ surface: 'dom', evidence: 'browser_get_clickable_elements / browser_click (always available)' });
      surfaces.push({ surface: 'visual', evidence: 'browser_vision_control (last resort: pixels)' });
      return { status: 'success', url: a.url, goal: a.goal || '', recommended: surfaces[0].surface, surfaces };
    });

  return tools;
}

// Install the WebMCP shim on every page, now and for pages created later, before page scripts run.
export async function installWebMcp(agent: any, installed: WeakSet<object>): Promise<void> {
  const manager = agent.environment?.getBrowserManager?.();
  if (!manager?.isLaunchingComplete?.()) return;
  const browser = manager.getBrowser().getBrowser();
  if (installed.has(browser)) return;
  installed.add(browser);
  const attach = async (page: any) => {
    try {
      await page.evaluateOnNewDocument(WEBMCP_SHIM);
      await page.evaluate(WEBMCP_SHIM);
    } catch { /* page closing */ }
  };
  for (const page of await browser.pages()) await attach(page);
  browser.on('targetcreated', async (target: any) => {
    if (target.type() === 'page') { const page = await target.page(); if (page) await attach(page); }
  });
}
