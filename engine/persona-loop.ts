// Persona (the mind) drives Agent TARS (the execution engine).
//
// Boundary (owner ruling): Persona owns reasoning, planning, decomposition,
// verification, memory, approvals, lanes and the live cards. Agent TARS owns
// ALL execution: browser (DOM / visual / hybrid), filesystem, terminal, search,
// MCP. Nothing here executes a tool on Persona's behalf; the only thing this
// file injects is (a) Persona's model as the LLM client, (b) Persona's
// observation, (c) an approval + receipt hook around each native TARS tool.
//
// Private JSON-lines transport on stdio (no network, no provider credentials):
//   child -> Persona: ready | catalog | observe | think | approve | receipt | event | tool | done | error
//   Persona -> child: start | {id, result|error}
import { createInterface } from 'node:readline';
import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
// The real Agent TARS core, built from the vendored upstream tree (npm run build:tars).
const coreDir = resolve(here, 'ui-tars-desktop/multimodal/agent-tars/core');
const { AgentTARS } = require(resolve(coreDir, 'dist/index.js'));
// Tool class from the SAME @tarko/agent copy the TARS core runs on.
const coreRequire = createRequire(resolve(coreDir, 'package.json'));
const { Tool, zodToJsonSchema: tarkoSchema } = coreRequire('@tarko/agent');
const { zodToJsonSchema: zodSchema } = coreRequire('zod-to-json-schema');
// Tools from other zod copies are not recognized by tarko's converter; fall back to the zod one.
const toSchema = (schema: any) => {
  const out: any = tarkoSchema(schema);
  return out && out._def ? zodSchema(schema) : out;
};

import { desktopAvailable, desktopTools } from './desktop-tools.ts';

// stdout is exclusively the private JSON-lines transport, never model logs.
const realWrite = process.stdout.write.bind(process.stdout);
process.stdout.write = ((chunk: any, ...rest: any[]) => process.stderr.write(chunk, ...rest)) as any;
console.log = console.info = console.debug = console.warn = (...a: unknown[]) => { process.stderr.write(a.join(' ') + '\n'); };
const send = (message: unknown) => realWrite(JSON.stringify(message) + '\n');
const pending = new Map<string, { resolve: (value: any) => void; reject: (e: Error) => void }>();
let sequence = 0;
let started = false;
let toolsMode = false;
let engineAgent: any;
let onClose: () => void = () => {};

function request(type: string, payload: Record<string, unknown> = {}): Promise<any> {
  const id = String(++sequence);
  return new Promise((resolvePromise, reject) => {
    pending.set(id, { resolve: resolvePromise, reject });
    send({ type, id, ...payload });
  });
}

// Tools that change the world. Persona's approval gate applies to these and
// mutations on one surface are serialized (separate lanes stay concurrent).
const MUTATING = new Set([
  'run_command', 'run_script', 'write_file', 'edit_file', 'move_file', 'create_directory',
  'browser_click', 'browser_form_input_fill', 'browser_select', 'browser_press_key',
  'browser_navigate', 'browser_evaluate', 'browser_vision_control', 'browser_new_tab',
  'desktop_vision_control', 'browser_close_tab', 'browser_switch_tab', 'browser_scroll', 'browser_go_back', 'browser_go_forward',
]);

let personaOnly = new Set<string>();
let chain: Promise<unknown> = Promise.resolve();
// One browser surface, one tool at a time: TARS's DOM mapping is global state, so a screenshot
// for the live pane must never run in the middle of the loop's own click or element read.
let browserChain: Promise<unknown> = Promise.resolve();
function onBrowser<T>(name: string, run: () => Promise<T>): Promise<T> {
  if (!name.startsWith('browser_')) return run();
  const mine = browserChain.then(run, run);
  browserChain = mine.catch(() => undefined);
  return mine;
}
const wrapped = new WeakSet<object>();
const originals = new Map<string, (args: any) => any>();

// Wrap a NATIVE TARS tool: Persona approves first, TARS executes it itself,
// Persona receives the receipt (speculation / verification).
function wrapNative(tool: any): any {
  if (!tool || wrapped.has(tool) || personaOnly.has(tool.name)) return tool;
  wrapped.add(tool);
  const original = tool.function.bind(tool);
  originals.set(tool.name, original);
  const mutating = MUTATING.has(tool.name);
  tool.function = async (args: any) => {
    const gate = await request('approve', { name: tool.name, args, mutating });
    if (!gate || gate.allow === false) {
      return JSON.stringify({ error: `denied: ${gate?.reason || 'not approved'}` });
    }
    const exec = () => onBrowser(tool.name, async () => asText(await Promise.resolve(original(args)).catch((e: any) => ({ error: String(e?.message || e) }))));
    let result: string;
    if (mutating) {
      const mine = chain.then(exec, exec);
      chain = mine.catch(() => undefined);
      result = await mine;
    } else {
      result = await exec();
    }
    return await request('receipt', { name: tool.name, args, result });
  };
  return tool;
}

class PersonaTARS extends AgentTARS {
  observation: any;

  // Every tool TARS registers, at any time (including lazily), passes the gate.
  registerTool(tool: any): void {
    super.registerTool(wrapNative(tool));
  }

  // TARS's pre/post tool hooks (lazy launch, liveness probe, recovery) touch the page too, so
  // they queue on the same browser lock as every tool.
  async onBeforeToolCall(id: string, toolCall: any, args: any): Promise<any> {
    return onBrowser(toolCall.name, () => super.onBeforeToolCall(id, toolCall, args));
  }

  async onAfterToolCall(id: string, toolCall: any, result: any): Promise<any> {
    return onBrowser(toolCall.name, () => super.onAfterToolCall(id, toolCall, result));
  }

  async onEachAgentLoopStart(sessionId: string): Promise<void> {
    this.observation = await request('observe');
    await super.onEachAgentLoopStart?.(sessionId);
  }

  async onLLMRequest(id: string, payload: any): Promise<void> {
    await super.onLLMRequest?.(id, payload);
    const observation = this.observation;
    if (!observation) return;
    const content: any[] = [{ type: 'text', text: `Current lane observation:\n${observation.text || ''}` }];
    if (observation.image_url) content.push({ type: 'image_url', image_url: { url: observation.image_url } });
    payload.request.messages.push({ role: 'user', content });
  }
}

function asText(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value ?? null);
}

let driving: Promise<void> | null = null;

// Boot the real Agent TARS core. In loop mode `drive` follows immediately; in tools mode the
// engine stays up and serves `call` / `run` requests for Persona.
async function boot(config: any) {
  personaOnly = new Set((config.tools || []).map((t: any) => (t.function || t).name));
  const agent: any = new PersonaTARS({
    name: 'Persona R / Agent TARS',
    // Routing label only: all inference uses the injected Persona client below. 'volcengine'
    // is the label Agent TARS requires to enable its vision (visual/hybrid) browser control,
    // which is keyed to the provider name, not to a real endpoint (documented deviation D1).
    model: { provider: config.tars?.provider_label || 'volcengine', id: 'persona-r', baseURL: 'http://127.0.0.1:1', apiKey: 'unused' },
    instructions: config.instructions,
    workspace: config.tars?.workspace,
    browser: {
      control: config.tars?.browser_control || 'hybrid',
      headless: config.tars?.headless ?? true,
      cdpEndpoint: config.tars?.cdp_endpoint,
      executablePath: config.tars?.chrome_path,
    },
    search: config.tars?.search,
    mcpServers: config.tars?.mcp_servers,
    // Persona emits structured tool calls; never TARS's text-parsing engines.
    toolCallEngine: 'native',
    maxIterations: (config.max_steps || 20) + 1,
    logLevel: 'silent',
    context: { maxImagesCount: 1 },
  } as any);
  registerPersonaTools(agent, config.tools || []);
  await agent.initialize();
  // Desktop GUI (UI-TARS nut-js operator) as native TARS tools, when a screen is available.
  if (config.tars?.desktop !== false && desktopAvailable()) {
    for (const tool of desktopTools(Tool)) agent.registerTool(tool);
  }
  engineAgent = agent;
  // TARS's own event stream, forwarded to Persona for the live cards.
  agent.getEventStream().subscribe((event: any) => {
    try {
      send({ type: 'event', kind: event.type, name: event.name, id: event.toolCallId, ts: event.timestamp });
    } catch { /* a card must never break a lane */ }
  });
  agent.setCustomLLMClient({
    chat: { completions: { create: async (options: any) => {
      await agent.onLLMRequest('', { provider: 'persona', request: options });
      const result = await request('think', { messages: options.messages, tools: options.tools });
      const toolCalls = (result.tool_calls || []).map((tool: any, index: number) => ({
        index, id: tool.id, type: 'function',
        function: { name: tool.name, arguments: JSON.stringify(tool.arguments || {}) },
      }));
      return (async function* () {
        yield {
          id: 'persona-r', object: 'chat.completion.chunk', model: 'persona-r',
          created: Math.floor(Date.now() / 1000),
          choices: [{ index: 0, delta: { role: 'assistant', content: result.text || '',
            ...(toolCalls.length ? { tool_calls: toolCalls } : {}) },
            finish_reason: toolCalls.length ? 'tool_calls' : 'stop' }],
        };
      })();
    } } },
  } as any);
  return agent;
}

function registerPersonaTools(agent: any, tools: any[]) {
  // Persona-only tools (e.g. the planning tool) run on Persona's side.
  for (const definition of tools) {
    const spec = definition.function || definition;
    personaOnly.add(spec.name);
    agent.registerTool(new Tool({
      id: spec.name,
      description: spec.description || '',
      parameters: spec.parameters || { type: 'object', properties: {} },
      function: (args: unknown) => request('tool', { name: spec.name, args }),
    }));
  }
}

function nativeNames(agent: any) {
  return (agent.getTools() as any[]).filter((t) => !personaOnly.has(t.name));
}

// One run of the real TARS agent loop with Persona as its mind.
async function drive(agent: any, config: any) {
  const result = await agent.run({ input: config.goal, stream: false });
  if (result.finishReason === 'max_iterations') throw new Error('lane step limit reached');
  if (['error', 'abort'].includes(result.finishReason) || !result.content?.trim()) {
    throw new Error('agent ended without a completed answer');
  }
  send({ type: 'done', text: result.content });
}

async function run(config: any) {
  let agent: any;
  try {
    agent = await boot(config);
    const natives = nativeNames(agent);
    if (config.mode === 'tools') {
      // Tools mode: TARS is booted (browser, filesystem, commands, MCP, search) and serves
      // Persona's `call` requests with its native tools; `run` starts the TARS loop on demand.
      send({ type: 'catalog', tools: natives.map((t) => t.name), specs: natives.map((t) => ({
        name: t.name, description: t.description || '', parameters: toSchema(t.schema),
      })) });
      toolsMode = true;
      await new Promise<void>((done) => { onClose = done; });
      return;
    }
    send({ type: 'catalog', tools: natives.map((t) => t.name) });
    await drive(agent, config);
    process.exitCode = 0;
  } catch (error) {
    send({ type: 'error', error: error instanceof Error ? error.message : String(error) });
    process.exitCode = 1;
  } finally {
    try { await agent?.cleanup?.(); } catch { /* best effort */ }
    input.close();
    process.stdin.destroy();
    setTimeout(() => process.exit(process.exitCode ?? 0), 500).unref();
  }
}

const input = createInterface({ input: process.stdin });
input.on('line', (line) => {
  try {
    const message = JSON.parse(line);
    if (message.type === 'start' && !started) {
      started = true;
      void run(message);
      return;
    }
    if (message.type === 'abort') {
      try { engineAgent?.abort?.(); } catch { /* nothing running */ }
      return;
    }
    if (message.type === 'run' && toolsMode && engineAgent && !driving) {
      // A TARS loop on the persistent engine: same browser, same session, Persona as the mind.
      registerPersonaTools(engineAgent, message.tools || []);
      driving = drive(engineAgent, message)
        .catch((e: any) => send({ type: 'error', error: e instanceof Error ? e.message : String(e) }))
        .finally(() => { driving = null; });
      return;
    }
    if (message.type === 'call') {
      // Persona asks the engine to run one of TARS's own read-only tools (e.g. a
      // screenshot for the live card). TARS executes it; Persona only reads the result.
      const fn = originals.get(message.name);
      // Same path TARS takes for its own calls: pre-hook (lazy browser launch, workspace paths),
      // the native tool, post-hook (browser state).
      const nativeCall = async () => {
        const call = { toolCallId: `persona-${message.id}`, name: message.name };
        const args = (await engineAgent.onBeforeToolCall(call.toolCallId, call, message.args || {})) ?? message.args ?? {};
        const result = await onBrowser(message.name, async () => fn!(args));
        return (await engineAgent.onAfterToolCall(call.toolCallId, call, result)) ?? result;
      };
      Promise.resolve(fn && engineAgent ? nativeCall() : { error: `unknown tool ${message.name}` })
        .then((r) => send({ type: 'result', id: message.id, result: asText(r) }))
        .catch((e) => send({ type: 'result', id: message.id, result: asText({ error: String(e?.message || e) }) }));
      return;
    }
    const callback = pending.get(message.id);
    if (!callback) throw new Error('unrecognized protocol response');
    pending.delete(message.id);
    if (message.error) callback.reject(new Error(message.error));
    else callback.resolve(message.result);
  } catch {
    send({ type: 'error', error: 'invalid loop protocol message' });
    process.exit(1);
  }
});
input.on('close', () => {
  onClose();
  for (const callback of pending.values()) callback.reject(new Error('Persona disconnected'));
  pending.clear();
});
send({ type: 'ready' });
