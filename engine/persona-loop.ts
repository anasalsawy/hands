import { createInterface } from 'node:readline';
import { LogLevel, NativeToolCallEngine, Tool } from '@tarko/agent';
import ComposableAgentModule from './ui-tars-desktop/multimodal/omni-tars/core/src/ComposableAgent.ts';
import AgentPluginModule from './ui-tars-desktop/multimodal/omni-tars/core/src/AgentPlugin.ts';
import type { LLMRequestHookPayload } from '@tarko/agent';

const { ComposableAgent } = ComposableAgentModule;
const { AgentPlugin } = AgentPluginModule;

// stdout is exclusively the private JSON-lines transport, never model logs.
console.log = console.info = console.debug = console.warn = () => {};
const send = (message: unknown) => process.stdout.write(JSON.stringify(message) + '\n');
const pending = new Map<string, { resolve: (value: any) => void; reject: (e: Error) => void }>();
let sequence = 0;
let started = false;

function request(type: string, payload: Record<string, unknown> = {}): Promise<any> {
  const id = String(++sequence);
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    send({ type, id, ...payload });
  });
}

class PersonaHands extends AgentPlugin {
  readonly name = 'persona-hands';
  private observation: any;

  constructor(definitions: any[]) {
    super();
    this.tools = definitions.map(({ function: tool }) => new Tool({
      id: tool.name,
      description: tool.description || '',
      parameters: tool.parameters || { type: 'object', properties: {} },
      function: (args: unknown) => request('tool', { name: tool.name, args }),
    }));
  }

  async onEachAgentLoopStart(): Promise<void> {
    this.observation = await request('observe');
  }

  async onLLMRequest(_id: string, payload: LLMRequestHookPayload): Promise<void> {
    const observation = this.observation;
    if (!observation) return;
    const content: any[] = [{ type: 'text', text: `Current lane observation:\n${observation.text || ''}` }];
    if (observation.image_url) {
      content.push({ type: 'image_url', image_url: { url: observation.image_url } });
    }
    payload.request.messages.push({ role: 'user', content });
  }
}

async function run(config: any) {
  try {
    const hands = new PersonaHands(config.tools);
    const agent = new ComposableAgent({
      name: 'Persona R / Omni-TARS',
      // This is a routing label, not a provider. All inference uses the injected client below.
      model: { provider: 'openai', id: 'persona-r', baseURL: 'http://127.0.0.1:1' },
      instructions: config.instructions,
      plugins: [hands],
      toolCallEngine: NativeToolCallEngine,
      maxIterations: config.max_steps + 1,
      logLevel: LogLevel.SILENT,
      context: { maxImagesCount: 1 },
    });
    agent.setCustomLLMClient({
      chat: { completions: { create: async (options: any) => {
        // Injected clients bypass Tarko's provider-request hook.
        await hands.onLLMRequest('', { provider: 'persona', request: options });
        const result = await request('think', { messages: options.messages });
        const toolCalls = (result.tool_calls || []).map((tool: any, index: number) => ({
          index, id: tool.id, type: 'function',
          function: { name: tool.name, arguments: JSON.stringify(tool.arguments || {}) },
        }));
        // Tarko consumes a streaming client even when run(stream: false) is used.
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
    const result = await agent.run({ input: config.goal, stream: false });
    if (result.finishReason === 'max_iterations') throw new Error('lane step limit reached');
    if (['error', 'abort'].includes(result.finishReason) || !result.content?.trim()) {
      throw new Error('agent ended without a completed answer');
    }
    send({ type: 'done', text: result.content });
    process.exitCode = 0;
  } catch (error) {
    send({ type: 'error', error: error instanceof Error ? error.message : String(error) });
    process.exitCode = 1;
  } finally {
    input.close();
    process.stdin.destroy();
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
  for (const callback of pending.values()) callback.reject(new Error('Persona disconnected'));
  pending.clear();
});
send({ type: 'ready' });
