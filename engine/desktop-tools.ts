// Native desktop GUI tools for the Agent TARS engine.
//
// Agent TARS's core ships browser (DOM + vision) tools; UI-TARS ships the desktop operator
// (`@ui-tars/operator-nut-js`, driven by `@ui-tars/action-parser` actions). This file
// registers that same operator as native TARS tools, the way `browser_vision_control`
// exposes the browser operator: the model writes a UI-TARS action string, the operator
// executes it on the real screen. No second agent loop, no other automation library.
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

export function desktopAvailable(): boolean {
  if (process.platform === 'linux' && !process.env.DISPLAY) return false;
  try {
    require.resolve('@ui-tars/operator-nut-js');
    return true;
  } catch {
    return false;
  }
}

export function desktopTools(Tool: any): any[] {
  const { NutJSOperator } = require('@ui-tars/operator-nut-js');
  const { actionParser } = require('@ui-tars/action-parser');
  const operator = new NutJSOperator();
  const images = resolve(here, 'images');

  const shoot = async () => {
    const shot = await operator.screenshot();
    return shot as { base64: string; scaleFactor: number };
  };

  const screenshot = new Tool({
    id: 'desktop_screenshot',
    description: '[desktop] Take a screenshot of the whole desktop (the real screen, any application).',
    parameters: { type: 'object', properties: {} },
    function: async () => {
      const shot = await shoot();
      mkdirSync(images, { recursive: true });
      const slug = `desktop-${new Date().toISOString().replace(/[:.]/g, '-')}`;
      const filepath = resolve(images, `${slug}.png`);
      writeFileSync(filepath, Buffer.from(shot.base64, 'base64'));
      return { status: 'success', filepath, slug, scaleFactor: shot.scaleFactor };
    },
  });

  const control = new Tool({
    id: 'desktop_vision_control',
    description:
      '[desktop] Control any desktop application visually with one UI-TARS action. ' +
      "Coordinates are on a 0-1000 grid over the screen. Action space: click(start_box='(x,y)'), " +
      "left_double(start_box='(x,y)'), right_single(start_box='(x,y)'), drag(start_box='(x,y)', end_box='(x,y)'), " +
      "hotkey(key='ctrl c'), type(content='text'), scroll(start_box='(x,y)', direction='down'), wait().",
    parameters: {
      type: 'object',
      properties: {
        thought: { type: 'string', description: 'What you see and the next step, one sentence.' },
        action: { type: 'string', description: "One UI-TARS action, e.g. click(start_box='(500,400)')" },
      },
      required: ['action'],
    },
    function: async (args: { thought?: string; action: string }) => {
      const shot = await shoot();
      const png = Buffer.from(shot.base64, 'base64');
      const width = png.readUInt32BE(16);
      const height = png.readUInt32BE(20);
      const { parsed } = actionParser({
        prediction: `Thought: ${args.thought || ''}\nAction: ${args.action}`,
        factor: [1000, 1000],
        screenContext: { width, height },
        scaleFactor: shot.scaleFactor,
      });
      if (!parsed.length || !parsed[0].action_type) return { status: 'error', message: `unparseable action: ${args.action}` };
      const result = await operator.execute({
        parsedPrediction: parsed[0], screenWidth: width, screenHeight: height, scaleFactor: shot.scaleFactor,
      });
      return { status: 'success', action: parsed[0].action_type, inputs: parsed[0].action_inputs, result: result ?? null };
    },
  });
  return [screenshot, control];
}
