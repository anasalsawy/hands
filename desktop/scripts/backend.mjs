// Starts the Hands backend for the desktop app, in local or remote mode.
//
//   node scripts/backend.mjs                      local: python stitch.py local --port 8800
//   node scripts/backend.mjs --root C:\work       same, with a folder the engine may use
//   node scripts/backend.mjs --remote https://host --key sp_...   just prints what to type
//
// Nothing here is served to the UI. The exe talks to whatever URL it is given;
// this script only exists so "start the backend" is one command.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const desktop = resolve(here, "..");
const handsRoot = resolve(desktop, "..");
const stitch = join(handsRoot, "stitch.py");

function arg(name, fallback = "") {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const next = process.argv[i + 1];
  return next && !next.startsWith("--") ? next : fallback;
}

const port = arg("port", process.env.HANDS_PORT || "8800");
const root = arg("root", process.env.HANDS_ROOT || "");
const remote = arg("remote");
const key = arg("key", process.env.SAWII_PERSONA_KEY || "");
const extra = [];

const py = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");

if (remote) {
  console.log("Remote mode — this app does not start a backend.");
  console.log("Open the exe, press Settings, and enter:");
  console.log(`  URL:  ${remote}`);
  console.log(`  Key:  ${key ? key.slice(0, 7) + "…" : "(your sp_… persona key)"}`);
  process.exit(0);
}

if (!existsSync(stitch)) {
  console.error(`stitch.py not where it should be: ${stitch}`);
  console.error("Run this from a checkout, or pass --remote <url> --key <sp_…> instead.");
  process.exit(1);
}

const cmd = [stitch, "local", "--port", String(port)];
if (root) cmd.push("--root", root);
if (process.env.HANDS_APPROVE) cmd.push("--approve", process.env.HANDS_APPROVE);
if (process.env.HANDS_DATA_DIR) cmd.push("--data-dir", process.env.HANDS_DATA_DIR);
if (key) cmd.push("--persona-key", key);
extra.forEach((a) => cmd.push(a));

console.log(`$ ${py} ${cmd.join(" ")}`);
console.log(`backend will listen on http://127.0.0.1:${port}`);
console.log("the persona key it prints is what the app's Settings screen wants");

const child = spawn(py, cmd, { cwd: handsRoot, stdio: "inherit", env: process.env });
child.on("exit", (code) => process.exit(code ?? 1));
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    child.kill(sig);
  });
}