// Sequential build: icons -> typecheck -> front-end bundle -> Rust deps -> exe.
// One script because this machine is memory-tight; running these in parallel
// makes node or rustc fail to allocate.
import { spawnSync } from "node:child_process";

const steps = [
  ["icons", "npx", ["tauri", "icon", "src/assets/app-icon.png"]],
  ["typecheck", "npx", ["tsc", "--noEmit"]],
  ["bundle", "npm", ["run", "build"]],
  ["crates", "cargo", ["fetch", "--manifest-path", "src-tauri/Cargo.toml"]],
  ["exe", "npx", ["tauri", "build"]],
];

const only = process.argv[2] || "all";
let failed = false;

for (const [name, cmd, args] of steps) {
  if (only !== "all" && only !== name) continue;
  console.log(`\n=== ${name}: ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", shell: true });
  console.log(`=== ${name}: exit ${r.status}`);
  if (r.status !== 0) {
    failed = true;
    break;
  }
}
process.exit(failed ? 1 : 0);