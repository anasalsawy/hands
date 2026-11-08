// Environment report for the desktop build. Cheap: no WMI, no recursion.
// Run: node scripts/env.mjs
import { execFileSync } from "node:child_process";
import { statfs } from "node:fs/promises";

function ver(cmd, args) {
  try {
    return execFileSync(cmd, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], shell: true }).trim().split("\n")[0];
  } catch (err) {
    return `MISSING (${String(err.message).split("\n")[0].slice(0, 60)})`;
  }
}

console.log("node      ", ver("node", ["--version"]));
console.log("npm       ", ver("npm", ["--version"]));
console.log("cargo     ", ver("cargo", ["--version"]));
console.log("rustc     ", ver("rustc", ["--version"]));
console.log("rustup    ", ver("rustup", ["show", "active-version"]));
console.log("link.exe  ", ver("where", ["link.exe"]));
console.log("cl.exe    ", ver("where", ["cl.exe"]));
console.log("cmake     ", ver("cmake", ["--version"]));
console.log("vswhere   ", ver('"C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"', ["-products", "*", "-property", "installationPath"]));

const df = await statfs(process.env.SystemDrive || "C:\\");
console.log(
  "disk C    ",
  `${Math.round((Number(df.bsize) * Number(df.bavail)) / 1024 / 1024 / 1024)} GB free of ${Math.round((Number(df.bsize) * Number(df.blocks)) / 1024 / 1024 / 1024)} GB`,
);