// Unauthenticated route probe: proves a route is mounted (401) rather than
// absent (404) without needing anyone's persona key.
// Run: node scripts/routes.mjs [base]
const BASE = (process.argv[2] || "http://127.0.0.1:8800").replace(/\/+$/, "");
const paths = [
  "GET /v1/courts",
  "GET /v1/hands",
  "GET /v1/computer",
  "GET /v1/link",
  "GET /v1/voice-status",
  "GET /v1/calls",
  "GET /v1/computer/windows",
  "GET /v1/computer/capture",
  "GET /v1/timeline",
  "GET /v1/models",
];
for (const line of paths) {
  const [method, path] = line.split(" ");
  const t = Date.now();
  try {
    const res = await fetch(BASE + path, { method, signal: AbortSignal.timeout(8000) });
    const body = (await res.text()).slice(0, 90);
    const verdict = res.status === 401 ? "mounted, key required" : res.status === 404 ? "NOT MOUNTED" : `HTTP ${res.status}`;
    console.log(`${method.padEnd(5)} ${path.padEnd(26)} ${verdict.padEnd(22)} ${Date.now() - t}ms  ${body}`);
  } catch (err) {
    console.log(`${method.padEnd(5)} ${path.padEnd(26)} ${err.name}: ${err.message}`);
  }
}