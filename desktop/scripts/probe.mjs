// Quick reachability probe: node scripts/probe.mjs [port ...]
const ports = process.argv.slice(2).length ? process.argv.slice(2) : ["8800", "8899"];
for (const p of ports) {
  const base = `http://127.0.0.1:${p}`;
  const t = Date.now();
  try {
    const res = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(4000) });
    const body = await res.text();
    console.log(`${base} -> HTTP ${res.status} in ${Date.now() - t}ms ${body.slice(0, 120)}`);
  } catch (err) {
    console.log(`${base} -> ${err.name}: ${err.message} (${Date.now() - t}ms)`);
  }
}