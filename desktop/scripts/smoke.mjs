// Runtime smoke test for the Hands desktop app's data path.
//
// It re-implements, line for line, what src/lib/connection.ts + src/lib/client.ts
// send: same URL building, same bearer header, same key-in-query for the socket,
// same route order — then drives a real backend the same way the exe does.
// Run:  node scripts/smoke.mjs --base http://127.0.0.1:8800 --key sp_...
import { argv } from "node:process";

function arg(name, fallback = "") {
  const i = argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const next = argv[i + 1];
  return next && !next.startsWith("--") ? next : fallback;
}

const BASE = (arg("base", "http://127.0.0.1:8800") || "").replace(/\/+$/, "");
const KEY = arg("key", "");

let pass = 0;
let fail = 0;

function ok(label, detail = "") {
  pass += 1;
  console.log(`  PASS  ${label}${detail ? `  ${detail}` : ""}`);
}
function bad(label, detail = "") {
  fail += 1;
  console.log(`  FAIL  ${label}${detail ? `  ${detail}` : ""}`);
}

function httpUrl(path, query) {
  const u = new URL(path, BASE);
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== undefined && v !== "") u.searchParams.set(k, String(v));
  }
  return u.toString();
}

function keyedWsUrl(path, query) {
  const u = new URL(httpUrl(path, { ...(query || {}), ...(KEY ? { key: KEY } : {}) }));
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  return u.toString();
}

async function call(path, init, opts = {}) {
  const keyless = opts.keyless === true;
  const res = await fetch(httpUrl(path), {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(!keyless && KEY ? { authorization: `Bearer ${KEY}` } : {}),
    },
  });
  const text = await res.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text.slice(0, 200) };
    }
  }
  return { status: res.status, ok: res.ok, data };
}

async function main() {
  console.log(`backend: ${BASE}`);
  console.log(`key:     ${KEY ? `${KEY.slice(0, 7)}…(${KEY.length} chars)` : "(none)"}`);

  if (!KEY) {
    console.log("  SKIP  no --key given; nothing further is reachable");
    process.exit(0);
  }
  // Auth is enforced before anything else, exactly as the app relies on it.
  const noKey = await call("/v1/courts", undefined, { keyless: true }).catch(() => null);
  if (noKey && noKey.status === 401) ok("GET /v1/courts without a key is refused", "HTTP 401");
  else if (!noKey) bad("GET /v1/courts without a key is refused", "no answer");
  else bad("GET /v1/courts without a key is refused", `HTTP ${noKey.status}`);

  // --- health probes (what the footer status bar reads) ---
  for (const [label, path] of [
    ["GET /v1/models", "/v1/models"],
    ["GET /v1/link", "/v1/link"],
    ["GET /v1/computer", "/v1/computer"],
    ["GET /v1/voice-status", "/v1/voice-status"],
    ["GET /v1/calls", "/v1/calls"],
    ["GET /v1/hands", "/v1/hands"],
    ["GET /v1/computer/capture", "/v1/computer/capture"],
  ]) {
    const r = await call(path).catch((e) => ({ status: 0, data: { err: e.message } }));
    if (r.ok) ok(label, `HTTP ${r.status} ${JSON.stringify(r.data).slice(0, 90)}`);
    else bad(label, `HTTP ${r.status} ${JSON.stringify(r.data).slice(0, 90)}`);
  }

  // --- the deck, exactly as DeckView + useDeck drive it ---
  const empty = await call("/v1/courts");
  if (!empty.ok) {
    bad("GET /v1/courts", `HTTP ${empty.status} ${JSON.stringify(empty.data).slice(0, 120)}`);
    console.log("\ncourt routes are not mounted (SAWII_COURTS?). Aborting deck checks.");
    process.exit(1);
  }
  ok("GET /v1/courts", `count=${empty.data.count} running=${empty.data.running}`);

  const before = empty.data.count;
  const made = [];
  for (const title of ["smoke A", "smoke B"]) {
    const r = await call("/v1/courts", { method: "POST", body: JSON.stringify({ title, kind: "desktop" }) });
    if (r.status !== 201 || !r.data.id) {
      bad(`POST /v1/courts (${title})`, `HTTP ${r.status}`);
      continue;
    }
    made.push(r.data);
    ok(`POST /v1/courts (${title})`, r.data.id);
  }
  if (made.length !== 2) {
    console.log("\nneed two cards to prove concurrency");
    process.exit(1);
  }

  const afterCreate = await call("/v1/courts");
  if (afterCreate.data.count === before + 2) ok("deck grew by 2", `count=${afterCreate.data.count}`);
  else bad("deck grew by 2", `expected ${before + 2}, got ${afterCreate.data.count}`);

  // Concurrency: two spawns must produce two distinct lanes.
  const spawned = [];
  for (const c of made) {
    const r = await call(`/v1/courts/${c.id}/spawn`, { method: "POST", body: JSON.stringify({ goal: "smoke" }) });
    if (r.status === 202) {
      spawned.push(r.data);
      ok(`POST /v1/courts/${c.id}/spawn`, `status=${r.data.status} lane=${r.data.lane || "-"} link=${r.data.link || "-"}`);
    } else bad(`spawn ${c.id}`, `HTTP ${r.status} ${JSON.stringify(r.data).slice(0, 120)}`);
  }
  const lanes = new Set(spawned.map((c) => c.lane).filter(Boolean));
  if (lanes.size === 2) ok("two courts on two distinct lanes", [...lanes].join(", "));
  else if (lanes.size === 0) bad("two distinct lanes", "no lane assigned: no Link registered, card is surface-only");
  else bad("two distinct lanes", `lanes=${[...lanes].join(",") || "none"}`);

  // Steer, then the voice hook, then cancel, then delete.
  const goal = await call(`/v1/courts/${made[0].id}/goal`, {
    method: "POST",
    body: JSON.stringify({ goal: "smoke steer" }),
  });
  if (goal.ok && goal.data.goal === "smoke steer") ok(`POST …/${made[0].id}/goal`, "goal recorded");
  else bad("goal", `HTTP ${goal.status} ${JSON.stringify(goal.data).slice(0, 120)}`);

  const voice = await call(`/v1/courts/${made[0].id}/voice`, {
    method: "POST",
    body: JSON.stringify({ action: "status" }),
  });
  if (voice.ok && voice.data.websocket) ok(`POST …/${made[0].id}/voice`, voice.data.websocket);
  else bad("voice hook", `HTTP ${voice.status}`);

  const cancel = await call(`/v1/courts/${made[1].id}/cancel`, { method: "POST", body: "{}" });
  if (cancel.ok && cancel.data.status === "stopped") ok(`POST …/${made[1].id}/cancel`, "stopped, card kept");
  else bad("cancel", `HTTP ${cancel.status} ${JSON.stringify(cancel.data).slice(0, 120)}`);

  // The event socket the grid actually listens to.
  const wsOk = await new Promise((resolve) => {
    let sock;
    try {
      sock = new WebSocket(keyedWsUrl("/v1/court-events"));
    } catch (e) {
      resolve(`ctor: ${e.message}`);
      return;
    }
    const timer = setTimeout(() => {
      try {
        sock.close();
      } catch {
        /* noop */
      }
      resolve("timeout");
    }, 8000);
    sock.onmessage = (ev) => {
      clearTimeout(timer);
      let d = null;
      try {
        d = JSON.parse(String(ev.data));
      } catch {
        /* ignore */
      }
      try {
        sock.close();
      } catch {
        /* noop */
      }
      resolve(d ? `${d.kind} count=${d.count ?? "-"} courts=${Array.isArray(d.courts) ? d.courts.length : "-"}` : "unparsable");
    };
    sock.onerror = () => {
      clearTimeout(timer);
      resolve("socket error");
    };
  });
  if (wsOk.startsWith("snapshot")) ok("WS /v1/court-events", wsOk);
  else bad("WS /v1/court-events", wsOk);

  const del = await call(`/v1/courts/${made[1].id}`, { method: "DELETE" });
  if (del.ok && del.data.ok === true) ok(`DELETE /v1/courts/${made[1].id}`, "ok");
  else bad("delete", `HTTP ${del.status}`);

  const finalSnap = await call("/v1/courts");
  ok("final deck state", `count=${finalSnap.data.count} running=${finalSnap.data.running}`);
  console.log("\nfinal courts as the app sees them:");
  console.log(JSON.stringify(finalSnap.data, null, 2).slice(0, 1600));

  console.log(`\nsmoke: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}

void main();