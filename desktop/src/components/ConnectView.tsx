// First-run connection. Hands is a standalone exe: it needs a backend URL and a
// persona key. Both live in this webview's own storage. Nothing here is written
// to a tracked file, and the key field is never echoed back.
import { useState } from "react";
import { apiBase, personaKey, setApiBase, setPersonaKey } from "../lib/connection";
import { httpUrl } from "../lib/connection";

export function ConnectView({ onDone }: { onDone: () => void }) {
  const [base, setBase] = useState(apiBase());
  const [key, setKey] = useState(personaKey());
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  const test = async () => {
    setBusy(true);
    setNote("");
    const clean = base.trim().replace(/\/+$/, "");
    try {
      const res = await fetch(`${clean}/v1/models`, {
        headers: key.trim() ? { authorization: `Bearer ${key.trim()}` } : {},
      });
      setNote(res.ok ? "reachable · GET /v1/models answered" : `replied HTTP ${res.status}`);
    } catch (err) {
      setNote(`unreachable: ${err instanceof Error ? err.message : "network"}`);
    } finally {
      setBusy(false);
    }
  };

  const save = () => {
    setApiBase(base.trim());
    setPersonaKey(key.trim());
    onDone();
  };

  return (
    <div className="empty">
      <h2>Connect to a Persona backend</h2>
      <p className="hint">
        {"Local mode: run "}
        <code>{"python stitch.py local --root DIR --port 8800"}</code>
        {" from the hands folder and use "}
        <code>http://127.0.0.1:8800</code>
        {". The persona key is printed by that command ("}
        <code>{"Created local persona id=… key=…"}</code>
        {"), or comes from "}
        <code>SAWII_PERSONA_KEY</code>
        {"."}
      </p>
      <p className="hint">
        Remote mode: point at an existing Persona URL and use its <code>sp_…</code> key.
      </p>
      <div className="form">
        <span className="label">Backend URL</span>
        <input value={base} placeholder="http://127.0.0.1:8800" onChange={(e) => setBase(e.target.value)} />
        <span className="label">Persona key</span>
        <input
          type="password"
          value={key}
          placeholder="sp_…"
          onChange={(e) => setKey(e.target.value)}
          autoComplete="off"
        />
      </div>
      <div className="row">
        <button className="btn" disabled={busy} onClick={() => void test()}>
          Test connection
        </button>
        <button className="btn" data-kind="primary" disabled={!base.trim()} onClick={save}>
          Save and open the deck
        </button>
      </div>
      {note && <div className="err">{note}</div>}
      <p className="hint">
        Stored only in this app's local storage. Requests to {httpUrl("/v1/courts").replace(/\/v1\/courts$/, "")} use
        it as a bearer token.
      </p>
    </div>
  );
}