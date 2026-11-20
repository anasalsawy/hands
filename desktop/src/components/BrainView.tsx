// The Persona connection: which deployed Persona thinks for Hands, and which
// folders the engines may touch. Saved by the local backend (hands_app.py) and
// applied on restart. The key is write-only here: it is never shown again.
import { PhoneAccess } from "./PairPanels";
import { useEffect, useState } from "react";
import { httpUrl, authHeaders } from "../lib/connection";

export interface BrainInfo {
  url: string;
  model: string;
  key_set: boolean;
  default_url: string;
  configured: boolean;
}
export interface Boot {
  persona_key: string;
  brain: BrainInfo;
  roots: string[];
  approve: "ask" | "writes" | "auto";
  cloud?: boolean;
  computer_use?: boolean;
  missing?: string[];
}

interface CheckResult {
  ok: boolean;
  error?: string;
  reply?: string;
  seconds?: number;
  tools?: { supported: boolean; called?: string; seconds?: number; error?: string };
}

export async function fetchBoot(): Promise<Boot | null> {
  try {
    const res = await fetch(httpUrl("/hands/bootstrap"), { headers: authHeaders() });
    return res.ok ? ((await res.json()) as Boot) : null;
  } catch {
    return null;
  }
}

export function BrainView({ boot, first, onSaved }: { boot: Boot; first: boolean; onSaved: () => void }) {
  const [url, setUrl] = useState(boot.brain.url || boot.brain.default_url);
  const [key, setKey] = useState("");
  const [model, setModel] = useState(boot.brain.model);
  const [roots, setRoots] = useState(boot.roots.join("\n"));
  const [approve, setApprove] = useState(boot.approve);
  const [busy, setBusy] = useState("");
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => setCheck(null), [boot.brain.configured]);

  const runCheck = async () => {
    setBusy("check");
    setErr("");
    setCheck(null);
    try {
      const res = await fetch(httpUrl("/hands/brain-check"), { method: "POST", headers: authHeaders() });
      setCheck((await res.json()) as CheckResult);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "check failed");
    } finally {
      setBusy("");
    }
  };

  const save = async () => {
    setBusy("save");
    setErr("");
    try {
      const res = await fetch(httpUrl("/hands/config"), {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          brain_url: url.trim(),
          brain_key: key.trim(),
          brain_model: model.trim(),
          roots: roots.split("\n").map((r) => r.trim()).filter(Boolean),
          approve,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // the backend restarts to apply the lobes; wait for it to come back
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        const b = await fetchBoot();
        if (b && b.brain.configured) {
          setKey("");
          onSaved();
          return;
        }
      }
      throw new Error("the backend did not come back; restart Hands");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "save failed");
    } finally {
      setBusy("");
    }
  };

  const canSave = Boolean(url.trim()) && (boot.brain.key_set || key.trim()) && !busy;

  return (
    <div className="brain">
      <div className="brain-head">
        <h2>{first ? "Connect Hands to your Persona" : "Persona connection"}</h2>
        <p className="hint">
          Persona is the mind: it reads your goal, splits it into lines of work and thinks for every card. Hands is the
          body: each card drives this computer. Point Hands at your deployed Persona and give it that persona&apos;s key.
        </p>
      </div>

      <div className="panel">
        <h3>Brain</h3>
        <div className="form">
          <span className="label">Persona URL</span>
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder={boot.brain.default_url} />
          <span className="label">Persona key</span>
          <input
            type="password"
            autoComplete="off"
            value={key}
            placeholder={boot.brain.key_set ? "saved — leave blank to keep" : "sp_…"}
            onChange={(e) => setKey(e.target.value)}
          />
          <span className="label">Model name</span>
          <input value={model} onChange={(e) => setModel(e.target.value)} />
        </div>
        <p className="hint">
          Any OpenAI-compatible Persona endpoint works: the Modal <code>persona-crossing</code> URL or the Railway
          gateway. The URL may end in <code>/v1</code> or not.
        </p>
      </div>

      <div className="panel">
        <h3>What the hands may touch</h3>
        <div className="form">
          <span className="label">Folders</span>
          <textarea value={roots} onChange={(e) => setRoots(e.target.value)} rows={3} placeholder="one folder per line" />
          <span className="label">Approvals</span>
          <select value={approve} onChange={(e) => setApprove(e.target.value as Boot["approve"])}>
            <option value="ask">ask before acting (safest)</option>
            <option value="writes">ask before writes</option>
            <option value="auto">act without asking</option>
          </select>
        </div>
        <p className="hint">Engines only read and write inside these folders. Shell commands always follow the approval mode. There is no delete tool.</p>
      </div>

      <div className="row">
        <button className="btn" data-kind="primary" disabled={!canSave} onClick={() => void save()}>
          {busy === "save" ? "Saving, restarting…" : first ? "Save and open the deck" : "Save and restart"}
        </button>
        <button className="btn" disabled={!boot.brain.configured || Boolean(busy)} onClick={() => void runCheck()}>
          {busy === "check" ? "Testing…" : "Test the brain"}
        </button>
        {!boot.brain.configured && <span className="card-meta">save once, then test</span>}
      </div>

      {err && <div className="err">{err}</div>}
      {check && (
        <div className="panel" data-ok={check.ok ? "1" : "0"}>
          <h3>{check.ok ? "Brain answered" : "Brain did not answer"}</h3>
          {check.ok ? (
            <div className="kv">
              <b>Reply</b>
              <span>
                “{check.reply}” in {check.seconds}s
              </span>
              <b>Tool calls</b>
              <span>
                {check.tools?.supported
                  ? `working (called ${check.tools.called} in ${check.tools.seconds}s)`
                  : `not seen${check.tools?.error ? `: ${check.tools.error}` : " — cards need tool calls to act"}`}
              </span>
            </div>
          ) : (
            <div className="err">{check.error}</div>
          )}
        </div>
      )}
      {!first && !boot.cloud && <PhoneAccess />}
    </div>
  );
}
