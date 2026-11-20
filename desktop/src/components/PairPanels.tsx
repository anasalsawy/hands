import { useCallback, useEffect, useState } from "react";
import { authHeaders, httpUrl, setSessionKey } from "../lib/connection";

interface PairInfo {
  lan: boolean;
  code: string;
  expires_in: number;
  urls: string[];
}

/** On the computer: switch on phone access and show the address + one-time code. */
export function PhoneAccess() {
  const [info, setInfo] = useState<PairInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const load = useCallback(async (fresh = false) => {
    try {
      const res = await fetch(httpUrl("/hands/pair-info", fresh ? { new: 1 } : undefined), { headers: authHeaders() });
      if (res.ok) setInfo((await res.json()) as PairInfo);
    } catch {
      /* backend restarting */
    }
  }, []);

  useEffect(() => {
    void load();
    const t = window.setInterval(() => void load(), 15000);
    return () => window.clearInterval(t);
  }, [load]);

  const toggle = async () => {
    if (!info) return;
    setBusy(true);
    setErr("");
    try {
      await fetch(httpUrl("/hands/config"), {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ lan: !info.lan }),
      });
      setErr("Restarting Hands to apply…");
    } catch {
      setErr("Could not change it.");
    }
    setBusy(false);
  };

  return (
    <div className="panel">
      <h3>Phone and other devices</h3>
      <p className="card-meta">
        Open Hands from a phone on the same Wi-Fi. Your computer still does the work; the phone is a window onto it.
      </p>
      <div className="row">
        <button className="btn" onClick={() => void toggle()} disabled={busy || !info}>
          {info?.lan ? "Turn off phone access" : "Turn on phone access"}
        </button>
        {info?.lan && (
          <button className="btn ghost" onClick={() => void load(true)}>
            New code
          </button>
        )}
      </div>
      {info?.lan && (
        <div className="kv">
          <b>Open on phone</b>
          <span className="mono">{info.urls.length ? info.urls.join("   ") : "no network address found"}</span>
          <b>Code</b>
          <span className="mono pair-code">{info.code ? `${info.code.slice(0, 3)} ${info.code.slice(3)}` : "—"}</span>
          <b>Valid for</b>
          <span>{Math.ceil(info.expires_in / 60)} min, one use</span>
        </div>
      )}
      <p className="card-meta">Anyone on your network who has the code can use your computer through Hands. Turn it off when not needed.</p>
      {err && <div className="err">{err}</div>}
    </div>
  );
}

/** On a phone: trade the code for access. */
export function PairScreen({ onPaired }: { onPaired: () => void }) {
  const [code, setCode] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setErr("");
    try {
      const res = await fetch(httpUrl("/hands/pair"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: code.trim() }),
      });
      const j = (await res.json()) as { persona_key?: string; error?: { message: string } };
      if (res.ok && j.persona_key) {
        setSessionKey(j.persona_key);
        try {
          localStorage.setItem("hands.paired_key", j.persona_key);
        } catch {
          /* private mode */
        }
        onPaired();
      } else setErr(j.error?.message || "Could not pair");
    } catch {
      setErr("Could not reach your computer.");
    }
    setBusy(false);
  };

  return (
    <div className="panel pair">
      <h3>Sign in to Hands</h3>
      <p className="card-meta">Type the access key for this Hands. On a phone paired with your own computer, type the 6-digit code shown under Persona → Phone access.</p>
      <input
        className="pair-input"
        type="password"
        autoComplete="current-password"
        autoCapitalize="off"
        spellCheck={false}
        placeholder="access key or code"
        value={code}
        onChange={(e) => setCode(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && void submit()}
      />
      <button className="btn" onClick={() => void submit()} disabled={busy || code.trim().length < 6}>
        {busy ? "Signing in…" : "Sign in"}
      </button>
      {err && <div className="err">{err}</div>}
    </div>
  );
}
