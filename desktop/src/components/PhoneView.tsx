// Phone. Real routes, no make-believe: GET /v1/calls lists sessions,
// POST /v1/calls/outbound places one, POST /v1/calls/{sid}/end hangs up.
// Without Twilio credentials the backend runs the call hub in dry-run — the
// call is placed against the hub and you type what the other side says, which
// is exactly what this screen shows.
import { useCallback, useEffect, useState } from "react";
import { probe, systemApi, type CallRow } from "../lib/client";

export function PhoneView() {
  const [to, setTo] = useState("");
  const [hint, setHint] = useState("");
  const [calls, setCalls] = useState<CallRow[] | null>(null);
  const [note, setNote] = useState("");
  const [routeState, setRouteState] = useState<{ ok: boolean; detail: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    const res = await probe<{ calls: CallRow[] }>("/v1/calls");
    if (res.ok) {
      setCalls(res.data?.calls ?? []);
      setRouteState({ ok: true, detail: "GET /v1/calls · 200" });
    } else {
      setCalls(null);
      setRouteState({
        ok: false,
        detail: `GET /v1/calls · ${res.status || "no response"} ${res.error || ""}`.trim(),
      });
    }
  }, []);

  useEffect(() => {
    void reload();
    const t = window.setInterval(() => void reload(), 8000);
    return () => window.clearInterval(t);
  }, [reload]);

  const place = async () => {
    if (!to.trim()) return;
    setBusy(true);
    setNote("");
    try {
      const out = await systemApi.placeCall(to.trim(), hint.trim());
      setNote(
        `placed · ${Object.entries(out)
          .slice(0, 5)
          .map(([k, v]) => `${k}=${String(v)}`)
          .join(" ")}`,
      );
      await reload();
    } catch (err) {
      setNote(`place failed: ${err instanceof Error ? err.message : "failed"}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="split">
      <div className="panel">
        <h3>Place a call</h3>
        <div className="form">
          <span className="label">Number</span>
          <input value={to} placeholder="+15551234567" onChange={(e) => setTo(e.target.value)} />
          <span className="label">Context</span>
          <input
            value={hint}
            placeholder="what the call is about"
            onChange={(e) => setHint(e.target.value)}
          />
        </div>
        <div className="row">
          <button className="btn" data-kind="primary" disabled={busy || !to.trim()} onClick={() => void place()}>
            Dial
          </button>
          <button className="btn" disabled={busy} onClick={() => void reload()}>
            Refresh
          </button>
        </div>
        {note && <div className="err">{note}</div>}
        <p className="hint">
          Dials <code>POST /v1/calls/outbound</code>. With no Twilio credentials the hub answers in dry-run, so the
          call registers and you can drive it by typing what the other person says.
        </p>
      </div>

      <div className="panel">
        <h3>Sessions</h3>
        <div className="row">
          <span className="pill">
            <span className="dot" data-s={routeState?.ok ? "live" : "down"} />
            {routeState?.detail ?? "probing"}
          </span>
        </div>
        {calls && calls.length === 0 && <p className="hint">No calls in this persona's hub.</p>}
        {calls && calls.length > 0 && (
          <div className="list">
            {calls.map((c) => (
              <div key={c.sid} style={{ display: "flex", gap: 8 }}>
                <span style={{ flex: 1 }}>{c.to}</span>
                <span>{c.status || "—"}</span>
                <span>{new Date(c.started_at * 1000).toLocaleTimeString()}</span>
                <button
                  className="btn"
                  data-size="sm"
                  data-kind="danger"
                  onClick={() =>
                    void systemApi
                      .endCall(c.sid)
                      .then(() => reload())
                      .catch((err: Error) => setNote(`end failed: ${err.message}`))
                  }
                >
                  End
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}