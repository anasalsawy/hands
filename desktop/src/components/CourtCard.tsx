// One court = one card. Everything on it is a real call to the deck:
//   Start  -> POST /v1/courts/{id}/spawn   (its own lane + engine process)
//   Send   -> POST /v1/courts/{id}/goal    (starts it if it has no lane yet)
//   Stop   -> POST /v1/courts/{id}/cancel  (keeps the card)
//   Remove -> DELETE /v1/courts/{id}
//   Voice  -> POST /v1/courts/{id}/voice
import { useEffect, useState } from "react";
import { courtsApi, type Court, type CourtKind, type CourtTarget } from "../lib/client";
import { LiveSurface } from "./LiveSurface";

const LIVE: Court["status"][] = ["spawning", "running", "waiting"];

function ago(sec: number): string {
  const s = Math.max(0, Date.now() / 1000 - sec);
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

export function CourtCard({
  court,
  onChanged,
  onBusy,
}: {
  court: Court;
  onChanged: (c: Court) => void;
  onBusy: (id: string, busy: boolean, error?: string) => void;
}) {
  const [goal, setGoal] = useState(court.goal || "");
  const [target, setTarget] = useState<CourtTarget>(court.target || { kind: court.kind });
  const [showTarget, setShowTarget] = useState(false);

  useEffect(() => {
    setTarget(court.target || { kind: court.kind });
  }, [court.target, court.kind]);

  const live = LIVE.includes(court.status);

  const run = async (label: string, fn: () => Promise<Court>) => {
    onBusy(court.id, true);
    try {
      onChanged(await fn());
    } catch (err) {
      onBusy(court.id, false, `${label}: ${err instanceof Error ? err.message : "failed"}`);
      return;
    }
    onBusy(court.id, false);
  };

  const motor = typeof court.meta?.motor === "string" ? court.meta.motor : "";

  return (
    <section className="card" data-running={live ? "1" : "0"}>
      <header className="card-head">
        <span className="status" data-st={court.status}>
          {court.status}
        </span>
        <span className="card-title">{court.title || court.id}</span>
        <span className="spacer" />
        <span className="card-meta">{ago(court.updated_at)}</span>
        <button
          className="btn"
          data-size="sm"
          title="Change what this card shows"
          onClick={() => setShowTarget((v) => !v)}
        >
          target
        </button>
      </header>

      <LiveSurface target={target} kind={court.kind} lane={court.lane || undefined} active />

      <div className="card-body">
        {showTarget && (
          <div className="row">
            <label className="field" style={{ flex: "0 0 110px" }}>
              <span className="label">Kind</span>
              <select
                value={target.kind}
                onChange={(e) => setTarget({ ...target, kind: e.target.value as CourtKind })}
              >
                <option value="desktop">desktop</option>
                <option value="window">window</option>
                <option value="region">region</option>
                <option value="browser">browser</option>
              </select>
            </label>
            <label className="field">
              <span className="label">Window title</span>
              <input
                value={target.title || ""}
                placeholder="substring of a window title"
                onChange={(e) => setTarget({ ...target, title: e.target.value })}
              />
            </label>
            <label className="field" style={{ flex: "0 0 90px" }}>
              <span className="label">FPS</span>
              <input
                type="number"
                min={1}
                max={20}
                value={target.fps ?? 6}
                onChange={(e) => setTarget({ ...target, fps: Number(e.target.value) || 6 })}
              />
            </label>
            <span className="hint">
              Changing the target creates a replacement card; there is no retarget route on the deck.
            </span>
          </div>
        )}

        <div className="kv">
          <b>Card</b>
          <span>{court.id}</span>
          <b>Work line</b>
          <span>{court.lane || "not bound"}</span>
          <b>Engine</b>
          <span>
            {court.link || "none"}
            {motor ? ` · ${motor}` : ""}
          </span>
          <b>Surface</b>
          <span>
            {court.kind}
            {target.title ? ` · ${target.title}` : ""}
          </span>
        </div>

        <label className="field">
          <span className="label">Goal</span>
          <textarea
            className="goal"
            value={goal}
            placeholder="what this card should do on the desktop"
            onChange={(e) => setGoal(e.target.value)}
          />
        </label>

        <div className="row">
          {!live && court.status !== "stopped" && (
            <button
              className="btn"
              data-kind="primary"
              onClick={() => void run("start", () => courtsApi.spawn(court.id, { goal: goal || undefined }))}
            >
              Start instance
            </button>
          )}
          <button
            className="btn"
            disabled={!goal.trim()}
            title="Starts the instance if none is running, otherwise steers it"
            onClick={() => void run("goal", () => courtsApi.send(court.id, goal.trim()))}
          >
            {live ? "Steer" : "Send goal"}
          </button>
          <button
            className="btn"
            data-kind="danger"
            disabled={!live}
            onClick={() => void run("stop", () => courtsApi.cancel(court.id))}
          >
            Stop
          </button>
          <button
            className="btn"
            title="Hand this card's last update to the voice line"
            onClick={() => void courtsApi.voice(court.id, "status").catch((err: Error) => onBusy(court.id, false, err.message))}
          >
            Voice
          </button>
          <span className="spacer" />
          <button
            className="btn"
            data-kind="danger"
            onClick={() => {
              onBusy(court.id, true);
              courtsApi
                .remove(court.id)
                .then(() => onChanged({ ...court, status: "stopped", running: false }))
                .catch((err: Error) => onBusy(court.id, false, err.message));
            }}
          >
            Remove
          </button>
        </div>

        {court.error && <div className="err">{court.error}</div>}
      </div>
    </section>
  );
}