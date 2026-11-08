// The deck: one card per Court, plus the controls that act on the whole deck.
import { useMemo, useState } from "react";
import { courtsApi, type Court, type CourtKind } from "../lib/client";
import type { DeckState } from "../hooks/useDeck";
import { CourtCard } from "./CourtCard";

interface Busy {
  busy: boolean;
  error?: string;
}

export function DeckView({ deck }: { deck: DeckState }) {
  const [title, setTitle] = useState("");
  const [kind, setKind] = useState<CourtKind>("desktop");
  const [busy, setBusy] = useState<Record<string, Busy>>({});
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);

  const errors = useMemo(() => {
    const rows: string[] = [];
    for (const [id, b] of Object.entries(busy)) if (b.error) rows.push(`${id}: ${b.error}`);
    if (error) rows.push(error);
    return rows;
  }, [busy, error]);

  const onBusy = (id: string, next: boolean, message?: string) =>
    setBusy((rows) => ({ ...rows, [id]: { busy: next, error: message } }));

  const create = async (spawn: boolean) => {
    setCreating(true);
    setError("");
    try {
      const court = await courtsApi.create({ title: title.trim() || undefined, kind });
      deck.apply(court);
      setTitle("");
      if (spawn) deck.apply(await courtsApi.spawn(court.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "create failed");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="stack">
      <div className="panel">
        <h3>Deck</h3>
        <div className="row">
          <label className="field" style={{ flex: "0 0 240px" }}>
            <span className="label">Card title</span>
            <input
              value={title}
              placeholder="optional"
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void create(true);
              }}
            />
          </label>
          <label className="field" style={{ flex: "0 0 130px" }}>
            <span className="label">Surface</span>
            <select value={kind} onChange={(e) => setKind(e.target.value as CourtKind)}>
              <option value="desktop">desktop</option>
              <option value="window">window</option>
              <option value="region">region</option>
              <option value="browser">browser</option>
            </select>
          </label>
          <button className="btn" data-kind="primary" disabled={creating} onClick={() => void create(true)}>
            Add card and start
          </button>
          <button className="btn" disabled={creating} onClick={() => void create(false)}>
            Add card
          </button>
          <span className="spacer" />
          <span className="pill">
            <span className="dot" data-s={deck.link === "live" ? "live" : deck.link === "down" ? "down" : "idle"} />
            {deck.snapshot ? `${deck.snapshot.count} cards · ${deck.snapshot.running} running` : "no snapshot"}
          </span>
          <button className="btn" onClick={() => void deck.refresh()}>
            Refresh
          </button>
        </div>
        {errors.map((e) => (
          <div className="err" key={e}>
            {e}
          </div>
        ))}
        {deck.error && <div className="err">deck: {deck.error}</div>}
      </div>

      {deck.loading ? (
        <div className="empty">
          <p>Reading the deck from the backend.</p>
        </div>
      ) : deck.courts.length === 0 ? (
        <div className="empty">
          <h2>No cards yet</h2>
          <p className="hint">
            Add a card above. Each card runs its own desktop instance, and they run at the same time — a card holds
            its own work line and its own engine process.
          </p>
        </div>
      ) : (
        <div className="grid">
          {deck.courts.map((court: Court) => (
            <CourtCard key={court.id} court={court} onChanged={deck.apply} onBusy={onBusy} />
          ))}
        </div>
      )}

      <div className="panel">
        <h3>Deck events</h3>
        {deck.log.length === 0 ? (
          <p className="hint">
            Nothing yet. Events arrive over <code>WS /v1/court-events</code>; the grid also re-reads{" "}
            <code>GET /v1/courts</code> every 5 seconds.
          </p>
        ) : (
          <div className="log">
            {deck.log.map((row) => (
              <div key={row.id}>
                <span>{new Date(row.at).toLocaleTimeString()}</span>
                {row.text}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}