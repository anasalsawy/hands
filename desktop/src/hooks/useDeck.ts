// Live deck state. Keeps a local copy of every court, refreshed by the typed
// WebSocket `/v1/court-events`, with a REST poll behind it so the grid is never
// stuck if the socket drops (dev reloads, backend restarts, proxy hiccups).
import { useCallback, useEffect, useRef, useState } from "react";
import { courtsApi, type Court, type CourtEvent, type DeckSnapshot } from "../lib/client";
import { appendDeckEvent, type DeckLogRow } from "../lib/deckLog";
export type { DeckLogRow } from "../lib/deckLog";

export type LinkState = "connecting" | "live" | "down";

export interface DeckState {
  courts: Court[];
  snapshot: DeckSnapshot | null;
  log: DeckLogRow[];
  link: LinkState;
  error: string;
  loading: boolean;
  refresh: () => Promise<void>;
  apply: (court: Court) => void;
}

export function useDeck(): DeckState {
  const [courts, setCourts] = useState<Court[]>([]);
  const [snapshot, setSnapshot] = useState<DeckSnapshot | null>(null);
  const [log, setLog] = useState<DeckLogRow[]>([]);
  const [link, setLink] = useState<LinkState>("connecting");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const seq = useRef(0);

  const pushLog = useCallback((ev: CourtEvent) => {
    seq.current += 1;
    setLog(appendDeckEvent(ev, seq.current));
  }, []);

  const apply = useCallback((court: Court) => {
    setCourts((rows) => {
      const i = rows.findIndex((c) => c.id === court.id);
      if (i === -1) return [...rows, court];
      const next = rows.slice();
      next[i] = court;
      return next;
    });
  }, []);

  const refresh = useCallback(async () => {
    try {
      const snap = await courtsApi.snapshot();
      setSnapshot(snap);
      setCourts(snap.courts);
      setLink("live");
      setError("");
    } catch (err) {
      setLink("down");
      setError(err instanceof Error ? err.message : "deck unreachable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Typed socket: snapshot frame first, then one frame per court change.
  useEffect(() => {
    let closed = false;
    let sock: WebSocket | null = null;
    let retry: number | undefined;

    const connect = () => {
      if (closed) return;
      try {
        sock = courtsApi.socket();
      } catch {
        setLink("down");
        return;
      }
      sock.onmessage = (msg) => {
        let data: (CourtEvent & { courts?: Court[] }) | null = null;
        try {
          data = JSON.parse(String(msg.data));
        } catch {
          return;
        }
        if (!data || typeof data !== "object") return;
        if (data.kind === "snapshot") {
          const courts = (data as unknown as DeckSnapshot).courts;
          if (Array.isArray(courts)) {
            setCourts(courts);
            setSnapshot(data as unknown as DeckSnapshot);
          }
          setLink("live");
          setError("");
          return;
        }
        if (data.kind === "error") {
          setError(String(data.detail || "event socket rejected"));
          setLink("down");
          return;
        }
        if (data.court) {
          apply(data.court);
          pushLog(data);
          setLink("live");
        }
      };
      sock.onclose = () => {
        if (closed) return;
        setLink("down");
        retry = window.setTimeout(connect, 3000);
      };
      sock.onerror = () => {
        try {
          sock?.close();
        } catch {
          /* already gone */
        }
      };
    };

    connect();
    const poll = window.setInterval(() => void refresh(), 5000);
    return () => {
      closed = true;
      window.clearTimeout(retry);
      window.clearInterval(poll);
      try {
        sock?.close();
      } catch {
        /* noop */
      }
    };
  }, [apply, pushLog, refresh]);

  return { courts, snapshot, log, link, error, loading, refresh, apply };
}