// UI-TARS motor panel. The engine (`sawii_hands.py`) has no LLM of its own and
// is started per work line, so "start a motor" means "spawn an instance" —
// the same deck route a card uses. This panel reports what is actually wired:
//   GET /v1/hands      engines connected to the hands socket, with their tools
//   GET /v1/computer   computer tool surface available to this persona
//   GET /v1/link       registered Link adapters, strategy + live lane counts
//   GET /v1/computer/capture   whether continuous capture is available
import { useCallback, useEffect, useState } from "react";
import {
  courtsApi,
  probe,
  systemApi,
  type ComputerStatus,
  type HandsStatus,
  type LinkInfo,
  type WindowRow,
} from "../lib/client";
import type { DeckState } from "../hooks/useDeck";

type Row = { label: string; ok: boolean; note: string };

export function MotorView({ deck }: { deck: DeckState }) {
  const [hands, setHands] = useState<HandsStatus | null>(null);
  const [computer, setComputer] = useState<ComputerStatus | null>(null);
  const [links, setLinks] = useState<Record<string, LinkInfo>>({});
  const [rows, setRows] = useState<Row[]>([]);
  const [windows, setWindows] = useState<WindowRow[] | null>(null);
  const [note, setNote] = useState("");

  const reload = useCallback(async () => {
    const [h, c, l, cap] = await Promise.all([
      probe<HandsStatus>("/v1/hands"),
      probe<ComputerStatus>("/v1/computer"),
      probe<{ links: Record<string, LinkInfo> }>("/v1/link"),
      probe<{ ok: boolean; [k: string]: unknown }>("/v1/computer/capture"),
    ]);
    setHands(h.data ?? null);
    setComputer(c.data ?? null);
    setLinks(l.data?.links ?? {});
    setRows([
      { label: "GET /v1/hands", ok: h.ok, note: h.ok ? `${h.data?.connected ? "engine connected" : "no engine connected"}` : `${h.status} ${h.error}` },
      { label: "GET /v1/computer", ok: c.ok, note: c.ok ? `${c.data?.tools?.length ?? 0} tools` : `${c.status} ${c.error}` },
      { label: "GET /v1/link", ok: l.ok, note: l.ok ? `${Object.keys(l.data?.links ?? {}).length} adapters` : `${l.status} ${l.error}` },
      {
        label: "GET /v1/computer/capture",
        ok: cap.ok,
        note: cap.ok
          ? Object.entries(cap.data ?? {})
              .filter(([k]) => k !== "ok")
              .map(([k, v]) => `${k}=${String(v)}`)
              .join(" ") || "available"
          : `${cap.status} ${cap.error}`,
      },
    ]);
    setNote("");
  }, []);

  useEffect(() => {
    void reload();
    const t = window.setInterval(() => void reload(), 6000);
    return () => window.clearInterval(t);
  }, [reload]);

  const running = deck.courts.filter((c) => c.running);

  return (
    <div className="split">
      <div className="stack">
        <div className="panel">
          <h3>Engine instances</h3>
          <p className="hint">
            Each running card owns one <code>sawii_hands.py</code> process, started by its own work line. Starting or
            stopping here uses the same deck routes as the cards.
          </p>
          {running.length === 0 ? (
            <div className="row">
              <span className="pill">
                <span className="dot" />
                no instance running
              </span>
              <button
                className="btn"
                data-kind="primary"
                onClick={() =>
                  void courtsApi
                    .create({ title: "motor", kind: "desktop" })
                    .then((c) => courtsApi.spawn(c.id))
                    .then((c) => {
                      deck.apply(c);
                      void reload();
                    })
                    .catch((err: Error) => setNote(err.message))
                }
              >
                Start one instance
              </button>
            </div>
          ) : (
            <div className="stack" style={{ gap: 6 }}>
              {running.map((c) => (
                <div className="row" key={c.id}>
                  <span className="status" data-st={c.status}>
                    {c.status}
                  </span>
                  <span style={{ minWidth: 120 }}>{c.title}</span>
                  <span className="card-meta" style={{ flex: 1 }}>
                    lane {c.lane || "—"} · {c.link || "no adapter"}
                  </span>
                  <button
                    className="btn"
                    data-kind="danger"
                    data-size="sm"
                    onClick={() =>
                      void courtsApi
                        .cancel(c.id)
                        .then((next) => {
                          deck.apply(next);
                          void reload();
                        })
                        .catch((err: Error) => setNote(err.message))
                    }
                  >
                    Stop
                  </button>
                </div>
              ))}
            </div>
          )}
          {note && <div className="err">{note}</div>}
        </div>

        <div className="panel">
          <h3>Connected engines · GET /v1/hands</h3>
          {!hands || !hands.connected ? (
            <p className="hint">
              Nothing is on the hands socket yet. An engine appears here once a card spawns one and it authenticates
              back over <code>WS /v1/hands</code>.
            </p>
          ) : (
            <div className="list">
              {Object.entries(hands.places).map(([place, p]) => (
                <div key={place}>
                  {place} · {p.host} · lane {p.lane || "—"} · {p.tools.length} tools
                </div>
              ))}
              <div>tools: {hands.tools.slice(0, 24).join(", ")}</div>
            </div>
          )}
        </div>

        <div className="panel">
          <h3>Link adapters · GET /v1/link</h3>
          {Object.keys(links).length === 0 ? (
            <p className="hint">No adapters registered. Local mode registers <code>hands-engine</code>.</p>
          ) : (
            <div className="list">
              {Object.entries(links).map(([name, info]) => (
                <div key={name}>
                  {name} · {info.strategy} · lanes {info.active_lanes ?? 0}/{info.total_lanes ?? 0} · steer{" "}
                  {info.supports_steer ? "yes" : "no"}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="stack">
        <div className="panel">
          <h3>Endpoint health</h3>
          <div className="list">
            {rows.map((r) => (
              <div key={r.label} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span className="dot" data-s={r.ok ? "live" : "down"} />
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>{r.label}</span>
                <span style={{ color: "var(--fg-faint)" }}>{r.note}</span>
              </div>
            ))}
          </div>
          <button className="btn" onClick={() => void reload()}>
            Re-check
          </button>
        </div>

        <div className="panel">
          <h3>Computer surface · GET /v1/computer</h3>
          {computer ? (
            <div className="kv">
              <b>Desktop</b>
              <span>{String(computer.desktop ?? "—")}</span>
              <b>Browser hub</b>
              <span>{String(computer.browser ?? "—")}</span>
              <b>Tools</b>
              <span>{(computer.tools || []).slice(0, 18).join(", ") || "none"}</span>
            </div>
          ) : (
            <p className="hint">Route not reachable.</p>
          )}
        </div>

        <div className="panel">
          <h3>Windows · GET /v1/computer/windows</h3>
          <div className="row">
            <button
              className="btn"
              onClick={() =>
                void systemApi
                  .windows()
                  .then((r) => setWindows(r.windows))
                  .catch(() => setWindows([]))
              }
            >
              List windows
            </button>
          </div>
          {windows && windows.length === 0 && <p className="hint">None returned (capture disabled or denied).</p>}
          {windows && windows.length > 0 && (
            <div className="list">
              {windows.slice(0, 40).map((w) => (
                <div key={w.hwnd}>
                  {w.title || "(untitled)"} · hwnd {w.hwnd}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}