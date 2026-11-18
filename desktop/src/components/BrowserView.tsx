// Browser pane. Paints the live page the browser hub is driving, from
// `WS /v1/browser/stream` (`{type:"frame", url, title, screenshot_png_b64}`).
// Whether that hub exists at all is reported by `GET /v1/computer` → `browser`.
import { useCallback, useEffect, useRef, useState } from "react";
import { probe, systemApi, type ComputerStatus } from "../lib/client";
import { keyedWsUrl } from "../lib/connection";
import { frameBlob, latestFrames, paintBitmap } from "../lib/frames";

export function BrowserView() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [conn, setConn] = useState<"off" | "open" | "closed" | "error">("off");
  const [meta, setMeta] = useState({ title: "", url: "" });
  const [frames, setFrames] = useState(0);
  const [hub, setHub] = useState<ComputerStatus | null>(null);
  const [note, setNote] = useState("");
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    void probe<ComputerStatus>("/v1/computer").then((r) => setHub(r.data ?? null));
  }, []);

  useEffect(() => {
    let alive = true;
    let sock: WebSocket | null = null;
    let count = 0;
    let lastUpdate = 0;
    const renderer = latestFrames({
      decode: (blob) => createImageBitmap(blob),
      paint: (bitmap) => {
        const canvas = canvasRef.current;
        if (!alive || !canvas || sock?.readyState !== WebSocket.OPEN) return;
        paintBitmap(canvas, bitmap);
        count += 1;
        const now = performance.now();
        if (count === 1 || now - lastUpdate >= 1000) {
          lastUpdate = now;
          setFrames(count);
        }
        setNote("");
      },
      onError: () => setNote("invalid browser frame"),
    });
    setFrames(0);
    setNote("");
    try {
      sock = new WebSocket(systemApi.browserStreamUrl());
    } catch {
      setConn("error");
      setNote("browser stream socket refused — no hub on this backend");
      return () => {
        alive = false;
        renderer.dispose();
      };
    }
    sock.onopen = () => setConn("open");
    sock.onclose = () => {
      if (!alive) return;
      renderer.dispose();
      setConn("closed");
      setNote("closed — the browser hub is not attached, or the session ended");
    };
    sock.onerror = () => {
      if (!alive) return;
      renderer.dispose();
      setConn("error");
      setNote("error — no browser hub on this backend");
    };
    sock.onmessage = (ev) => {
      if (!alive) return;
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(String(ev.data)) as Record<string, unknown>;
      } catch {
        return;
      }
      if (data.type !== "frame") return;
      setMeta({ title: String(data.title || ""), url: String(data.url || "") });
      try {
        const blob = frameBlob(data);
        if (blob) renderer.push(blob);
      } catch {
        setNote("invalid browser frame");
      }
    };
    return () => {
      alive = false;
      renderer.dispose();
      try {
        sock?.close();
      } catch {
        /* noop */
      }
    };
  }, [nonce]);

  const reconnect = useCallback(() => setNonce((n) => n + 1), []);

  return (
    <div className="split">
      <div className="panel">
        <h3>Live browser surface</h3>
        <div className="row">
          <span className="pill">
            <span className="dot" data-s={conn === "open" ? "live" : conn === "off" ? "idle" : "down"} />
            WS /v1/browser/stream · {conn}
            {frames ? ` · ${frames} frames` : ""}
          </span>
          <button className="btn" onClick={reconnect}>
            Reconnect
          </button>
          <span className="spacer" />
          <span className="card-meta">{meta.title || meta.url || "no page reported"}</span>
        </div>
        <div className="screen">
          <canvas ref={canvasRef} width={1280} height={720} aria-label="browser live view" />
          {(frames === 0 || note) && <div className="screen-note">{note || "waiting for the first frame"}</div>}
        </div>
      </div>

      <div className="stack">
        <div className="panel">
          <h3>Hub</h3>
          {hub ? (
            <div className="kv">
              <b>Browser hub</b>
              <span>{String(hub.browser ?? "—")}</span>
              <b>Desktop</b>
              <span>{String(hub.desktop ?? "—")}</span>
            </div>
          ) : (
            <p className="hint">GET /v1/computer not answering.</p>
          )}
          <p className="hint">
            The pane renders frames only while a browser hub is attached to the persona. It is not a general web
            browser and does not browse on its own — the hub drives the page, this window shows it.
          </p>
          <p className="hint">
            Stream endpoint: <code>{keyedWsUrl("/v1/browser/stream").replace(/key=[^&]*/, "key=…")}</code>
          </p>
        </div>
      </div>
    </div>
  );
}