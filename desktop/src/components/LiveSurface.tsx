// One card = one live desktop. Paints continuous JPEG frames from
// `/v1/computer/stream` onto a canvas. When the capture route is not enabled
// (SAWII_COMPUTER_STREAM=0) or the backend refuses, the slot says so instead of
// faking a picture.
import { useEffect, useRef, useState } from "react";
import { systemApi, type CourtTarget } from "../lib/client";
import { frameBlob, latestFrames, paintBitmap } from "../lib/frames";

type StreamState = "probing" | "live" | "down" | "off";

function paintIdle(ctx: CanvasRenderingContext2D, w: number, h: number, note: string) {
  ctx.fillStyle = "#07080a";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(148,163,184,0.14)";
  ctx.strokeRect(10.5, 10.5, w - 21, h - 21);
  ctx.fillStyle = "rgba(148,163,184,0.75)";
  ctx.font = "600 13px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("Live desktop", 22, h / 2 - 6);
  ctx.fillStyle = "rgba(148,163,184,0.5)";
  ctx.font = "11px ui-monospace, Consolas, monospace";
  ctx.fillText(note.slice(0, 72), 22, h / 2 + 14);
}

export function LiveSurface({
  target,
  kind,
  lane,
}: {
  target: CourtTarget;
  kind: string;
  lane?: string;
  active: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [state, setState] = useState<StreamState>("probing");
  const [note, setNote] = useState("connecting");
  const [frames, setFrames] = useState(0);

  useEffect(() => {
    let alive = true;
    let gotFrame = false;
    let sock: WebSocket | null = null;
    let timer = 0;
    let count = 0;
    let lastUpdate = 0;
    let failureNote = "";
    setState("probing");
    setNote("connecting");
    setFrames(0);

    const paintPlaceholder = (message: string) => {
      if (!alive) return;
      const c = canvasRef.current;
      const ctx = c?.getContext("2d");
      if (c && ctx) paintIdle(ctx, c.width, c.height, message);
    };
    paintPlaceholder("connecting");

    const renderer = latestFrames({
      decode: (blob) => createImageBitmap(blob),
      paint: (bitmap) => {
        const canvas = canvasRef.current;
        if (!alive || !canvas || sock?.readyState !== WebSocket.OPEN) return;
        paintBitmap(canvas, bitmap);
        gotFrame = true;
        failureNote = "";
        window.clearTimeout(timer);
        setState("live");
        setNote("live capture");
        count += 1;
        const now = performance.now();
        if (count === 1 || now - lastUpdate >= 1000) {
          lastUpdate = now;
          setFrames(count);
        }
      },
      onError: () => {
        failureNote = "invalid capture frame";
        setState("down");
        setNote(failureNote);
      },
    });

    // A browser card shows the browser hub surface; everything else uses the
    // computer capture route with the card's own target.
    const url = kind === "browser" ? systemApi.browserStreamUrl(lane) : systemApi.computerStreamUrl(target);
    if (!url) {
      setState("off");
      setNote("no backend configured");
      paintPlaceholder("no backend configured");
      return () => {
        alive = false;
        renderer.dispose();
      };
    }

    try {
      sock = new WebSocket(url);
      sock.binaryType = "arraybuffer";
    } catch {
      setState("down");
      setNote("socket refused");
      paintPlaceholder("socket refused");
      return () => {
        alive = false;
        renderer.dispose();
      };
    }

    timer = window.setTimeout(() => {
      if (!alive || gotFrame) return;
      setState("down");
      failureNote = "no frames — capture route off or busy";
      setNote(failureNote);
      paintPlaceholder(failureNote);
    }, 4000);

    sock.onmessage = (ev) => {
      if (!alive) return;
      if (ev.data instanceof ArrayBuffer) {
        renderer.push(new Blob([ev.data], { type: "image/jpeg" }));
        return;
      }
      let data: Record<string, unknown> | null = null;
      try {
        data = JSON.parse(String(ev.data));
      } catch {
        return;
      }
      if (!data) return;
      if (data.type === "error") {
        window.clearTimeout(timer);
        renderer.dispose();
        setState("down");
        failureNote = String(data.detail || data.message || "capture refused");
        setNote(failureNote);
        paintPlaceholder(failureNote);
        return;
      }
      if (data.type === "ready") {
        setNote(`${String(data.kind || target.kind)} · ${String(data.backend || "capture")}`);
        return;
      }
      try {
        const blob = frameBlob(data);
        if (blob) renderer.push(blob);
      } catch {
        failureNote = "invalid capture frame";
        setState("down");
        setNote(failureNote);
      }
    };

    sock.onerror = () => {
      if (!alive) return;
      window.clearTimeout(timer);
      renderer.dispose();
      setState("down");
      failureNote ||= "capture socket error";
      setNote(failureNote);
    };

    sock.onclose = (event) => {
      if (!alive) return;
      window.clearTimeout(timer);
      renderer.dispose();
      setState("down");
      const message = failureNote || event.reason || "capture closed";
      setNote(message);
      paintPlaceholder(message);
    };

    return () => {
      alive = false;
      window.clearTimeout(timer);
      renderer.dispose();
      try {
        sock?.close();
      } catch {
        /* already closed */
      }
    };
  }, [target.hwnd, target.pid, target.title, target.x, target.y, target.w, target.h, target.fps, target.kind, kind, lane]);

  return (
    <div className="screen">
      <canvas ref={canvasRef} width={960} height={540} aria-label={`live ${kind} surface`} />
      {state !== "live" && <div className="screen-note">{note}</div>}
      {state === "live" && <div className="screen-note">{frames} frames · {target.kind}</div>}
    </div>
  );
}