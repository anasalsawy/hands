// One card = one live desktop. Paints continuous JPEG frames from
// `/v1/computer/stream` onto a canvas. When the capture route is not enabled
// (SAWII_COMPUTER_STREAM=0) or the backend refuses, the slot says so instead of
// faking a picture.
import { useEffect, useRef, useState } from "react";
import { systemApi, type CourtTarget } from "../lib/client";

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

async function paintFrame(canvas: HTMLCanvasElement, blob: Blob) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const bmp = await createImageBitmap(blob);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Math.min(canvas.width / bmp.width, canvas.height / bmp.height);
  const dw = Math.round(bmp.width * scale);
  const dh = Math.round(bmp.height * scale);
  ctx.drawImage(bmp, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
  bmp.close();
}

export function LiveSurface({
  target,
  kind,
  active,
}: {
  target: CourtTarget;
  kind: string;
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
    let raf = 0;
    let timer = 0;

    const idleLoop = (message: string) => {
      const tick = () => {
        if (!alive || gotFrame) return;
        const c = canvasRef.current;
        const ctx = c?.getContext("2d");
        if (c && ctx) paintIdle(ctx, c.width, c.height, message);
        raf = requestAnimationFrame(tick);
      };
      tick();
    };

    // A browser card shows the browser hub surface; everything else uses the
    // computer capture route with the card's own target.
    const url = kind === "browser" ? systemApi.browserStreamUrl() : systemApi.computerStreamUrl(target);
    if (!url) {
      setState("off");
      setNote("no backend configured");
      idleLoop("no backend configured");
      return () => {
        alive = false;
        cancelAnimationFrame(raf);
      };
    }

    try {
      sock = new WebSocket(url);
      sock.binaryType = "arraybuffer";
    } catch {
      setState("down");
      setNote("socket refused");
      idleLoop("socket refused");
      return () => {
        alive = false;
        cancelAnimationFrame(raf);
      };
    }

    timer = window.setTimeout(() => {
      if (!alive || gotFrame) return;
      setState("down");
      setNote("no frames — capture route off or busy");
      idleLoop("no frames — capture route off or busy");
    }, 4000);

    sock.onmessage = (ev) => {
      if (!alive) return;
      const canvas = canvasRef.current;
      if (ev.data instanceof ArrayBuffer) {
        if (!canvas) return;
        gotFrame = true;
        setState("live");
        setNote("live capture");
        void paintFrame(canvas, new Blob([ev.data], { type: "image/jpeg" })).then(() => {
          if (alive) setFrames((n) => n + 1);
        });
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
        setState("down");
        setNote(String(data.detail || data.message || "capture refused"));
        return;
      }
      if (data.type === "ready") {
        setNote(`${String(data.kind || target.kind)} · ${String(data.backend || "capture")}`);
        window.clearTimeout(timer);
        return;
      }
      const b64 = (data.jpeg_b64 || data.png_b64 || data.data_b64) as string | undefined;
      if (!b64 || !canvas) return;
      gotFrame = true;
      setState("live");
      setNote("live capture");
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
      const mime = (data.mime as string) || (data.png_b64 ? "image/png" : "image/jpeg");
      void paintFrame(canvas, new Blob([bytes], { type: mime })).then(() => {
        if (alive) setFrames((n) => n + 1);
      });
    };

    sock.onerror = () => {
      if (!alive) return;
      window.clearTimeout(timer);
      if (!gotFrame) {
        setState("down");
        setNote("capture socket error");
        idleLoop("capture socket error");
      }
    };

    sock.onclose = () => {
      if (!alive) return;
      window.clearTimeout(timer);
      if (!gotFrame) {
        setState("down");
        setNote("capture closed");
        idleLoop("capture closed");
      }
    };

    return () => {
      alive = false;
      window.clearTimeout(timer);
      cancelAnimationFrame(raf);
      try {
        sock?.close();
      } catch {
        /* already closed */
      }
    };
  }, [target.hwnd, target.pid, target.title, target.x, target.y, target.w, target.h, target.fps, target.kind, kind]);

  useEffect(() => {
    if (!active) return;
    const c = canvasRef.current;
    if (!c) return;
    c.width = 960;
    c.height = 540;
  }, [active]);

  return (
    <div className="screen">
      <canvas ref={canvasRef} width={960} height={540} aria-label={`live ${kind} surface`} />
      {state !== "live" && <div className="screen-note">{note}</div>}
      {state === "live" && <div className="screen-note">{frames} frames · {target.kind}</div>}
    </div>
  );
}