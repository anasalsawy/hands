export interface FrameBitmap {
  width: number;
  height: number;
  close(): void;
}

// At most one decode and one pending frame, even when capture outruns paint.
export function latestFrames<T extends FrameBitmap>({ decode, paint, onError }: {
  decode: (blob: Blob) => Promise<T>;
  paint: (bitmap: T) => void;
  onError: (error: unknown) => void;
}) {
  let pending: Blob | null = null;
  let running = false;
  let disposed = false;

  async function drain() {
    running = true;
    try {
      while (pending && !disposed) {
        const blob = pending;
        pending = null;
        let bitmap: T | undefined;
        try {
          bitmap = await decode(blob);
          if (!disposed) paint(bitmap);
        } catch (error) {
          if (!disposed) onError(error);
        } finally {
          bitmap?.close();
        }
      }
    } finally {
      running = false;
    }
  }

  return {
    push(blob: Blob) {
      if (disposed) return;
      pending = blob;
      if (!running) void drain();
    },
    dispose() {
      disposed = true;
      pending = null;
    },
  };
}

export function frameBlob(data: Record<string, unknown>): Blob | null {
  const b64 = data.screenshot_png_b64 || data.jpeg_b64 || data.png_b64 || data.data_b64;
  if (typeof b64 !== "string" || !b64) return null;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const mime = typeof data.mime === "string" ? data.mime :
    data.screenshot_png_b64 || data.png_b64 ? "image/png" : "image/jpeg";
  return new Blob([bytes], { type: mime });
}

export function paintBitmap(canvas: HTMLCanvasElement, bitmap: ImageBitmap) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Math.min(canvas.width / bitmap.width, canvas.height / bitmap.height);
  const dw = Math.round(bitmap.width * scale);
  const dh = Math.round(bitmap.height * scale);
  ctx.drawImage(bitmap, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
}
