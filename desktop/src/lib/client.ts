// The Hands backend client. Thin, typed, and deliberately pixel-free: it binds
// to the contract `persona/court.py` and `persona/api/app.py` already expose,
// so the visual layer stays swappable.
//
//   Courts  GET/POST /v1/courts · GET/DELETE /v1/courts/{id}
//           POST /v1/courts/{id}/spawn | goal | cancel | voice
//           GET  /v1/courts/{id}/events (SSE) · WS /v1/court-events
//   Motor   GET /v1/hands · GET /v1/computer · GET /v1/link · GET /v1/models
//   Voice   GET /v1/voice-status · WS /v1/voice
//   Phone   GET /v1/calls · POST /v1/calls/outbound · POST /v1/calls/{sid}/end
//   Chat    POST /v1/chat/completions (SSE) · GET /v1/timeline
//   Surfaces GET /v1/computer/windows · WS /v1/computer/stream · WS /v1/browser/stream

import { authHeaders, httpUrl, keyedUrl, keyedWsUrl, personaKey } from "./connection";

export type CourtKind = "desktop" | "window" | "region" | "browser";
export type CourtStatus =
  | "empty"
  | "spawning"
  | "running"
  | "waiting"
  | "done"
  | "failed"
  | "stopped";

export interface CourtTarget {
  kind: CourtKind;
  hwnd?: number;
  pid?: number;
  title?: string;
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  fps?: number;
  label?: string;
}

export interface Court {
  id: string;
  persona_id: number;
  title: string;
  kind: CourtKind;
  target: CourtTarget;
  lane: string;
  link: string;
  goal: string;
  status: CourtStatus;
  error: string;
  running: boolean;
  created_at: number;
  updated_at: number;
  meta: Record<string, unknown>;
}

export interface DeckSnapshot {
  courts: Court[];
  count: number;
  running: number;
  kinds: CourtKind[];
  statuses: CourtStatus[];
}

export interface CourtEvent {
  kind: string;
  court: Court;
  at: number;
  lane?: string;
  motor?: string;
  goal?: string;
  status?: string;
  error?: string;
  changed?: string;
  action?: string;
  detail?: string;
}

export interface LinkInfo {
  name: string;
  strategy: string;
  agent_type?: string;
  supports_steer?: boolean;
  supports_stop?: boolean;
  active_lanes?: number;
  total_lanes?: number;
}

export interface HandsStatus {
  connected: boolean;
  places: Record<string, { host: string; lane: string; tools: string[] }>;
  host: string;
  tools: string[];
}

export interface ComputerStatus {
  ok?: boolean;
  enabled?: boolean;
  browser?: boolean;
  desktop?: boolean;
  tools?: string[];
  platform?: string;
  denied?: boolean;
}

export interface WindowRow {
  hwnd: number;
  title: string;
  pid?: number;
  process?: string;
  rect?: { x: number; y: number; w: number; h: number };
}

export interface VoiceStatus {
  ok: boolean;
  websocket: string;
  web_ui?: string;
  sidecar?: { running: boolean; url: string; pid: number | null };
  persona_speech?: Record<string, unknown>;
  pipeline?: Record<string, unknown>;
}

export interface CallRow {
  sid: string;
  to: string;
  started_at: number;
  status?: string;
}

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(httpUrl(path), {
      ...init,
      headers: authHeaders((init?.headers as Record<string, string>) || {}),
    });
  } catch (err) {
    throw new ApiError(0, `backend unreachable (${err instanceof Error ? err.message : "network"})`);
  }
  const text = await res.text();
  let data: unknown = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }
  }
  if (!res.ok) {
    const body = data as { error?: { message?: string }; detail?: string; message?: string };
    const msg = body?.error?.message || body?.detail || body?.message || `HTTP ${res.status}`;
    throw new ApiError(res.status, String(msg));
  }
  return data as T;
}

/** Health probe: never throws, reports what it saw. Used by the status bar. */
export async function probe<T>(path: string): Promise<{ ok: boolean; status: number; data?: T; error?: string }> {
  try {
    const data = await call<T>(path);
    return { ok: true, status: 200, data };
  } catch (err) {
    const e = err as ApiError;
    return { ok: false, status: e.status || 0, error: e.message };
  }
}

export const courtsApi = {
  snapshot: () => call<DeckSnapshot>("/v1/courts"),
  list: async () => (await call<DeckSnapshot>("/v1/courts")).courts,
  get: (id: string) => call<Court>(`/v1/courts/${id}`),
  create: (opts: { title?: string; kind?: CourtKind; target?: CourtTarget; meta?: Record<string, unknown> } = {}) =>
    call<Court>("/v1/courts", { method: "POST", body: JSON.stringify(opts) }),
  remove: (id: string) => call<{ ok: boolean }>(`/v1/courts/${id}`, { method: "DELETE" }),
  /** Start a real desktop instance for this card: its own lane + engine process. */
  spawn: (id: string, opts: { link?: string; goal?: string } = {}) =>
    call<Court>(`/v1/courts/${id}/spawn`, { method: "POST", body: JSON.stringify(opts) }),
  send: (id: string, goal: string) =>
    call<Court>(`/v1/courts/${id}/goal`, { method: "POST", body: JSON.stringify({ goal }) }),
  cancel: (id: string) => call<Court>(`/v1/courts/${id}/cancel`, { method: "POST", body: JSON.stringify({}) }),
  voice: (id: string, action: "speak" | "listen" | "stop" | "status", text = "") =>
    call<Record<string, unknown>>(`/v1/courts/${id}/voice`, {
      method: "POST",
      body: JSON.stringify({ action, text }),
    }),
  /** One socket for every card. `/v1/courts/{id}/events` (SSE) is the fallback. */
  socket: (): WebSocket => {
    const ws = new WebSocket(keyedWsUrl("/v1/court-events"));
    ws.onmessage = null;
    return ws;
  },
  eventsUrl: (id: string) => keyedUrl(`/v1/courts/${id}/events`),
};

export const systemApi = {
  models: () => call<{ data: { id: string }[] }>("/v1/models"),
  link: () => call<{ links: Record<string, LinkInfo> }>("/v1/link"),
  hands: () => call<HandsStatus>("/v1/hands"),
  computer: () => call<ComputerStatus>("/v1/computer"),
  windows: () => call<{ ok: boolean; windows: WindowRow[] }>("/v1/computer/windows"),
  capture: () => call<{ ok: boolean; [k: string]: unknown }>("/v1/computer/capture"),
  voice: () => call<VoiceStatus>("/v1/voice-status"),
  calls: () => call<{ calls: CallRow[] }>("/v1/calls"),
  placeCall: (to: string, hint = "") =>
    call<Record<string, unknown>>("/v1/calls/outbound", { method: "POST", body: JSON.stringify({ to, hint }) }),
  endCall: (sid: string) => call<Record<string, unknown>>(`/v1/calls/${sid}/end`, { method: "POST", body: "{}" }),
  timeline: (after = 0, limit = 120) =>
    call<{ events: Record<string, unknown>[]; next_cursor: number }>(
      `/v1/timeline?after=${after}&limit=${limit}`,
    ),
  computerStreamUrl: (target: CourtTarget) =>
    keyedWsUrl("/v1/computer/stream", {
      kind: target.kind === "browser" ? "desktop" : target.kind || "desktop",
      hwnd: target.hwnd,
      pid: target.pid,
      title: target.title,
      x: target.x,
      y: target.y,
      w: target.w,
      h: target.h,
      fps: target.fps ?? 6,
      exclude_self: 1,
    }),
  browserStreamUrl: (lane?: string) => keyedWsUrl("/v1/browser/stream", { lane: lane || undefined }),
  voiceSocketUrl: (lane = "L") => keyedWsUrl("/v1/voice", { lane }),
  hasKey: () => Boolean(personaKey()),
};

export type { Court as CourtView };