// Where the app talks to. Hands is a standalone exe: it is never served by the
// backend, so the origin has to be stated. Build-time default comes from
// VITE_HANDS_API_BASE; the value can be changed at runtime on the connect
// screen and is kept in the webview's own storage.
//
// The persona key is a secret. It is only ever kept in local webview storage
// (never in a tracked file, never in the bundle) and is sent as a bearer header.

const BASE_KEY = "hands.api_base";
const KEY_KEY = "hands.persona_key";

function builtBase(): string {
  return (import.meta.env?.VITE_HANDS_API_BASE || "").trim();
}

function safeGet(k: string): string {
  try {
    return localStorage.getItem(k) || "";
  } catch {
    return "";
  }
}

function safeSet(k: string, v: string): void {
  try {
    if (v) localStorage.setItem(k, v);
    else localStorage.removeItem(k);
  } catch {
    /* storage unavailable (private mode): in-memory only */
  }
}

/** A shell loads from tauri://localhost, so it has no same-origin backend. */
export function isStandaloneShell(): boolean {
  if (typeof window === "undefined") return false;
  const { protocol, hostname } = window.location;
  return !/^https?:$/.test(protocol) || hostname.endsWith(".localhost");
}

export function apiBase(): string {
  const stored = safeGet(BASE_KEY).trim().replace(/\/+$/, "");
  return (stored || builtBase() || "http://127.0.0.1:8800").replace(/\/+$/, "");
}

export function setApiBase(base: string): void {
  safeSet(BASE_KEY, base.trim().replace(/\/+$/, ""));
}

export function personaKey(): string {
  return safeGet(KEY_KEY) || (import.meta.env?.VITE_HANDS_PERSONA_KEY || "").trim();
}

export function setPersonaKey(key: string): void {
  safeSet(KEY_KEY, key.trim());
}

export function hasConnection(): boolean {
  return Boolean(apiBase() && personaKey());
}

/** Absolute http(s) URL for an API path. */
export function httpUrl(path: string, query?: Record<string, string | number | undefined>): string {
  const base = apiBase();
  const origin = /^https?:/.test(base) ? base : "http://127.0.0.1:8800";
  const u = new URL(path.startsWith("/") ? path : `/${path}`, origin);
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== undefined && v !== "") u.searchParams.set(k, String(v));
  }
  return u.toString();
}

/** Same, with the persona key in the query (for EventSource / WebSocket). */
export function keyedUrl(path: string, query?: Record<string, string | number | undefined>): string {
  const key = personaKey();
  return httpUrl(path, { ...(query || {}), ...(key ? { key } : {}) });
}

/** WebSocket variant of `keyedUrl`. */
export function keyedWsUrl(path: string, query?: Record<string, string | number | undefined>): string {
  const u = new URL(keyedUrl(path, query));
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  return u.toString();
}

export function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const key = personaKey();
  return {
    "content-type": "application/json",
    ...(key ? { authorization: `Bearer ${key}` } : {}),
    ...(extra || {}),
  };
}