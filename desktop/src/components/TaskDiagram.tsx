// A card's picture of the work: the goal on the left, the steps this lane took
// as stations along a line, and where it is now on the right. It is drawn from
// the lane's real tool trace (court.meta.trace), so it is always what happened.
import { useEffect, useState } from "react";
import type { Court } from "../lib/client";

interface Step {
  t: number;
  tool: string;
  arg: string;
  ok: boolean;
  out: string;
}

type Cat = "app" | "pointer" | "keys" | "eye" | "files" | "shell" | "web";

const COLORS: Record<Cat, string> = {
  app: "#a78bfa",
  pointer: "#34d399",
  keys: "#60a5fa",
  eye: "#94a3b8",
  files: "#f59e0b",
  shell: "#f472b6",
  web: "#22d3ee",
};

function norm(tool: string): string {
  return tool.replace(/^(computer|cloud|browser)_/, (m) => (m === "browser_" ? "web_" : ""));
}

function describe(step: Step): { cat: Cat; verb: string; sub: string } {
  const n = norm(step.tool);
  let args: Record<string, unknown> = {};
  try {
    args = step.arg ? (JSON.parse(step.arg) as Record<string, unknown>) : {};
  } catch {
    /* truncated arg text: show it raw below */
  }
  const s = (k: string) => (typeof args[k] === "string" ? (args[k] as string) : "");
  const short = (v: string, n2 = 22) => (v.length > n2 ? v.slice(0, n2 - 1) + "…" : v);
  if (n === "win_launch" || n === "win_bind") return { cat: "app", verb: "open", sub: short(s("command") || s("title") || s("app")) };
  if (n === "win_list") return { cat: "eye", verb: "find windows", sub: "" };
  if (n === "win_click") return { cat: "pointer", verb: args.count === 2 ? "double-click" : "click", sub: `${Math.round(Number(args.x))}, ${Math.round(Number(args.y))}` };
  if (n === "win_drag") return { cat: "pointer", verb: "drag", sub: "" };
  if (n === "win_scroll") return { cat: "pointer", verb: "scroll", sub: String(args.dy ?? "") };
  if (n === "win_type") return { cat: "keys", verb: "type", sub: short(s("text")) };
  if (n === "win_hotkey") return { cat: "keys", verb: "keys", sub: s("keys") };
  if (n === "win_screenshot" || n === "win_wait" || n === "screen_text") return { cat: "eye", verb: "look", sub: "" };
  if (n === "shell") return { cat: "shell", verb: "run", sub: short(s("command"), 26) };
  if (/file/.test(n)) return { cat: "files", verb: n.replace(/_/g, " "), sub: short(s("path") || s("pattern"), 24) };
  if (n.startsWith("web_")) {
    const w = n.slice(4);
    if (/navigate|goto|open/.test(w)) return { cat: "web", verb: "go to", sub: short(s("url").replace(/^https?:\/\//, ""), 24) };
    if (/click/.test(w)) return { cat: "web", verb: "click", sub: short(s("selector") || s("text") || s("element"), 20) };
    if (/type|fill|input/.test(w)) return { cat: "web", verb: "type", sub: short(s("text") || s("value"), 20) };
    return { cat: "web", verb: w.replace(/_/g, " "), sub: "" };
  }
  if (n.startsWith("ocu_")) return { cat: "eye", verb: n.slice(4).replace(/_/g, " "), sub: "" };
  return { cat: "eye", verb: n.replace(/_/g, " "), sub: short(step.arg, 20) };
}

function Glyph({ cat, x, y, c }: { cat: Cat; x: number; y: number; c: string }) {
  const p = { stroke: c, strokeWidth: 1.8, fill: "none", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  return (
    <g transform={`translate(${x - 8} ${y - 8})`}>
      {cat === "app" && <rect x="1" y="2" width="14" height="12" rx="2" {...p} />}
      {cat === "pointer" && <path d="M3 2l9 6-4 1-2 4z" {...p} />}
      {cat === "keys" && <path d="M2 5h12v7H2zM5 9h6" {...p} />}
      {cat === "eye" && <path d="M1 8s3-5 7-5 7 5 7 5-3 5-7 5-7-5-7-5zM8 6.5v3" {...p} />}
      {cat === "files" && <path d="M3 2h6l4 4v8H3z" {...p} />}
      {cat === "shell" && <path d="M2 4l5 4-5 4M8 13h6" {...p} />}
      {cat === "web" && <path d="M8 1.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13zM1.5 8h13M8 1.5c2 2 2 11 0 13M8 1.5c-2 2-2 11 0 13" {...p} />}
    </g>
  );
}

const W = 640;
const H = 300;
const SHOW = 6;

export function TaskDiagram({ court }: { court: Court }) {
  const steps = ((court.meta?.trace as Step[] | undefined) ?? []).slice();
  const total = steps.length;
  const shown = steps.slice(-SHOW);
  const live = ["spawning", "running", "waiting"].includes(court.status);
  const seen = typeof court.meta?.l_vision === "string" ? (court.meta.l_vision as string) : "";
  const part = typeof court.meta?.my_part === "string" ? (court.meta.my_part as string) : "";
  const goal = part || court.goal || court.title;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!live) return;
    const t = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(t);
  }, [live]);

  const x0 = 70;
  const x1 = W - 70;
  const gap = shown.length > 0 ? (x1 - x0) / shown.length : x1 - x0;
  const yAt = (i: number) => 150 + Math.sin(i * 1.3) * 34;
  const pts = [{ x: 34, y: 150 }, ...shown.map((_, i) => ({ x: x0 + gap * (i + 0.5), y: yAt(i) })), { x: W - 34, y: 150 }];
  const path = pts.map((p, i) => (i === 0 ? `M${p.x} ${p.y}` : `L${p.x} ${p.y}`)).join(" ");
  const endColor = court.status === "failed" ? "#f43f5e" : court.status === "done" ? "#34d399" : "#60a5fa";
  const elapsed = court.created_at ? Math.max(0, Math.round(Date.now() / 1000 - court.created_at)) : 0;

  return (
    <div className="diagram" data-st={court.status}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Task diagram: ${goal}`}>
        <defs>
          <linearGradient id={`g-${court.id}`} x1="0" x2="1">
            <stop offset="0" stopColor="#334155" />
            <stop offset="1" stopColor={endColor} />
          </linearGradient>
        </defs>
        <foreignObject x="14" y="10" width={W - 28} height="64">
          <div className="dg-goal">{goal}</div>
        </foreignObject>
        <path d={path} className={live ? "dg-line flow" : "dg-line"} stroke={`url(#g-${court.id})`} />
        <g>
          <circle cx={34} cy={150} r={9} fill="#0f172a" stroke="#64748b" strokeWidth={1.6} />
          <path d="M31 146v9M31 146h8l-2 2.5 2 2.5h-8" stroke="#94a3b8" strokeWidth={1.4} fill="none" />
        </g>
        {shown.map((st, i) => {
          const d = describe(st);
          const p = pts[i + 1];
          const col = st.ok ? COLORS[d.cat] : "#f43f5e";
          const last = i === shown.length - 1;
          const above = i % 2 === 0;
          return (
            <g key={`${st.t}-${i}`} className="dg-step" style={{ animationDelay: `${i * 40}ms` }}>
              {last && live && <circle cx={p.x} cy={p.y} r={22} className="dg-pulse" stroke={col} />}
              <circle cx={p.x} cy={p.y} r={16} fill="#0b1220" stroke={col} strokeWidth={2} />
              <Glyph cat={d.cat} x={p.x} y={p.y} c={col} />
              <text x={p.x} y={above ? p.y - 28 : p.y + 36} textAnchor="middle" className="dg-verb" fill={col}>
                {d.verb}
              </text>
              {d.sub && (
                <text x={p.x} y={above ? p.y - 41 : p.y + 49} textAnchor="middle" className="dg-sub">
                  {d.sub}
                </text>
              )}
            </g>
          );
        })}
        <g>
          <circle cx={W - 34} cy={150} r={14} fill="#0b1220" stroke={endColor} strokeWidth={2} />
          {court.status === "done" && <path d={`M${W - 41} 150l5 5 9-10`} stroke={endColor} strokeWidth={2.4} fill="none" />}
          {court.status === "failed" && <path d={`M${W - 40} 144l12 12M${W - 28} 144l-12 12`} stroke={endColor} strokeWidth={2.4} />}
          {live && <circle cx={W - 34} cy={150} r={4} fill={endColor} className="dg-blink" />}
        </g>
        {total === 0 && (
          <text x={W / 2} y={150} textAnchor="middle" className="dg-sub">
            {live ? "planning the first step…" : court.status === "empty" ? "no task yet" : "no steps recorded"}
          </text>
        )}
        <text x={14} y={H - 12} className="dg-meta">
          {total} step{total === 1 ? "" : "s"}
          {live ? ` · ${elapsed}s` : ""}
        </text>
      </svg>
      {seen && <div className="dg-seen">sees: {seen}</div>}
    </div>
  );
}
