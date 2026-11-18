import type { CourtEvent } from "./client";

export interface DeckLogRow {
  id: string;
  at: number;
  kind: string;
  courtId: string;
  text: string;
}

function describe(ev: CourtEvent): string {
  const c = ev.court;
  switch (ev.kind) {
    case "court.created": return `card created · ${c.title}`;
    case "court.removed": return `card removed · ${c.title}`;
    case "court.spawned":
      return `instance started · ${c.title} · ${ev.motor || "link"}${ev.lane ? ` · ${ev.lane}` : ""}`;
    case "court.attached": return `bound ${ev.lane || "lane"} · ${c.title}`;
    case "court.goal": return `goal sent · ${c.title}`;
    case "court.cancelled": return `stopped · ${c.title}`;
    case "court.voice": return `voice ${ev.action || ""} · ${c.title}`;
    case "court.status": return `${ev.status || ""} · ${c.title}${ev.error ? ` · ${ev.error}` : ""}`;
    default: return `${ev.kind} · ${c?.title ?? ""}`;
  }
}

export function appendDeckEvent(ev: CourtEvent, sequence: number) {
  const now = Date.now();
  const row: DeckLogRow = {
    id: `${now}-${sequence}`,
    at: ev.at ? ev.at * 1000 : now,
    kind: ev.kind,
    courtId: ev.court?.id || "",
    text: describe(ev),
  };
  return (rows: DeckLogRow[]): DeckLogRow[] => [row, ...rows].slice(0, 60);
}
