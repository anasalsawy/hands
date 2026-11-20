// Line icons in the Nivo style: 1.75px strokes, rounded, 24px grid.
export type IconName = "tasks" | "files" | "orb" | "chat" | "settings" | "cpu" | "phone" | "globe" | "more";

const PATHS: Record<IconName, string> = {
  tasks: "M4 5h7v7H4zM13 5h7v4h-7zM13 11h7v8h-7zM4 14h7v5H4z",
  files: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  orb: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM8.5 12.5c1.2-2.6 5.8-2.6 7 0M9 9.5h.01M15 9.5h.01",
  chat: "M4 5h16v11H9l-5 4z",
  settings: "M5 6h9M18 6h1M5 12h3M12 12h7M5 18h11M20 18h-1M16 4v4M10 10v4M18 16v4",
  cpu: "M7 7h10v10H7zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4",
  phone: "M8 3h8a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM11 18h2",
  globe: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18",
  more: "M5 12h.01M12 12h.01M19 12h.01",
};

export function Icon({ name }: { name: IconName }) {
  return (
    <svg className="ico" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}
