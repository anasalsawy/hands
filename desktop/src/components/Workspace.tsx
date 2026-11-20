// What the cards are making: the workspace files, shown as they appear.
// Used inside a card (the files that card's run produced) and as its own tab.
import { useCallback, useEffect, useState } from "react";
import { authHeaders, httpUrl, personaKey } from "../lib/connection";
import type { Court } from "../lib/client";

interface FileInfo {
  path: string;
  size: number;
  mtime: number;
}

const TEXT = /\.(txt|md|markdown|csv|tsv|json|jsonl|py|js|ts|tsx|jsx|css|yml|yaml|toml|ini|log|sh|sql|xml|rs|go|java|c|h|cpp|rb|php)$/i;
const IMG = /\.(png|jpe?g|gif|webp|svg|bmp)$/i;

export function fileUrl(path: string, download = false): string {
  return httpUrl("/hands/workspace/file", { path, key: personaKey(), ...(download ? { download: 1 } : {}) });
}

export function FilePreview({ path, tall }: { path: string; tall?: boolean }) {
  const [text, setText] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const h = tall ? "62vh" : "260px";

  useEffect(() => {
    setText(null);
    setErr("");
    if (!TEXT.test(path)) return;
    let live = true;
    void fetch(fileUrl(path), { headers: authHeaders() })
      .then(async (r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        const t = await r.text();
        if (live) setText(t.length > 200_000 ? t.slice(0, 200_000) + "\n… (shortened)" : t);
      })
      .catch((e) => live && setErr(e instanceof Error ? e.message : "could not open"));
    return () => {
      live = false;
    };
  }, [path]);

  if (/\.html?$/i.test(path))
    return <iframe className="ws-frame" title={path} sandbox="allow-scripts" src={fileUrl(path)} style={{ height: h }} />;
  if (/\.pdf$/i.test(path)) return <iframe className="ws-frame" title={path} src={fileUrl(path)} style={{ height: h }} />;
  if (IMG.test(path)) return <img className="ws-img" alt={path} src={fileUrl(path)} style={{ maxHeight: h }} />;
  if (TEXT.test(path))
    return (
      <pre className="ws-text" style={{ maxHeight: h }}>
        {err ? `could not open (${err})` : text ?? "opening…"}
      </pre>
    );
  return (
    <div className="ws-other">
      <a className="btn" href={fileUrl(path, true)} download>
        Download {path.split("/").pop()}
      </a>
    </div>
  );
}

function useFiles(active: boolean, every = 2500) {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const load = useCallback(async () => {
    try {
      const r = await fetch(httpUrl("/hands/workspace"), { headers: authHeaders() });
      if (r.ok) setFiles(((await r.json()) as { files: FileInfo[] }).files);
    } catch {
      /* offline: keep what we have */
    }
  }, []);
  useEffect(() => {
    void load();
    if (!active) return;
    const t = window.setInterval(() => void load(), every);
    return () => window.clearInterval(t);
  }, [active, every, load]);
  return { files, reload: load };
}

/** The files one card's run made, with the newest open. */
export function WorkspacePeek({ court }: { court: Court }) {
  const live = ["spawning", "running", "waiting"].includes(court.status);
  const { files } = useFiles(live);
  const [pick, setPick] = useState("");
  const [tall, setTall] = useState(false);
  const since = (court.created_at || 0) - 2;
  const mine = files.filter((f) => f.mtime >= since).slice(0, 8);
  const current = pick && mine.some((f) => f.path === pick) ? pick : mine[0]?.path || "";
  if (!mine.length)
    return <div className="ws-empty">{live ? "Nothing made yet. Files appear here as the card writes them." : "This run made no files."}</div>;
  return (
    <div className="ws-peek">
      <div className="ws-chips">
        {mine.map((f) => (
          <button key={f.path} className="ws-chip" data-on={f.path === current ? "1" : "0"} onClick={() => setPick(f.path)}>
            {f.path}
          </button>
        ))}
        <span className="spacer" />
        <button className="btn" data-size="sm" onClick={() => setTall((v) => !v)}>
          {tall ? "smaller" : "bigger"}
        </button>
        <a className="btn" data-size="sm" href={fileUrl(current, true)} download>
          save
        </a>
      </div>
      <FilePreview path={current} tall={tall} />
    </div>
  );
}

/** The whole workspace as a tab. */
export function FilesView() {
  const { files, reload } = useFiles(true, 4000);
  const [pick, setPick] = useState("");
  const current = pick || files[0]?.path || "";
  return (
    <div className="split">
      <div className="panel">
        <h3>Workspace files</h3>
        <div className="row">
          <button className="btn" onClick={() => void reload()}>
            Refresh
          </button>
          <span className="card-meta">{files.length} file{files.length === 1 ? "" : "s"}, newest first</span>
        </div>
        <div className="ws-list">
          {files.length === 0 && <div className="ws-empty">Empty. Ask a card to make something.</div>}
          {files.map((f) => (
            <button key={f.path} className="ws-item" data-on={f.path === current ? "1" : "0"} onClick={() => setPick(f.path)}>
              <span>{f.path}</span>
              <span className="card-meta">{f.size < 1024 ? `${f.size} B` : `${Math.round(f.size / 1024)} KB`}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="panel">
        <h3>{current || "Preview"}</h3>
        {current && <FilePreview path={current} tall />}
        {current && (
          <a className="btn" href={fileUrl(current, true)} download>
            Download
          </a>
        )}
      </div>
    </div>
  );
}
