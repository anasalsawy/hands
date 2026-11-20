import { useCallback, useEffect, useMemo, useState } from "react";
import { apiBase, hasConnection, isServedByBackend, personaKey, setApiBase, setPersonaKey, setSessionKey } from "./lib/connection";
import { probe, systemApi, type ComputerStatus, type LinkInfo, type VoiceStatus } from "./lib/client";
import { useDeck } from "./hooks/useDeck";
import { ConnectView } from "./components/ConnectView";
import { DeckView } from "./components/DeckView";
import { MotorView } from "./components/MotorView";
import { ChatView } from "./components/ChatView";
import { VoiceView } from "./components/VoiceView";
import { PhoneView } from "./components/PhoneView";
import { BrowserView } from "./components/BrowserView";
import { BrainView, fetchBoot, type Boot } from "./components/BrainView";

type Tab = "courts" | "motor" | "chat" | "voice" | "phone" | "browser" | "brain";

const TABS: { id: Tab; label: string }[] = [
  { id: "courts", label: "Courts" },
  { id: "motor", label: "Motor" },
  { id: "chat", label: "Chat" },
  { id: "voice", label: "Voice" },
  { id: "phone", label: "Phone" },
  { id: "browser", label: "Browser" },
  { id: "brain", label: "Persona" },
];

interface Health {
  deck: boolean;
  link: string;
  computer: string;
  voice: string;
  models: number;
}

export default function App() {
  const embedded = isServedByBackend();
  const [boot, setBoot] = useState<Boot | null>(null);
  const [booting, setBooting] = useState(embedded);
  const [connected, setConnected] = useState(embedded ? false : hasConnection());
  const [tab, setTab] = useState<Tab>("courts");
  const [health, setHealth] = useState<Health | null>(null);
  const [editing, setEditing] = useState(false);
  const [draftBase, setDraftBase] = useState(apiBase());
  const [draftKey, setDraftKey] = useState(personaKey());
  const [note, setNote] = useState("");
  const deck = useDeck();

  // Served by the local Hands backend: it hands its own window the local key.
  const loadBoot = useCallback(async () => {
    const b = await fetchBoot();
    if (b) {
      setSessionKey(b.persona_key);
      setBoot(b);
      setConnected(true);
      void deck.refresh();
    }
    setBooting(false);
    return b;
  }, [deck]);

  useEffect(() => {
    if (embedded) void loadBoot();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [embedded]);

  const checkHealth = useCallback(async () => {
    if (!hasConnection()) {
      setHealth(null);
      return;
    }
    const [models, link, computer, voice] = await Promise.all([
      probe<{ data: { id: string }[] }>("/v1/models"),
      probe<{ links: Record<string, LinkInfo> }>("/v1/link"),
      probe<ComputerStatus>("/v1/computer"),
      probe<VoiceStatus>("/v1/voice-status"),
    ]);
    const links = link.data?.links ?? {};
    setHealth({
      deck: deck.link === "live",
      link: Object.keys(links).length ? Object.keys(links).join(",") : models.ok ? "none registered" : "offline",
      computer: computer.ok ? `${computer.data?.tools?.length ?? 0} tools` : "offline",
      voice: voice.ok ? (voice.data?.sidecar?.running ? "sidecar up" : "server") : "offline",
      models: models.data?.data.length ?? 0,
    });
  }, [deck.link]);

  useEffect(() => {
    void checkHealth();
    const t = window.setInterval(() => void checkHealth(), 10000);
    return () => window.clearInterval(t);
  }, [checkHealth]);

  const runningCount = useMemo(() => deck.courts.filter((c) => c.running).length, [deck.courts]);

  if (embedded && booting) {
    return (
      <div className="app">
        <div className="body">
          <div className="empty">
            <p>Starting Hands…</p>
          </div>
        </div>
      </div>
    );
  }

  if (embedded && boot && !boot.brain.configured) {
    return (
      <div className="app">
        <div className="bar">
          <span className="brand">
            <span className="mark" />
            Hands
          </span>
        </div>
        <div className="body">
          <BrainView boot={boot} first onSaved={() => void loadBoot()} />
        </div>
        <div className="footer">
          <span>no Persona connected yet</span>
        </div>
      </div>
    );
  }

  if (!connected) {
    return (
      <div className="app">
        <div className="bar">
          <span className="brand">
            <span className="mark" />
            Hands
          </span>
        </div>
        <div className="body">
          <ConnectView
            onDone={() => {
              setConnected(true);
              void deck.refresh();
            }}
          />
        </div>
        <div className="footer">
          <span>not connected</span>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="bar">
        <span className="brand">
          <span className="mark" />
          Hands
        </span>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className="tab" data-on={tab === t.id ? "1" : "0"} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        <span className="pill">
          <span className="dot" data-s={deck.link === "live" ? "live" : deck.link === "down" ? "down" : "idle"} />
          {deck.snapshot ? `${deck.snapshot.count} cards` : "no data"} · {runningCount} running
        </span>
        <span className="card-meta" title="backend">
          {apiBase()}
        </span>
        {!embedded && (
        <button
          className="btn"
          data-size="sm"
          onClick={() => {
            setDraftBase(apiBase());
            setDraftKey(personaKey());
            setEditing((v) => !v);
          }}
        >
          {editing ? "Close" : "Settings"}
        </button>
        )}
      </div>

      <div className="body">
        {editing && (
          <div className="panel" style={{ marginBottom: 12 }}>
            <h3>Backend</h3>
            <div className="form">
              <span className="label">URL</span>
              <input value={draftBase} onChange={(e) => setDraftBase(e.target.value)} />
              <span className="label">Persona key</span>
              <input
                type="password"
                autoComplete="off"
                value={draftKey}
                onChange={(e) => setDraftKey(e.target.value)}
              />
            </div>
            <div className="row">
              <button
                className="btn"
                data-kind="primary"
                onClick={() => {
                  setApiBase(draftBase);
                  setPersonaKey(draftKey);
                  setEditing(false);
                  void deck.refresh();
                  void checkHealth();
                }}
              >
                Save
              </button>
              <button
                className="btn"
                onClick={() =>
                  void systemApi
                    .models()
                    .then((m) => setNote(`reachable · ${m.data.length} models`))
                    .catch((err: Error) => setNote(`unreachable: ${err.message}`))
                }
              >
                Test
              </button>
              <span className="card-meta">{note}</span>
            </div>
          </div>
        )}
        {tab === "courts" && <DeckView deck={deck} missing={boot?.missing ?? []} />}
        {tab === "motor" && <MotorView deck={deck} />}
        {tab === "chat" && <ChatView />}
        {tab === "voice" && <VoiceView />}
        {tab === "phone" && <PhoneView />}
        {tab === "browser" && <BrowserView />}
        {tab === "brain" && boot && <BrainView boot={boot} first={false} onSaved={() => void loadBoot()} />}
        {tab === "brain" && !boot && (
          <div className="empty">
            <p>Persona settings are managed by the local Hands app. Open Hands from its own window.</p>
          </div>
        )}
      </div>

      <div className="footer">
        <span>{apiBase()}</span>
        <span>deck {health?.deck ? "live" : "down"}</span>
        <span>links {health?.link ?? "—"}</span>
        <span>computer {health?.computer ?? "—"}</span>
        <span>voice {health?.voice ?? "—"}</span>
        <span>{health?.models ?? 0} models</span>
        <span className="spacer" />
        <span>{deck.error || ""}</span>
      </div>
    </div>
  );
}