// Conversation layer. One turn = POST /v1/chat/completions with stream:true,
// reading the OpenAI-shaped SSE deltas the persona emits. The server owns the
// history, so only the newest user message is sent. `/v1/interrupt` is the
// steering path while a reply is still in flight, and `/v1/timeline` loads what
// the persona already recorded.
import { useCallback, useEffect, useRef, useState } from "react";
import { apiBase, authHeaders, httpUrl, personaKey } from "../lib/connection";

type Msg = { role: "user" | "assistant"; text: string; tools?: string[] };

interface StreamChunk {
  choices?: { delta?: { content?: string } }[];
  sawii_tool?: { name?: string; id?: string };
  sawii_inner?: { from?: string; kind?: string; text?: string };
  error?: { message?: string };
}

export function ChatView() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [historyNote, setHistoryNote] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [msgs]);

  const loadHistory = useCallback(async () => {
    setHistoryNote("");
    try {
      const res = await fetch(httpUrl("/v1/timeline?after=0&limit=60"), { headers: authHeaders() });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { events: Record<string, unknown>[] };
      const rows: Msg[] = [];
      for (const ev of data.events || []) {
        const role = String(ev.role || ev.type || "");
        const text = String(ev.text || ev.content || "").trim();
        if (!text) continue;
        if (role === "user" || role === "message_user") rows.push({ role: "user", text });
        else if (role === "assistant" || role === "message_assistant") rows.push({ role: "assistant", text });
      }
      setMsgs(rows.slice(-40));
      setHistoryNote(`${(data.events || []).length} timeline rows read · ${rows.length} messages shown`);
    } catch (err) {
      setHistoryNote(`timeline unreadable: ${err instanceof Error ? err.message : "failed"}`);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  const send = async () => {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    setMsgs((m) => [...m, { role: "user", text }, { role: "assistant", text: "" }]);
    setBusy(true);
    setStatus("streaming");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch(httpUrl("/v1/chat/completions"), {
        method: "POST",
        headers: authHeaders(),
        signal: ctrl.signal,
        body: JSON.stringify({
          model: "sawii-persona",
          stream: true,
          messages: [{ role: "user", content: text }],
        }),
      });
      if (!res.ok || !res.body) {
        const raw = await res.text().catch(() => "");
        throw new Error(raw.slice(0, 240) || `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let cut = buf.indexOf("\n\n");
        while (cut >= 0) {
          const frame = buf.slice(0, cut).trim();
          buf = buf.slice(cut + 2);
          cut = buf.indexOf("\n\n");
          if (!frame.startsWith("data:")) continue;
          const payload = frame.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          let chunk: StreamChunk;
          try {
            chunk = JSON.parse(payload) as StreamChunk;
          } catch {
            continue;
          }
          if (chunk.error?.message) {
            setStatus(`error: ${chunk.error.message}`);
            continue;
          }
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) {
            setMsgs((m) => {
              const next = m.slice();
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, text: last.text + delta };
              return next;
            });
          }
          if (chunk.sawii_tool?.name) {
            const name = chunk.sawii_tool.name;
            setMsgs((m) => {
              const next = m.slice();
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, tools: [...(last.tools || []), name] };
              return next;
            });
          }
        }
      }
      setStatus("");
    } catch (err) {
      const e = err as Error;
      setStatus(e.name === "AbortError" ? "stopped" : `turn failed: ${e.message}`);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const steer = async () => {
    const text = draft.trim();
    if (!text) return;
    try {
      const res = await fetch(httpUrl("/v1/interrupt"), {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ text }),
      });
      const data = (await res.json().catch(() => ({}))) as { accepted?: boolean; mode?: string | null };
      setStatus(
        data.accepted
          ? `steering accepted (${data.mode ?? "steer"})`
          : "not accepted — send it as the next message",
      );
      if (data.accepted) setDraft("");
    } catch (err) {
      setStatus(`interrupt failed: ${err instanceof Error ? err.message : "failed"}`);
    }
  };

  return (
    <div className="panel">
      <h3>Conversation</h3>
      <div className="row">
        <span className="pill">
          <span className="dot" data-s={busy ? "live" : "idle"} />
          {status || (personaKey() ? "ready" : "no key")} · {apiBase()}
        </span>
        <button className="btn" onClick={() => void loadHistory()}>
          Reload history
        </button>
        {busy && (
          <button className="btn" data-kind="danger" onClick={() => abortRef.current?.abort()}>
            Stop reply
          </button>
        )}
        <span className="spacer" />
        {historyNote && <span className="card-meta">{historyNote}</span>}
      </div>

      <div className="thread">
        {msgs.length === 0 && (
          <p className="hint">
            Nothing loaded. History comes from <code>GET /v1/timeline</code>; a turn is a streamed{" "}
            <code>POST /v1/chat/completions</code>. Provider failures show up here as an error status — that is the
            provider, not this window.
          </p>
        )}
        {msgs.map((m, i) => (
          <div className="msg" data-role={m.role} key={i}>
            <div className="who">{m.role === "user" ? "you" : "persona"}</div>
            {m.text || (busy && i === msgs.length - 1 ? "…" : "")}
            {m.tools && m.tools.length > 0 && (
              <div className="card-meta" style={{ marginTop: 4 }}>
                tools: {[...new Set(m.tools)].join(", ")}
              </div>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="row">
        <textarea
          className="goal"
          style={{ minHeight: 54 }}
          value={draft}
          placeholder="say something to the persona"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
      </div>
      <div className="row">
        <button className="btn" data-kind="primary" disabled={busy || !draft.trim()} onClick={() => void send()}>
          Send
        </button>
        <button className="btn" disabled={!busy || !draft.trim()} onClick={() => void steer()}>
          Steer current reply
        </button>
      </div>
    </div>
  );
}