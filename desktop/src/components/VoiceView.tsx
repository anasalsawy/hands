// Voice line. One WebSocket to `/v1/voice` (the persona's own realtime session —
// Hands does not invent a voice stack). Binary frames up are microphone audio,
// binary frames down are speech to play back; JSON frames carry `transcript`,
// `delta`, `done` and `error`. `GET /v1/voice-status` reports what the server
// can actually do (ASR / TTS / sidecar), so a dead provider reads as a provider
// problem, not a broken window.
import { useCallback, useEffect, useRef, useState } from "react";
import { systemApi, type VoiceStatus } from "../lib/client";
import { keyedWsUrl, personaKey } from "../lib/connection";

type Line = { id: number; who: "you" | "persona" | "note"; text: string };

let seq = 0;

export function VoiceView() {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [probeErr, setProbeErr] = useState("");
  const [conn, setConn] = useState<"idle" | "open" | "closed" | "error">("idle");
  const [ready, setReady] = useState<Record<string, unknown> | null>(null);
  const [recording, setRecording] = useState(false);
  const [typing, setTyping] = useState("");
  const [lines, setLines] = useState<Line[]>([]);
  const [note, setNote] = useState("");
  const sockRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const ctxRef = useRef<AudioContext | null>(null);
  const playRef = useRef<number>(0);
  const linesRef = useRef<HTMLDivElement | null>(null);

  const add = useCallback((who: Line["who"], text: string) => {
    if (!text) return;
    seq += 1;
    setLines((rows) => [...rows.slice(-160), { id: seq, who, text }]);
  }, []);

  useEffect(() => {
    linesRef.current?.scrollTo({ top: linesRef.current.scrollHeight });
  }, [lines]);

  const loadStatus = useCallback(async () => {
    try {
      setStatus(await systemApi.voice());
      setProbeErr("");
    } catch (err) {
      setProbeErr(err instanceof Error ? err.message : "unreachable");
    }
  }, []);

  useEffect(() => {
    void loadStatus();
  }, [loadStatus]);

  const play = useCallback(async (buf: ArrayBuffer) => {
    if (!buf.byteLength) return;
    const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    ctxRef.current ??= new Ctor();
    const ctx = ctxRef.current;
    const view = new DataView(buf);
    const pcm = new Int16Array(buf.byteLength / 2);
    for (let i = 0; i < pcm.length; i += 1) pcm[i] = view.getInt16(i * 2, true);
    const out = ctx.createBuffer(1, pcm.length, 24000);
    const chan = out.getChannelData(0);
    for (let i = 0; i < pcm.length; i += 1) chan[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = out;
    src.connect(ctx.destination);
    src.start(ctx.currentTime + playRef.current);
    playRef.current += out.duration;
  }, []);

  const connect = useCallback(() => {
    if (sockRef.current && sockRef.current.readyState <= WebSocket.OPEN) return;
    if (!personaKey()) {
      setNote("no persona key — connect first");
      return;
    }
    setNote("");
    let sock: WebSocket;
    try {
      sock = new WebSocket(keyedWsUrl("/v1/voice"));
    } catch (err) {
      setNote(`socket refused: ${err instanceof Error ? err.message : "ws"}`);
      return;
    }
    sock.binaryType = "arraybuffer";
    sock.onopen = () => setConn("open");
    sock.onclose = () => {
      setConn("closed");
      setRecording(false);
    };
    sock.onerror = () => setConn("error");
    sock.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        void play(ev.data as ArrayBuffer);
        return;
      }
      let frame: Record<string, unknown>;
      try {
        frame = JSON.parse(ev.data) as Record<string, unknown>;
      } catch {
        return;
      }
      switch (frame.type) {
        case "ready":
          setReady(frame);
          add("note", `voice session open · tts=${String(frame.tts)} asr=${String(frame.asr)}`);
          break;
        case "transcript":
          add("you", String(frame.text || ""));
          break;
        case "delta":
          add("persona", String(frame.text || ""));
          break;
        case "done":
          add("note", `turn done${frame.first_audio_ms ? ` · first audio ${Math.round(Number(frame.first_audio_ms))} ms` : ""}`);
          break;
        case "error":
          add("note", `error: ${String(frame.detail || "unknown")}`);
          break;
        default:
          break;
      }
    };
    sockRef.current = sock;
  }, [add, play]);

  const sendFrame = (frame: Record<string, unknown>) => {
    const sock = sockRef.current;
    if (!sock || sock.readyState !== WebSocket.OPEN) {
      setNote("voice socket is not open");
      return;
    }
    sock.send(JSON.stringify(frame));
  };

  const stopRecording = useCallback(async () => {
    const rec = recorderRef.current;
    if (!rec || rec.state === "inactive") {
      setRecording(false);
      return;
    }
    await new Promise<void>((resolve) => {
      rec.onstop = () => resolve();
      rec.stop();
    });
    recorderRef.current = null;
    setRecording(false);
  }, []);

  const startRecording = useCallback(async () => {
    if (recorderRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setNote("no microphone API in this webview");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        const sock = sockRef.current;
        const mime = rec.mimeType || "audio/webm";
        void (async () => {
          for (const blob of chunksRef.current) {
            if (sock?.readyState === WebSocket.OPEN) sock.send(await blob.arrayBuffer());
          }
          if (sock?.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ type: "end_utterance", mime }));
        })();
        chunksRef.current = [];
        stream.getTracks().forEach((t) => t.stop());
      };
      rec.start();
      recorderRef.current = rec;
      setRecording(true);
      connect();
    } catch (err) {
      setNote(`microphone refused: ${err instanceof Error ? err.message : "denied"}`);
    }
  }, [connect]);

  useEffect(
    () => () => {
      try {
        recorderRef.current?.stream.getTracks().forEach((t) => t.stop());
      } catch {
        /* noop */
      }
      try {
        sockRef.current?.close();
      } catch {
        /* noop */
      }
      void ctxRef.current?.close().catch(() => undefined);
    },
    [],
  );

  const speech = (status?.persona_speech ?? ready ?? {}) as Record<string, unknown>;

  return (
    <div className="split">
      <div className="panel">
        <h3>Voice line</h3>
        <div className="row">
          <span className="pill">
            <span className="dot" data-s={conn === "open" ? "live" : conn === "idle" ? "idle" : "down"} />
            WS /v1/voice · {conn}
          </span>
          {conn === "open" ? (
            <button
              className="btn"
              data-kind="danger"
              onClick={() => {
                sockRef.current?.close();
                setConn("closed");
              }}
            >
              Close session
            </button>
          ) : (
            <button className="btn" data-kind="primary" onClick={connect}>
              Open session
            </button>
          )}
          <button className="btn" onClick={() => void (recording ? stopRecording() : startRecording())}>
            {recording ? "Send take" : "Record and send"}
          </button>
          <button className="btn" onClick={() => sendFrame({ type: "interrupt" })}>
            Interrupt
          </button>
          <button className="btn" onClick={() => sendFrame({ type: "discard_audio" })}>
            Discard
          </button>
        </div>
        {note && <div className="err">{note}</div>}
        <div className="row">
          <input
            style={{ flex: 1 }}
            value={typing}
            placeholder="or type what you would say — sent as a voice turn"
            onChange={(e) => setTyping(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && typing.trim()) {
                sendFrame({ type: "text", text: typing.trim() });
                add("you", typing.trim());
                setTyping("");
              }
            }}
          />
          <button
            className="btn"
            disabled={!typing.trim()}
            onClick={() => {
              sendFrame({ type: "text", text: typing.trim() });
              add("you", typing.trim());
              setTyping("");
            }}
          >
            Send turn
          </button>
        </div>
        <div className="log" ref={linesRef} style={{ maxHeight: "42vh" }}>
          {lines.length === 0 && (
            <p className="hint">
              Open a session, then record or type. Transcripts and replies stream back on the same socket.
            </p>
          )}
          {lines.map((l) => (
            <div key={l.id}>
              <span>{l.who}</span>
              {l.text}
            </div>
          ))}
        </div>
      </div>

      <div className="stack">
        <div className="panel">
          <h3>Voice capability · GET /v1/voice-status</h3>
          <div className="row">
            <button className="btn" onClick={() => void loadStatus()}>
              Re-check
            </button>
          </div>
          {probeErr && <div className="err">{probeErr}</div>}
          {status ? (
            <div className="kv">
              <b>WebSocket</b>
              <span>{status.websocket}</span>
              <b>Sidecar</b>
              <span>
                {status.sidecar?.running ? `running · ${status.sidecar.url}` : "not running"}
              </span>
              <b>ASR</b>
              <span>{String(speech.asr ?? "—")}</span>
              <b>TTS</b>
              <span>{String(speech.speak ?? speech.tts ?? "—")}</span>
              <b>Native</b>
              <span>{String(speech.native ?? "—")}</span>
            </div>
          ) : (
            !probeErr && <p className="hint">Route not answering.</p>
          )}
          {status?.sidecar?.url && (
            <p className="hint">
              The sidecar is a separate process (<code>{status.sidecar.url}/health</code>). If ASR or TTS reports a
              provider failure, that is the key, not this window.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}