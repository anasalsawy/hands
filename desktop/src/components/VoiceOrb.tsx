// The voice orb: one living shape that shows who is talking.
// idle = slow breathing, listening = follows your voice, thinking = orbiting light,
// speaking = pulses with the persona's voice. `levelRef` is read every frame, so
// the orb moves with the audio without re-rendering React.
import { useEffect, useRef } from "react";
import type { MutableRefObject } from "react";

export type OrbState = "idle" | "listening" | "thinking" | "speaking";

const LABEL: Record<OrbState, string> = {
  idle: "Tap to talk",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

export function VoiceOrb({
  state,
  levelRef,
  caption,
  onTap,
}: {
  state: OrbState;
  levelRef: MutableRefObject<number>;
  caption?: string;
  onTap: () => void;
}) {
  const el = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    let raf = 0;
    let smooth = 0;
    const tick = () => {
      smooth += (levelRef.current - smooth) * 0.25;
      el.current?.style.setProperty("--lvl", smooth.toFixed(3));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [levelRef]);

  return (
    <div className="orb-stage">
      <button ref={el} className="orb" data-state={state} onClick={onTap} aria-label={LABEL[state]}>
        <span className="orb-glow" />
        <span className="orb-body">
          <span className="orb-blob b1" />
          <span className="orb-blob b2" />
          <span className="orb-blob b3" />
          <span className="orb-sheen" />
        </span>
        <span className="orb-ring r1" />
        <span className="orb-ring r2" />
      </button>
      <div className="orb-label">{LABEL[state]}</div>
      <div className="orb-caption" aria-live="polite">
        {caption || ""}
      </div>
    </div>
  );
}
