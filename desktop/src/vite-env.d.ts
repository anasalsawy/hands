/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Build-time default backend, e.g. http://127.0.0.1:8800 */
  readonly VITE_HANDS_API_BASE?: string;
  /** Build-time default persona key. Never commit a real one. */
  readonly VITE_HANDS_PERSONA_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}