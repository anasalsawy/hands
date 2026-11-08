import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import { fileURLToPath, URL } from "node:url";

// The Hands desktop front end. Tauri loads the dev server here in `tauri dev`,
// and the bundled dist/ inside the released exe — the app is never served by
// the Persona backend.
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5181,
    strictPort: true,
    host: host || "127.0.0.1",
    hmr: host ? { protocol: "ws", host, port: 5182 } : undefined,
  },
  build: { target: "es2022", outDir: "dist", sourcemap: false },
});