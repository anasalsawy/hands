import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./nivo.css";

const el = document.getElementById("root");
if (!el) throw new Error("no #root");
createRoot(el).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
if ("serviceWorker" in navigator && location.protocol.startsWith("http") && !/^(localhost|127\.)/.test(location.hostname)) {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}
