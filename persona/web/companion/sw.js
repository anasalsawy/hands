// Sawii's service worker: the app shell works offline-ish and installs as an
// app. Her mind is never cached - /v1 and /api always go to the server.
const CACHE = "sawii-shell-v1";
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./favicon.svg", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", (event) => {
	event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
	event.waitUntil(
		caches.keys()
			.then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
			.then(() => self.clients.claim()),
	);
});

self.addEventListener("fetch", (event) => {
	const req = event.request;
	const url = new URL(req.url);
	if (req.method !== "GET" || url.origin !== self.location.origin) return;
	if (/\/(v1|api)\//.test(url.pathname) || req.headers.get("accept")?.includes("text/event-stream")) return;
	if (req.mode === "navigate") {
		// Network first, so a deploy shows up at once; the shell when offline.
		event.respondWith(fetch(req).catch(() => caches.match("./index.html").then((r) => r || Response.error())));
		return;
	}
	// Hashed assets: cache first, fill on the way.
	event.respondWith(
		caches.match(req).then((hit) => hit || fetch(req).then((res) => {
			if (res.ok && res.type === "basic") {
				const copy = res.clone();
				caches.open(CACHE).then((c) => c.put(req, copy));
			}
			return res;
		})),
	);
});
