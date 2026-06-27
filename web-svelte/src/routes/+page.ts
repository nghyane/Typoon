// Public marketing landing — the ONE route we server-render and prerender so that
// search engines and first-time (logged-out) visitors receive real, indexable HTML.
// Everything else (the app shell, reader, login) stays a client-only SPA because it
// depends on IndexedDB / the Discord SDK / the source runtime — see each group's
// +layout.ts where ssr is kept false. These page-level options override the root
// layout's `ssr = false`, so only `/` is rendered on the server.
export const ssr = true;
export const prerender = true;
