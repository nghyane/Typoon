// $lib/config.ts — shared constants.  Single source of truth.
export const FETCH_CONCURRENCY = 6;
export const FETCH_TIMEOUT_MS = 30_000;
export const PAGE_MAX_WIDTH = 'max-w-3xl';
export const MIN_CANVAS_HEIGHT = 200;

// ── Source-proxy gateways ──────────────────────────────────────────────
// The browser cannot fetch upstream manga sites directly (CORS / Cloudflare),
// so every source request is routed through a proxy gateway. Gateways are tried
// in order: the first is the primary, the rest are fallback mirrors used only
// when an earlier one errors out.
//
// `discordsays` leads because the app ships as a Discord Activity: inside that
// sandbox only the mapped discordsays.com proxy is reachable (external hosts are
// blocked by Discord's CSP), and it also works fine in a plain browser — so it
// is the one origin that works everywhere. `bunle` (our own proxy, allows every
// host, never Cloudflare-blocked) and the Railway mirror follow as fallbacks for
// non-Activity deploys or when discordsays is rate-limited.
//
// Values are origins (scheme + host); the proxy path is appended by the
// transport, matching the server's PublicSettings.sourceFetch.origins shape.
// Override per-deployment with VITE_SOURCE_CDN_GATEWAYS (comma-separated, with
// or without a trailing /cdn/c) — no code change required.
const DEFAULT_SOURCE_GATEWAYS = [
	'https://927251094806098001.discordsays.com',
	'https://bunle-cdn-ceu.pages.dev',
	'https://function-bun-production-c2e1.up.railway.app',
];

/** Path segment that the proxy worker listens on, appended to each gateway origin. */
export const SOURCE_GATEWAY_PROXY_PATH = 'cdn/c';

/** Build-time gateway list: env override if provided, else the bunle-first defaults. */
export function sourceGateways(): string[] {
	const override = String(import.meta.env.VITE_SOURCE_CDN_GATEWAYS ?? '')
		.split(',')
		.map((value) => value.trim())
		.filter(Boolean);
	return override.length > 0 ? override : DEFAULT_SOURCE_GATEWAYS;
}
