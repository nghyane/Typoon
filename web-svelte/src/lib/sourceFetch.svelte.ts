// $lib/sourceFetch.svelte.ts
// Source transport only. Source-specific headers/cache are passed in by manifests/adapters.
//   fetchSource   — proxy fetch for source runtime/adapters
//   useSourceFetch() — proxy URL builder for browser-loaded images
//
// The gateway list lives in $lib/config (env-overridable, discordsays-first);
// this module only normalizes it into ready-to-use proxy bases and can be
// re-pointed at runtime via configureSourceGateways() when the server supplies
// origins.

import { sourceGateways, SOURCE_GATEWAY_PROXY_PATH, FETCH_TIMEOUT_MS } from '$lib/config';

const PROXY_PATH = SOURCE_GATEWAY_PROXY_PATH.replace(/^\/+|\/+$/gu, '');

/** Normalize origins (scheme+host, with or without /cdn/c) into proxy bases. */
function normalizeGateways(origins: readonly string[]): string[] {
	const bases = origins
		.map(g => g.trim().replace(/\/+$/u, ''))
		.filter(g => /^https?:\/\/[^?#]+$/i.test(g))
		.map(g => new RegExp(`/${PROXY_PATH}$`, 'iu').test(g) ? g : `${g}/${PROXY_PATH}`);
	// Same-origin fallback when nothing valid is configured (dev / self-hosted proxy).
	return bases.length > 0 ? bases : [`/${PROXY_PATH}`];
}

let _gateways: string[] | null = null;
function gateways(): string[] {
	return (_gateways ??= normalizeGateways(sourceGateways()));
}

/**
 * Override the gateway list at runtime — e.g. from the server's
 * PublicSettings.sourceFetch.origins. No-op when empty so the build-time
 * defaults stand. Affects every instance since they read the list live.
 */
export function configureSourceGateways(origins: readonly string[]): void {
	if (origins.length > 0) _gateways = normalizeGateways(origins);
}

/** Combine the caller's abort signal (if any) with a per-request timeout. */
function withTimeout(signal: AbortSignal | null | undefined): AbortSignal {
	const timeout = AbortSignal.timeout(FETCH_TIMEOUT_MS);
	return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

let _cdnInstance: ReturnType<typeof createSourceFetch> | null = null;

export type SourceCachePolicy = 'auto' | 'immutable' | 'ttl' | 'reload' | 'bypass' | 'no-store' | 'only-if-cached';

export interface SourceCacheOptions {
	policy?: SourceCachePolicy;
	key?: string;
	ttl?: number;
}

interface SourceFetchOptions {
	headers?: Record<string, string>;
	init?: RequestInit;
	cache?: SourceCacheOptions;
}

/** CDN-based instance for browser image URLs (Cover component). */
export function useSourceFetch() {
	return (_cdnInstance ??= createSourceFetch());
}

const _sourceCdn = createSourceFetch();
export const fetchSource = _sourceCdn.fetch;
export const toBrowserUrl = _sourceCdn.toBrowserUrl;

function createSourceFetch() {
	function gatewayUrl(key: string, index: number): string {
		const g = gateways();
		return `${g[index % g.length]!}/${key}`;
	}

	function _toBrowserUrl(url: string, headers?: Record<string, string>, cache?: SourceCacheOptions, attempt = 0): string {
		let u: URL;
		try { u = new URL(url); } catch { return url; }
		if (u.protocol !== 'https:' && u.protocol !== 'http:') return url;

		const key = `${u.host}${u.pathname}`;
		// attempt 0 = primary gateway; on image error the caller retries with
		// attempt 1, 2… to fall back to the mirrors (a gateway can be unreachable
		// or Cloudflare-blocked for a given host while another works).
		const path = gatewayUrl(key, attempt % gateways().length);
		const params = new URLSearchParams(u.searchParams);
		if (headers && Object.keys(headers).length > 0) params.set('_h', encodeHeaderBlob(headers));
		applyCacheParams(params, cache);
		const qs = params.toString();
		return qs ? `${path}?${qs}` : path;
	}

	async function _fetch(url: string, opts: SourceFetchOptions = {}): Promise<Response> {
		const h = new Headers(opts.init?.headers);
		if (opts.headers && Object.keys(opts.headers).length > 0) {
			h.set('X-Proxy-Headers', encodeHeaderBlob(opts.headers));
		}
		applyCacheHeaders(h, opts.cache);

		let u: URL;
		try { u = new URL(url); } catch { return fetch(url, { ...opts.init, headers: h }); }
		if (u.protocol !== 'https:' && u.protocol !== 'http:') return fetch(url, { ...opts.init, headers: h });

		// Forward original query params to each gateway attempt
		const params = new URLSearchParams(u.searchParams);

		const key = `${u.host}${u.pathname}`;
		// Always try gateways in order: primary first, fallbacks on failure
		const gw = gateways();
		let lastError: unknown;
		// Back off only when the previous gateway was rate-limited (429/503) —
		// waiting helps there. A Cloudflare challenge (403) or bad gateway (502)
		// won't clear by waiting, so fail straight over to the next gateway.
		let backoff = false;
		for (let attempt = 0; attempt < gw.length; attempt++) {
			const gwUrl = `${gw[attempt]!}/${key}?${params}`;

			if (attempt > 0 && backoff) {
				// Exponential backoff: 1s, 2s, 4s...
				await new Promise(r => setTimeout(r, Math.min(1000 * Math.pow(2, attempt - 1), 8000)));
			}

			try {
				// Per-gateway timeout: a gateway that opens the connection but never
				// responds (e.g. the discordsays proxy reached from a plain browser
				// instead of inside the Discord Activity) would otherwise hang the
				// request forever. Abort after FETCH_TIMEOUT_MS so we fail over.
				const response = await fetch(gwUrl, { ...opts.init, headers: h, signal: withTimeout(opts.init?.signal) });
				// 403 included: Cloudflare's "Just a moment" managed challenge 403s a
				// gateway whose egress it distrusts while another gateway (different
				// IP/fingerprint) sails through — so fail over instead of surfacing it.
				if ([403, 429, 502, 503].includes(response.status)) {
					lastError = new Error(`HTTP ${response.status} via ${gw[attempt]}`);
					backoff = response.status === 429 || response.status === 503;
					continue;
				}
				return response;
			} catch (err) {
				// Caller cancelled (modal closed, navigation away) — stop, don't churn
				// through the remaining gateways with an already-aborted signal.
				if (opts.init?.signal?.aborted) throw err;
				lastError = err;
				backoff = false;
			}
		}

		throw lastError ?? new Error('all gateways exhausted');
	}

	return { toBrowserUrl: _toBrowserUrl, fetch: _fetch, get gatewayCount() { return gateways().length; } };
}

function applyCacheParams(params: URLSearchParams, cache?: SourceCacheOptions): void {
	if (!cache) return;
	if (cache.policy) params.set('_pc', cache.policy);
	if (cache.key) params.set('_pk', cache.key);
	if (cache.ttl != null) params.set('_pt', String(Math.max(0, Math.floor(cache.ttl))));
}

function applyCacheHeaders(headers: Headers, cache?: SourceCacheOptions): void {
	if (!cache) return;
	if (cache.policy) headers.set('X-Proxy-Cache', cache.policy);
	if (cache.key) headers.set('X-Proxy-Cache-Key', cache.key);
	if (cache.ttl != null) headers.set('X-Proxy-Cache-TTL', String(Math.max(0, Math.floor(cache.ttl))));
}

function encodeHeaderBlob(headers: Record<string, string>): string {
  const sorted: Record<string, string> = {};
  for (const key of Object.keys(headers).sort()) sorted[key] = headers[key]!;
  const bytes = new TextEncoder().encode(JSON.stringify(sorted));
  let bin = '';
  for (const byte of bytes) bin += String.fromCharCode(byte);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
