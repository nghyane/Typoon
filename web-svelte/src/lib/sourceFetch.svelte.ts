// $lib/sourceFetch.svelte.ts
// Source transport only. Source-specific headers/cache are passed in by manifests/adapters.
//   fetchSource   — proxy fetch for source runtime/adapters
//   useSourceFetch() — proxy URL builder for browser-loaded images

const SOURCE_CDN_GATEWAYS = [
	'https://927251094806098001.discordsays.com/cdn/c',
	'https://function-bun-production-c2e1.up.railway.app/cdn/c',
];

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
	return _sourceCdn;
}

const _sourceCdn = createSourceFetch(SOURCE_CDN_GATEWAYS);
export const fetchSource = _sourceCdn.fetch;
export const toBrowserUrl = _sourceCdn.toBrowserUrl;

function createSourceFetch(origins: readonly string[]) {
	const gateways = origins
		.map(g => g.replace(/\/+$/u, ''))
		.filter(g => /^https?:\/\/[^?#]+$/i.test(g));

	if (gateways.length === 0) gateways.push('/cdn/c');

	function gatewayUrl(key: string, index: number): string {
		return `${gateways[index % gateways.length]!}/${key}`;
	}

	function _toBrowserUrl(url: string, headers?: Record<string, string>, cache?: SourceCacheOptions): string {
		let u: URL;
		try { u = new URL(url); } catch { return url; }
		if (u.protocol !== 'https:' && u.protocol !== 'http:') return url;

		const key = `${u.host}${u.pathname}`;
		const path = gatewayUrl(key, hash(key) % gateways.length);
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
		let lastError: unknown;
		for (let attempt = 0; attempt < gateways.length; attempt++) {
			const gw = gateways[attempt]!;
			const gwUrl = `${gw}/${key}?${params}`;

			if (attempt > 0) {
				// Exponential backoff: 1s, 2s, 4s...
				await new Promise(r => setTimeout(r, Math.min(1000 * Math.pow(2, attempt - 1), 8000)));
			}

			try {
				const response = await fetch(gwUrl, { ...opts.init, headers: h, signal: opts.init?.signal });
				if (response.status === 429 || response.status === 502 || response.status === 503) {
					lastError = new Error(`HTTP ${response.status} via ${gw}`);
					continue;
				}
				return response;
			} catch (err) {
				lastError = err;
			}
		}

		throw lastError ?? new Error('all gateways exhausted');
	}

	return { toBrowserUrl: _toBrowserUrl, fetch: _fetch };
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

function hash(value: string): number {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
