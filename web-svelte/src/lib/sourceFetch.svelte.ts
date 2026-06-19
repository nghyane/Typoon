// $lib/sourceFetch.svelte.ts
// Two fetch layers, matching React's proxy.ts architecture:
//   fetchSource   — module-level singleton for runtime.ts (uses configured CDN proxy)
//   useSourceFetch() — CDN origins for browser <img> URLs (Cover component)

let _cdnInstance: ReturnType<typeof createSourceFetch> | null = null;
const SOURCE_CDN_BASE = 'https://927251094806098001.discordsays.com/cdn/c';

/** CDN-based instance for browser image URLs (Cover component). */
export function useSourceFetch() {
	if (!_cdnInstance) {
		_cdnInstance = createSourceFetch([SOURCE_CDN_BASE]);
	}
	return _cdnInstance;
}

const _sourceCdn = createSourceFetch([SOURCE_CDN_BASE]);
export const fetchSource = _sourceCdn.fetch;
export const toBrowserUrl = _sourceCdn.toBrowserUrl;

function createSourceFetch(origins: readonly string[]) {
	const gateways = origins
		.map(g => g.replace(/\/+$/u, ''))
		.filter(g => /^https?:\/\/[^?#]+$/i.test(g));

	function gatewayFor(key: string): string {
		if (gateways.length === 0) return '/cdn/c';
		const origin = gateways[hash(key) % gateways.length];
		return origin;
	}

	function _toBrowserUrl(url: string, headers?: Record<string, string>, mode: 'fetch' | 'image' = 'image'): string {
		let u: URL;
		try { u = new URL(url); } catch { return url; }
		if (u.protocol !== 'https:' && u.protocol !== 'http:') return url;
		const path = `${gatewayFor(`${u.host}${u.pathname}`)}/${u.host}${u.pathname}`;
		const params = new URLSearchParams(u.searchParams);
		const proxyHeaders = headersFor(u, mode, headers);
		if (Object.keys(proxyHeaders).length > 0) params.set('_h', encodeHeaderBlob(proxyHeaders));
		const qs = params.toString();
		return qs ? `${path}?${qs}` : path;
	}

	function _fetch(url: string, opts: { headers?: Record<string, string>; init?: RequestInit } = {}): Promise<Response> {
		const headers = new Headers(opts.init?.headers);
		const proxyHeaders = headersForUrl(url, 'fetch', opts.headers);
		if (Object.keys(proxyHeaders).length > 0) {
			headers.set('X-Proxy-Headers', encodeHeaderBlob(proxyHeaders));
		}
		return fetch(_toBrowserUrl(url, undefined, 'fetch'), { ...opts.init, headers });
	}

	return { toBrowserUrl: _toBrowserUrl, fetch: _fetch };
}

function headersForUrl(url: string, mode: 'fetch' | 'image', explicit?: Record<string, string>): Record<string, string> {
	try { return headersFor(new URL(url), mode, explicit); } catch { return explicit ?? {}; }
}

function headersFor(url: URL, mode: 'fetch' | 'image', explicit?: Record<string, string>): Record<string, string> {
	return { ...mangadexHeaders(url, mode), ...(explicit ?? {}) };
}

function mangadexHeaders(url: URL, mode: 'fetch' | 'image'): Record<string, string> {
	if (url.host !== 'api.mangadex.org' && url.host !== 'uploads.mangadex.org') return {};
	return {
		'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
		'Accept': mode === 'fetch' ? 'application/json, text/plain, */*' : 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
		'Accept-Language': 'en-US,en;q=0.9',
		'Accept-Encoding': 'gzip, deflate, br, zstd',
		'Referer': 'https://mangadex.org/',
		'Origin': 'https://mangadex.org',
		'Sec-Fetch-Site': 'same-site',
		'Sec-Fetch-Mode': mode === 'fetch' ? 'cors' : 'no-cors',
		'Sec-Fetch-Dest': mode === 'fetch' ? 'empty' : 'image',
		'Sec-CH-UA': '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
		'Sec-CH-UA-Mobile': '?0',
		'Sec-CH-UA-Platform': '"macOS"',
	};
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
