/// <reference types="@sveltejs/kit" />
/// <reference lib="webworker" />

// ── PWA cache (from static/sw.js) ─────────────────────────────────

const CACHE_NAME = 'hoimetruyen-v2';
const APP_SHELL = [
	'/',
	'/manifest.webmanifest',
	'/brand/logo.webp',
	'/pwa/favicon-32.png',
	'/pwa/icon-192.png',
	'/pwa/icon-512.png',
	'/pwa/maskable-192.png',
	'/pwa/maskable-512.png',
	'/pwa/apple-touch-icon.png',
];

self.addEventListener('install', (event) => {
	(event as ExtendableEvent).waitUntil(self.skipWaiting());
	warmAppShell();
});

self.addEventListener('message', (event) => {
	if (event.data?.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('activate', (event) => {
	(event as ExtendableEvent).waitUntil(
		caches.keys()
			.then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
			.then(() => self.clients.claim()),
	);
});

// ── Combined fetch handler ───────────────────────────────────────

self.addEventListener('fetch', (event: FetchEvent) => {
	const request = event.request;
	const url = new URL(request.url);

	// 1. Source proxy: browser-side upstream fetch (bypasses CF Worker IP blocks)
	if (url.pathname.startsWith('/source-proxy/c/')) {
		const upstream = parseUpstream(url.pathname.slice(16), url);
		if (upstream) {
			event.respondWith(proxyFetch(upstream, request));
			return;
		}
	}

	if (request.method !== 'GET') return;
	if (url.origin !== self.location.origin) return;
	if (url.pathname.startsWith('/cdn/') || url.pathname.startsWith('/deepl/')) return;
	if (url.pathname.startsWith('/source-proxy/')) return;

	if (request.mode === 'navigate') {
		event.respondWith(networkFirst(request, '/'));
		return;
	}

	if (isStaticAsset(url.pathname)) {
		event.respondWith(cacheFirst(request));
		return;
	}

	event.respondWith(networkFirst(request));
});

// ── PWA helpers ──────────────────────────────────────────────────

function isStaticAsset(pathname: string) {
	return pathname.startsWith('/_app/immutable/')
		|| pathname.startsWith('/pwa/')
		|| pathname.startsWith('/brand/')
		|| pathname.startsWith('/assets/')
		|| pathname === '/manifest.webmanifest';
}

async function cacheFirst(request: Request) {
	const cached = await caches.match(request);
	if (cached) return cached;
	const response = await fetch(request);
	await putCache(request, response);
	return response;
}

async function networkFirst(request: Request, fallbackUrl?: string) {
	try {
		const response = await fetch(request);
		await putCache(request, response);
		return response;
	} catch {
		return await caches.match(request) ?? (fallbackUrl ? await caches.match(fallbackUrl) : undefined) ?? Response.error();
	}
}

async function putCache(request: Request, response: Response) {
	if (!response || !response.ok) return;
	const cache = await caches.open(CACHE_NAME);
	await cache.put(request, response.clone());
}

function warmAppShell() {
	caches.open(CACHE_NAME)
		.then((cache) => Promise.allSettled(APP_SHELL.map((url) => cache.add(url))))
		.catch(() => undefined);
}

// ── Source proxy ─────────────────────────────────────────────────

function parseUpstream(path: string, incoming: URL): URL | null {
	const slash = path.indexOf('/');
	const host = slash < 0 ? path : path.slice(0, slash);
	const rest = slash < 0 ? '' : path.slice(slash);
	if (!/^[a-z0-9.-]+$/i.test(host)) return null;

	const upstream = new URL(`https://${host}${rest}`);
	for (const [key, value] of incoming.searchParams) {
		if (key !== '_h') upstream.searchParams.append(key, value);
	}
	return upstream;
}

async function proxyFetch(upstream: URL, request: Request): Promise<Response> {
	const headers = new Headers();

	for (const name of ['accept', 'content-type', 'range', 'x-requested-with']) {
		const value = request.headers.get(name);
		if (value) headers.set(name, value);
	}

	const blob = request.headers.get('x-proxy-headers') ?? new URL(request.url).searchParams.get('_h');
	if (blob) {
		const explicit = decodeHeaderBlob(blob);
		for (const [name, value] of Object.entries(explicit)) {
			if (!/^(?:host|connection|cf-|x-forwarded-|sec-)/i.test(name)) {
				headers.set(name, value);
			}
		}
	}

	if (!headers.has('accept')) headers.set('accept', '*/*');

	let body: BodyInit | undefined;
	if (request.method !== 'GET' && request.method !== 'HEAD') {
		body = await request.clone().arrayBuffer();
	}

	try {
		const response = await fetch(upstream, {
			method: request.method,
			headers,
			body,
			redirect: 'follow',
		});
		return new Response(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers: {
				'content-type': response.headers.get('content-type') ?? 'application/octet-stream',
				'cache-control': response.headers.get('cache-control') ?? 'no-cache',
				'access-control-allow-origin': '*',
			},
		});
	} catch (err) {
		return new Response(`proxy error: ${(err as Error).message}`, { status: 502 });
	}
}

function decodeHeaderBlob(blob: string): Record<string, string> {
	if (!blob || blob.length > 8192) return {};
	try {
		const padded = blob.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - blob.length % 4) % 4);
		const bytes = Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
		const parsed = JSON.parse(new TextDecoder().decode(bytes)) as unknown;
		if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
		const out: Record<string, string> = {};
		for (const [key, value] of Object.entries(parsed)) {
			if (typeof value === 'string') out[key] = value;
		}
		return out;
	} catch {
		return {};
	}
}
