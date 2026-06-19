const CACHE_NAME = 'hoimetruyen-v1';
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
	event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
	event.waitUntil(
		caches.keys()
			.then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
			.then(() => self.clients.claim()),
	);
});

self.addEventListener('fetch', (event) => {
	const request = event.request;
	if (request.method !== 'GET') return;

	const url = new URL(request.url);
	if (url.origin !== self.location.origin) return;
	if (url.pathname.startsWith('/cdn/') || url.pathname.startsWith('/deepl/')) return;

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

function isStaticAsset(pathname) {
	return pathname.startsWith('/_app/immutable/')
		|| pathname.startsWith('/pwa/')
		|| pathname.startsWith('/brand/')
		|| pathname.startsWith('/assets/')
		|| pathname === '/manifest.webmanifest';
}

async function cacheFirst(request) {
	const cached = await caches.match(request);
	if (cached) return cached;
	const response = await fetch(request);
	await putCache(request, response);
	return response;
}

async function networkFirst(request, fallbackUrl) {
	try {
		const response = await fetch(request);
		await putCache(request, response);
		return response;
	} catch (error) {
		return await caches.match(request) ?? (fallbackUrl ? await caches.match(fallbackUrl) : undefined) ?? Response.error();
	}
}

async function putCache(request, response) {
	if (!response || !response.ok) return;
	const cache = await caches.open(CACHE_NAME);
	await cache.put(request, response.clone());
}
