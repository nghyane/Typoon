import type { RequestHandler } from './$types';

// Fallback: returns 503 when Service Worker hasn't intercepted yet (first load).
// Once SW is active, it handles all /source-proxy/c/* requests from the browser.
export const GET: RequestHandler = () =>
	new Response('source proxy: waiting for Service Worker', { status: 503 });
export const POST: RequestHandler = GET;
export const HEAD: RequestHandler = GET;
