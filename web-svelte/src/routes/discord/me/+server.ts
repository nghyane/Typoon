import type { RequestHandler } from './$types';

const DISCORD_API = 'https://discord.com/api/v10';

export const GET: RequestHandler = async ({ request, fetch }) => {
	const auth = bearerAuth(request);
	if (!auth) return json({ error: 'missing_bearer' }, 401);

	const res = await fetch(`${DISCORD_API}/users/@me`, {
		headers: { Authorization: auth },
	});
	return proxyJson(res);
};

function bearerAuth(request: Request): string {
	const auth = request.headers.get('Authorization') || '';
	return auth.startsWith('Bearer ') ? auth : '';
}

async function proxyJson(res: Response): Promise<Response> {
	return new Response(await res.text(), {
		status: res.status,
		headers: {
			'content-type': res.headers.get('content-type') || 'application/json',
			'cache-control': 'no-store',
		},
	});
}

function json(value: unknown, status: number): Response {
	return new Response(JSON.stringify(value), {
		status,
		headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
	});
}
