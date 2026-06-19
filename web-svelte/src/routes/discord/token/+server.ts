import type { RequestHandler } from './$types';

const DISCORD_API = 'https://discord.com/api/v10';

interface TokenRequestBody {
	readonly client_id?: unknown;
	readonly code?: unknown;
	readonly code_verifier?: unknown;
	readonly redirect_uri?: unknown;
}

export const POST: RequestHandler = async ({ request, fetch }) => {
	const body = await request.json().catch(() => null) as TokenRequestBody | null;
	const clientID = stringField(body?.client_id);
	const code = stringField(body?.code);
	const codeVerifier = stringField(body?.code_verifier);
	const redirectURI = stringField(body?.redirect_uri);

	if (!/^\d{5,32}$/.test(clientID)) return json({ error: 'invalid_client_id' }, 400);
	if (!code || code.length > 512) return json({ error: 'invalid_code' }, 400);
	if (!/^[A-Za-z0-9._~-]{43,128}$/.test(codeVerifier)) return json({ error: 'invalid_code_verifier' }, 400);
	if (!isCallbackUrl(redirectURI)) return json({ error: 'invalid_redirect_uri' }, 400);

	const form = new URLSearchParams({
		client_id: clientID,
		grant_type: 'authorization_code',
		code,
		code_verifier: codeVerifier,
		redirect_uri: redirectURI,
	});
	const res = await fetch(`${DISCORD_API}/oauth2/token`, {
		method: 'POST',
		headers: { 'content-type': 'application/x-www-form-urlencoded' },
		body: form,
	});
	return proxyJson(res);
};

function stringField(value: unknown): string {
	return typeof value === 'string' ? value.trim() : '';
}

function isCallbackUrl(value: string): boolean {
	try {
		const url = new URL(value);
		return (url.protocol === 'https:' || url.hostname === 'localhost') && url.pathname === '/auth/callback';
	} catch {
		return false;
	}
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
