// Comix token worker — isolated Web Worker.
// Polyfills all browser APIs the secure module needs.
// Only token strings leave this worker.

const VENDOR_BASE = '/comix-vendor/';

// Must be at top level — secure module reads cfg on import
let _cfg = '';
let _ready = false;
let _client: unknown = null;
let _generate: ((params: Record<string, unknown>) => Promise<string>) | null = null;

// ── Polyfill browser APIs for worker context ─────────────────────

// Worker already has: self, navigator, location, fetch, etc.
// window mock with comix origin — secure module may read window.location.origin
const windowMock = {
	get location() { return { origin: 'https://comix.to', href: 'https://comix.to/', hostname: 'comix.to', protocol: 'https:', toString() { return 'https://comix.to/'; } }; },
	get document() { return (self as unknown as Record<string, unknown>).document; },
	get navigator() { return self.navigator; },
};
(self as unknown as Record<string, unknown>).window = windowMock;
(self as unknown as Record<string, unknown>).document = {
	querySelector(sel: string) {
		if (String(sel).includes('cfg')) return { content: _cfg, getAttribute() { return _cfg; } };
		return null;
	},
	querySelectorAll() { return []; },
	createElement() { return { style: {}, setAttribute() {} }; },
	getElementsByTagName() { return [{ appendChild() {} }]; },
	body: { appendChild() {} },
	documentElement: {},
	head: { append() {}, appendChild() {} },
	addEventListener() {},
	createElementNS() { return { setAttribute() {}, setAttributeNS() {} }; },
};

async function init(): Promise<boolean> {
	try {
		const m = await (await fetch(VENDOR_BASE + 'manifest.json')).json();
		_cfg = m.cfg;
		if (!_cfg) throw new Error('no cfg');

		const [rt, vd, sc] = await Promise.all([
			fetch(VENDOR_BASE + m.runtime).then((r: Response) => r.text()),
			fetch(VENDOR_BASE + m.vendor).then((r: Response) => r.text()),
			fetch(VENDOR_BASE + m.secure).then((r: Response) => r.text()),
		]);

		const rb = blobModule(rt, {});
		const vb = blobModule(vd, { [m.runtime as string]: rb });
		const sb = blobModule(sc, { [m.vendor as string]: vb, [m.runtime as string]: rb });

		const secure = await import(sb);
		if (!secure?.i) throw new Error('no secure.i()');

		const axiosMod = await import('axios');
		const axiosInst = axiosMod.default;
		_client = axiosInst.create({
			baseURL: 'https://comix.to/api/v1',
			headers: { Accept: 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
		});
		secure.i(_client);

		_generate = async (params: Record<string, unknown>) => {
			let signed: Record<string, unknown> | null = null;
			const c = _client as { interceptors: { request: { use(fn: (c: Record<string, unknown>) => Record<string, unknown>): number; eject(n: number): void } }; get(path: string, opts?: Record<string, unknown>): Promise<unknown> };
			const capture = c.interceptors.request.use((cfg) => { signed = cfg.params as Record<string, unknown>; throw new Error('x'); });
			try { await c.get('/manga', { params }); } catch (err) { if ((err as Error).message !== 'x') throw err; }
			c.interceptors.request.eject(capture);
			const token = signed ? (signed as Record<string, unknown>)._ as string : undefined;
			if (!token) throw new Error('no token');
			return token;
		};

		return true;
	} catch (err) {
		self.postMessage({ id: -1, error: (err as Error).message });
		return false;
	}
}

function blobModule(code: string, importMap: Record<string, string>): string {
	for (const [file, url] of Object.entries(importMap)) {
		code = code.replace(
			new RegExp(`(from["'])\\.\\/${escapeRx(file)}(["'])`, 'g'),
			`$1${url}$2`,
		);
	}
	return URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
}
function escapeRx(s: string): string { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

// ── Message handler ──────────────────────────────────────────────

self.addEventListener('message', async (ev: MessageEvent<{ id: number; params: Record<string, unknown> }>) => {
	const { id, params } = ev.data;
	if (!id) return;

	if (!_ready) {
		_ready = await init();
		if (!_ready) { self.postMessage({ id, error: 'init failed' }); return; }
	}

	try {
		const token = await _generate!(params);
		self.postMessage({ id, token });
	} catch (err) {
		self.postMessage({ id, error: (err as Error).message });
	}
});

export {};
