import { fetchSource } from '$lib/sourceFetch.svelte';
import { getAdapter } from '../adapters';
import { applyChapterNumberNorm, compileChapterNumberNorm } from '../normalize';
import { sourceCache, type SourceCacheScope } from './cache';
import { queryHtmlAll, queryHtmlOne, queryJsonAll, queryJsonOne } from '../selectors';
import type {
	BrowseArgs, BrowseEndpoint, ChapterFields, ChapterListSpec, ChapterPages,
	ChaptersApiEndpoint, HttpRequest, MangaChapterRef, MangaDetail, MangaSummary,
	SourceManifest,
} from '../types';

type Vars = Record<string, string | number | undefined | null>;
type Fields = Record<string, string>;

interface Fetched { url: string; parsed: Document | unknown }

interface Row { get: (s: string) => string | null; raw: unknown }

// ── helpers ────────────────────────────────────────────────────────

function tpl(t: string, v: Vars) {
	return t.replace(/\{([a-zA-Z_][\w]*)(?::(q))?\}/g, (_m, n: string, mod?: string) => {
		const x = v[n]; if (x == null) return '';
		return mod === 'q' ? encodeURIComponent(String(x)) : String(x);
	});
}

function tplHeaders(headers: Record<string, string> | undefined, vars: Vars): Record<string, string> | undefined {
	if (!headers) return undefined;
	if (Object.keys(headers).length === 0) return undefined;
	const out: Record<string, string> = {};
	for (const [key, value] of Object.entries(headers)) out[key] = tpl(value, vars);
	return out;
}

function absUrl(href: string | null | undefined, base: string) {
	if (!href) return null;
	try { return new URL(href, base).href; } catch { return null; }
}

function extract(input: string, pattern?: string): Record<string, string> {
	if (!pattern) return {};
	try { return (new RegExp(pattern).exec(input)?.groups ?? {}) as Record<string, string>; } catch { return {}; }
}

function cookieHdr(m: SourceManifest, c: Record<string, string>) {
	if (!m.cookieNames?.length) return null;
	const p = m.cookieNames.map((n) => c[n] ? `${n}=${c[n]}` : null).filter((x): x is string => x !== null);
	return p.length > 0 ? p.join('; ') : null;
}

function isInternal(m: SourceManifest) { return m.kind === 'internal'; }

function dedupeMangaSummaries(items: MangaSummary[]): MangaSummary[] {
	const seen = new Set<string>();
	const out: MangaSummary[] = [];
	for (const item of items) {
		const key = item.url || item.id;
		if (seen.has(key)) continue;
		seen.add(key);
		out.push(item);
	}
	return out;
}

// ── http ───────────────────────────────────────────────────────────

async function fetchE(
	req: HttpRequest, vars: Vars,
	cookies: Record<string, string> = {},
	m?: SourceManifest,
	scope: SourceCacheScope = 'metadata',
): Promise<Fetched> {
	const u = tpl(req.url, vars);
	const requestBody = req.body ? tpl(req.body, vars) : void 0;
	const h: Record<string, string> = tplHeaders({ ...(m?.requestHeaders ?? {}), ...(req.headers ?? {}) }, vars) ?? {};
	if (m) { const c = cookieHdr(m, cookies); if (c) h.Cookie = c; }
	const r = await fetchSource(u, {
		headers: h,
		init: { method: req.method ?? 'GET', body: requestBody },
		cache: m ? sourceCache(m.id, scope, [req.method ?? 'GET', u, requestBody ?? '', vars], cookies) : undefined,
	});
	if (!r.ok) throw new Error(`HTTP ${r.status} on ${u}`);
	const parsedBody = req.parse === 'json' ? await r.json() : new DOMParser().parseFromString(await r.text(), 'text/html');
	return { url: u, parsed: parsedBody };
}

// ── select ─────────────────────────────────────────────────────────

function htmlRow(el: Element): Row {
	return { get: (s) => queryHtmlOne(el, s), raw: el };
}
function jsonRow(n: unknown): Row {
	return { get: (s) => { const v = queryJsonOne(n, s); return v == null ? null : typeof v === 'object' ? JSON.stringify(v) : String(v); }, raw: n };
}
function rootRow(p: Document | unknown, mode: 'html' | 'json'): Row {
	return mode === 'json' ? jsonRow(p) : { get: (s) => queryHtmlOne(p as Document, s), raw: p };
}
function rows(p: unknown, list: string, mode: 'html' | 'json'): Row[] {
	return mode === 'json' ? queryJsonAll(p, list).map(jsonRow) : queryHtmlAll(p as Document, list).map(htmlRow);
}

// ── resolve ────────────────────────────────────────────────────────

function resolve(row: Row, fields: Fields, globals: Vars): Record<string, string | null> {
	const o: Record<string, string | null> = {};
	for (const [k, d] of Object.entries(fields)) if (!d.startsWith('=')) o[k] = row.get(d);
	for (const [k, d] of Object.entries(fields)) if (d.startsWith('=')) o[k] = tpl(d.slice(1), { ...globals, ...o });
	return o;
}

function rootExtras(p: Document | unknown, mode: 'html' | 'json', e: Fields | undefined, g: Vars) {
	return e ? resolve(rootRow(p, mode), e, g) : {};
}

function keep(row: Row, preds: Fields, g: Vars) {
	for (const d of Object.values(preds)) {
		const v = d.startsWith('=') ? tpl(d.slice(1), g) : row.get(d);
		if (v == null) return false;
		const t = String(v).trim();
		if (!t || t === '0' || t === 'false' || t === 'null') return false;
	}
	return true;
}

function truthySel(row: Row, sel: string) {
	const v = row.get(sel);
	if (v == null) return false;
	const t = v.trim();
	return !!t && t !== '0' && t !== 'false' && t !== 'null';
}

// ── build: chapter ─────────────────────────────────────────────────

function chapterLabel(num: string, t: string | null) {
	const a = num.trim(), b = t?.trim() || '';
	if (a && b) return `Chương ${a} · ${b}`;
	if (a) return `Chương ${a}`;
	if (b) return b;
	return '?';
}

function chapterRow(row: Row, fields: ChapterFields, baseUrl: string, g: Vars, norm: ReturnType<typeof compileChapterNumberNorm>): MangaChapterRef | null {
	const f = resolve(row, fields as unknown as Fields, g);
	const u = absUrl(f.url, baseUrl); if (!u) return null;
	const n = f.number || '', t = f.title, l = f.label ?? chapterLabel(n, t);
	return {
		id: u, url: u, number: n,
		numberNorm: applyChapterNumberNorm(norm, { number: n, label: l }),
		title: t ?? null, label: l,
		date: f.date ?? null, language: f.language ?? null, scanlator: f.scanlator ?? null,
	};
}

function lastChapter(chapters: MangaChapterRef[]) {
	let b: MangaChapterRef | null = null, bk = -Infinity;
	for (const c of chapters) { const k = parseFloat(c.numberNorm); if (Number.isFinite(k) && k > bk) { b = c; bk = k; } }
	return b ?? chapters.at(-1) ?? null;
}

function inlineChapters(p: unknown, mode: 'html' | 'json', spec: ChapterListSpec, baseUrl: string, g: Vars, m: SourceManifest) {
	const norm = compileChapterNumberNorm(spec.chapterNumberNorm ?? m.chapterNumberNorm);
	return rows(p, spec.list, mode).map((r) => chapterRow(r, spec.fields, baseUrl, g, norm)).filter((c): c is MangaChapterRef => c !== null);
}

function externalChapterRows(rs: Row[], baseUrl: string, g: Vars, ep: ChaptersApiEndpoint, norm: ReturnType<typeof compileChapterNumberNorm>) {
	const filtered = ep.keepIf ? rs.filter((r) => keep(r, ep.keepIf!, g)) : rs;
	return filtered.map((r) => {
		const extras = ep.extras ? resolve(r, ep.extras, g) : {};
		return chapterRow(r, ep.fields, baseUrl, { ...g, ...extras }, norm);
	}).filter((c): c is MangaChapterRef => c !== null);
}

// ── build: page list ───────────────────────────────────────────────

function pageList(p: unknown, ep: { parse: 'html' | 'json'; list: string; fields: Fields }, vars: Vars, baseUrl: string) {
	const out: string[] = [];
	for (const r of rows(p, ep.list, ep.parse)) { const f = resolve(r, ep.fields, vars); const u = absUrl(f.url, baseUrl); if (u) out.push(u); }
	return out;
}

function pageSpec(p: unknown, ep: { parse: 'html' | 'json'; rootExtras?: Fields }, spec: { extras: Fields; iterate?: string; count?: string; template: string }, vars: Vars, sid: string) {
	const re = rootExtras(p, ep.parse, ep.rootExtras, vars);
	const root = rootRow(p, ep.parse);
	const se: Record<string, string | null> = { ...re };
	const skip = spec.iterate ?? spec.count;
	for (const [k, s] of Object.entries(spec.extras)) if (k !== skip) se[k] = root.get(s);
	const allVars: Vars = { ...vars, ...se };
	const out: string[] = [];
	if (spec.iterate) {
		const sel = spec.extras[spec.iterate];
		if (!sel) throw new Error(`pages.iterate "${spec.iterate}" not in extras`);
		const files = ep.parse === 'json' ? queryJsonAll(p, sel) : queryHtmlAll(p as Document, sel).map((el) => el.textContent);
		for (const f of files) { const t = typeof f === 'string' ? f : String(f ?? ''); if (t) out.push(tpl(spec.template, { ...allVars, file: t })); }
	} else if (spec.count) {
		const sel = spec.extras[spec.count];
		if (!sel) throw new Error(`pages.count "${spec.count}" not in extras`);
		const n = parseInt(root.get(sel) ?? '', 10);
		if (!Number.isFinite(n) || n <= 0) throw new Error(`pages.count invalid`);
		for (let i = 1; i <= n; i++) out.push(tpl(spec.template, { ...allVars, file: String(i) }));
	} else {
		throw new Error(`pages spec needs iterate or count (${sid})`);
	}
	return out;
}

// ── public ─────────────────────────────────────────────────────────

export async function fetchBrowse(
	m: SourceManifest, shelfId: string | { search: true }, args: BrowseArgs = {},
): Promise<MangaSummary[]> {
	if (isInternal(m)) return [];
	if (m.adapter) { const a = getAdapter(m.adapter); if (a?.fetchBrowse) return dedupeMangaSummaries(await a.fetchBrowse(m, shelfId, args)); }
	const ep = typeof shelfId === 'string' ? m.endpoints?.shelves.find((s) => s.id === shelfId)?.endpoint : m.endpoints?.search;
	if (!ep) return [];
	const page = args.page ?? 1;
	const off = ep.pagination?.type === 'offset' ? (page - 1) * ep.pagination.pageSize : 0;
	const vars: Vars = { q: args.q ?? '', page, offset: off, filterParams: args.filterParams ?? '' };
	const { url, parsed } = await fetchE(ep, vars, args.userCookies ?? {}, m, 'browse');
	const re = rootExtras(parsed, ep.parse, ep.rootExtras, vars);
	const g: Vars = { ...vars, ...re };
	const items = rows(parsed, ep.list, ep.parse)
		.filter((r) => ep.keepIf ? keep(r, ep.keepIf, g) : true)
		.map((r): MangaSummary | null => {
			const extras = ep.extras ? resolve(r, ep.extras, g) : {};
			const f = resolve(r, ep.fields as Fields, { ...g, ...extras });
			const u = absUrl(f.url, url); if (!u) return null;
			const summary: MangaSummary = { id: u, url: u, title: f.title || '(không tên)', cover: absUrl(f.cover, url) };
			if (m.imageHeaders) summary.coverHeaders = m.imageHeaders;
			return summary;
		})
		.filter((s): s is MangaSummary => s !== null);
	return dedupeMangaSummaries(items);
}

export async function fetchMangaDetail(
	m: SourceManifest, mangaUrl: string,
	args: { language?: string; userCookies?: Record<string, string> } = {},
): Promise<MangaDetail> {
	if (isInternal(m) || !m.endpoints?.manga) throw new Error(`Source ${m.id} has no manga endpoint`);
	const cookies = args.userCookies ?? {};
	if (m.adapter) { const a = getAdapter(m.adapter); if (a?.fetchMangaDetail) return a.fetchMangaDetail(m, mangaUrl, cookies); }
	const ep = m.endpoints.manga;
	const vars: Vars = { mangaUrl, language: args.language ?? m.languages[0], ...extract(mangaUrl, ep.extract) };
	const { url: base, parsed } = await fetchE(ep, vars, cookies, m, 'manga');
	const re = rootExtras(parsed, ep.parse, ep.rootExtras, vars);
	const rt = rootRow(parsed, ep.parse);
	const ex = ep.extras ? resolve(rt, ep.extras, { ...vars, ...re }) : {};
	const g: Vars = { ...vars, ...re, ...ex };
	const f = resolve(rt, ep.fields as Fields, g);

	let chapters: MangaChapterRef[];
	if (ep.chapters) chapters = inlineChapters(parsed, ep.parse, ep.chapters, base, g, m);
	else if (m.endpoints?.chaptersApi) chapters = await chaptersExternal(m.endpoints.chaptersApi, g, m, cookies);
	else chapters = [];

	if (f.updatedAt && chapters.length > 0 && chapters.every((c) => !c.date)) {
		const l = lastChapter(chapters); if (l) l.date = f.updatedAt;
	}
	const detail: MangaDetail = {
		id: mangaUrl, url: mangaUrl,
		title: f.title || '(không tên)', cover: absUrl(f.cover, base),
		description: f.description ?? null, author: f.author ?? null, status: f.status ?? null,
		chapters,
	};
	if (m.imageHeaders) detail.coverHeaders = m.imageHeaders;
	return detail;
}

async function chaptersExternal(
	ep: ChaptersApiEndpoint, vars: Vars, m: SourceManifest, cookies: Record<string, string>,
): Promise<MangaChapterRef[]> {
	const norm = compileChapterNumberNorm(ep.chapterNumberNorm ?? m.chapterNumberNorm);
	const ps = ep.pagination?.pageSize;
	if (!ps) {
		const { url, parsed } = await fetchE(ep, { ...vars, page: 1, offset: 0 }, cookies, m, 'chapters');
		return externalChapterRows(rows(parsed, ep.list, ep.parse), url, vars, ep, norm);
	}
	const all: MangaChapterRef[] = [];
	const seen = new Set<string>();
	const hasMoreSel = ep.pagination?.hasMore;
	let page = 1, off = 0;
	while (true) {
		const { url, parsed } = await fetchE(ep, { ...vars, page, offset: off }, cookies, m, 'chapters');
		const rs = rows(parsed, ep.list, ep.parse);
		let added = 0;
		for (const c of externalChapterRows(rs, url, vars, ep, norm)) {
			if (seen.has(c.url)) continue;
			seen.add(c.url); all.push(c); added += 1;
		}
		// Stop on short page, when the source signals no more pages, or when an
		// out-of-range page yields no new chapters (some APIs clamp overflow
		// pages to the last page instead of returning an empty list).
		if (rs.length < ps || added === 0) break;
		if (hasMoreSel && !truthySel(rootRow(parsed, ep.parse), hasMoreSel)) break;
		page += 1; off += ps;
	}
	return all;
}

export async function fetchChapterPages(
	m: SourceManifest, chapterUrl: string, cookies: Record<string, string> = {},
): Promise<ChapterPages> {
	if (isInternal(m)) throw new Error(`Source ${m.id} has no chapter endpoint`);
	if (m.adapter) { const a = getAdapter(m.adapter); if (!a) throw new Error(`Unknown adapter: "${m.adapter}"`); return a.fetchChapterPages(m, chapterUrl, cookies); }
	const ep = m.endpoints?.chapter;
	if (!ep) throw new Error(`Source ${m.id} has no chapter endpoint`);
	const vars: Vars = { chapterUrl, ...extract(chapterUrl, ep.extract) };
	const { url: reqUrl, parsed } = await fetchE(ep, vars, cookies, m, 'chapter');
	const pageHeaders = tplHeaders({ ...(m.imageHeaders ?? {}), ...(ep.pageHeaders ?? {}) }, vars);

	const l = ep.list, f = ep.fields;
	if (l && f) return { url: chapterUrl, pages: pageList(parsed, { parse: ep.parse, list: l, fields: f as Fields }, vars, reqUrl), pageHeaders };
	if (ep.pages) {
		const pages = pageSpec(parsed, ep, ep.pages, vars, m.id);
		return { url: chapterUrl, pages, pageHeaders };
	}
	throw new Error(`chapter endpoint missing list/pages spec (${m.id})`);
}

export async function resolvePageUrl(
	m: SourceManifest, token: string, cookies: Record<string, string> = {},
): Promise<string> {
	if (m.adapter) { const a = getAdapter(m.adapter); if (a?.resolvePageUrl) return a.resolvePageUrl(m, token, cookies); }
	throw new Error(`resolvePageUrl called on source "${m.id}" which has no adapter.resolvePageUrl`);
}
