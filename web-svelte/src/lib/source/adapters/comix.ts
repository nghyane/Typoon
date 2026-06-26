import { normalizeLang } from '$lib/lang';
import type { BrowseArgs, ChapterPages, MangaChapterRef, MangaDetail, MangaSummary, SourceManifest } from '../types';
import type { SourceAdapter } from './types';

const ORIGIN = 'https://comix.to';
const API_BASE = ORIGIN + '/api/v1';
const PAGE_SIZE = 28;
const SAFE_RATINGS = ['safe', 'suggestive'];

// ── Types ─────────────────────────────────────────────────────────

interface ComixManga { id: number; hid: string; title: string; url: string; poster?: { medium?: string; large?: string }; synopsis?: string; status?: string }
interface ComixChapter { id: number; number: number | string; name?: string; url: string; language?: string; createdAtFormatted?: string; group?: { name?: string }; isOfficial?: boolean }
interface ComixChapterDetail extends ComixChapter { pages?: { baseUrl?: string; items?: Array<{ url: string }> } | Array<{ url: string }> }

// ── Worker ────────────────────────────────────────────────────────

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (t: string) => void; reject: (e: Error) => void }>();

function getWorker(): Worker {
	if (!worker) {
		worker = new Worker(new URL('./comix-worker.ts', import.meta.url), { type: 'module' });
		worker.addEventListener('message', (ev: MessageEvent<{ id: number; token?: string; error?: string }>) => {
			const p = pending.get(ev.data.id);
			if (!p) return;
			pending.delete(ev.data.id);
			if (ev.data.error) p.reject(new Error(ev.data.error));
			else p.resolve(ev.data.token!);
		});
	}
	return worker;
}

function generateToken(params: Record<string, unknown>): Promise<string> {
	return new Promise((resolve, reject) => {
		const id = nextId++;
		pending.set(id, { resolve, reject });
		getWorker().postMessage({ id, params });
	});
}

// ── API ───────────────────────────────────────────────────────────

async function apiGet<T>(path: string, params: Record<string, unknown> = {}): Promise<T> {
	const token = await generateToken(params);
	const url = new URL(API_BASE + path);
	for (const [k, v] of Object.entries(params)) {
		if (v == null) continue;
		if (Array.isArray(v)) v.forEach(x => url.searchParams.append(k + '[]', String(x)));
		else if (typeof v === 'object') for (const [sk, sv] of Object.entries(v as Record<string, unknown>)) url.searchParams.set(k + '[' + sk + ']', String(sv));
		else url.searchParams.set(k, String(v));
	}
	url.searchParams.set('_', token);

	const res = await fetch(url.href, { headers: { Accept: 'application/json', 'X-Requested-With': 'XMLHttpRequest' } });
	if (!res.ok) throw new Error(`Comix API HTTP ${res.status} on ${path}`);
	const data = await res.json();
	if (data?.status === 'ok' && data.result) return data.result as T;
	return data as T;
}

// ── Adapter ───────────────────────────────────────────────────────

export const comixAdapter: SourceAdapter = {
	async fetchBrowse(_m: SourceManifest, shelf: string | { search: true }, args: BrowseArgs): Promise<MangaSummary[]> {
		const p: Record<string, unknown> = { page: args.page ?? 1, limit: PAGE_SIZE, content_rating: SAFE_RATINGS };
		if (typeof shelf !== 'string') { p.keyword = args.q ?? ''; p.order = { relevance: 'desc' }; }
		else if (shelf === 'popular') p.order = { follows_total: 'desc' };
		else { p.order = { chapter_updated_at: 'desc' }; p.scope = 'hot'; }
		const data = await apiGet<{ items?: ComixManga[] }>('/manga', p);
		return (data.items ?? []).map(summary);
	},
	async fetchMangaDetail(_m: SourceManifest, url: string): Promise<MangaDetail> {
		const hid = mangaHid(url); if (!hid) throw new Error(`Bad Comix URL: ${url}`);
		const [manga, chapters] = await Promise.all([apiGet<ComixManga>(`/manga/${hid}`), fetchChapters(hid)]);
		return { ...summary(manga), description: manga.synopsis ?? null, author: null, status: manga.status ?? null, availableLanguages: null, chapters };
	},
	async fetchChapterPages(_m: SourceManifest, url: string): Promise<ChapterPages> {
		const id = chapterId(url); if (!id) throw new Error(`Bad Comix URL: ${url}`);
		const ch = await apiGet<ComixChapterDetail>(`/chapters/${id}`);
		const raw = ch.pages;
		const pages = Array.isArray(raw) ? raw.map(p => p.url) : (raw?.items ?? []).map(p => new URL(p.url, (raw as { baseUrl?: string }).baseUrl || ORIGIN).href);
		return { url: absolute(ch.url || url), pages, pageHeaders: { Referer: ORIGIN + '/' } };
	},
};

async function fetchChapters(hid: string): Promise<MangaChapterRef[]> {
	const out: MangaChapterRef[] = [];
	for (let p = 1; p < 100; p++) {
		const data = await apiGet<{ items?: ComixChapter[]; meta?: { hasNext?: boolean } }>(`/manga/${hid}/chapters`, { page: p, limit: 100, order: { number: 'desc' } });
		for (const item of data.items ?? []) out.push(chapterRef(item));
		if (!data.meta?.hasNext) break;
	}
	return out;
}

function summary(m: ComixManga): MangaSummary { const u = absolute(m.url); return { id: u, url: u, title: m.title || '(không tên)', cover: m.poster?.medium ?? m.poster?.large ?? null, coverHeaders: { Referer: ORIGIN + '/' } }; }
function chapterRef(c: ComixChapter): MangaChapterRef { const u = absolute(c.url); const n = String(c.number ?? ''); const t = c.name?.trim() || null; return { id: u, url: u, number: n, numberNorm: n, title: t, label: t ? `Ch ${n}: ${t}` : `Ch ${n}`, date: c.createdAtFormatted ?? null, language: normalizeLang(c.language), scanlator: c.group?.name ?? (c.isOfficial ? 'Official' : null) }; }
function mangaHid(url: string): string | null { return /\/title\/([^/-]+)/.exec(url)?.[1] ?? null; }
function chapterId(url: string): string | null { return /\/([0-9]+)-chapter(?:-|$)/.exec(url)?.[1] ?? null; }
function absolute(url: string): string { return new URL(url, ORIGIN).href; }
