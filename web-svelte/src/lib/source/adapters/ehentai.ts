import { fetchSource } from '$lib/sourceFetch.svelte';
import { queryHtmlAll } from '../selectors';
import type { ChapterPages, MangaDetail, SourceManifest } from '../types';
import type { SourceAdapter } from './types';

const API = 'https://api.e-hentai.org/api.php';
const GALLERY_RE = /e-hentai\.org\/g\/(\d+)\/([a-f0-9]+)/;
const THUMBS_PER_PAGE = 20;
const PAGE_HASH_RE = /\/s\/([a-f0-9]+)\/\d+-(\d+)/;
const GDATA_TTL = 60 * 60 * 24 * 30;
const THUMB_TTL = 60 * 60 * 24 * 30;
const READER_TTL = 60 * 15;
const SHOWPAGE_TTL = 60 * 5;

const thumbCache = new Map<string, Promise<Map<number, string>>>();

function parseGalleryUrl(url: string): { gid: string; token: string } | null {
	const match = GALLERY_RE.exec(url);
	if (!match) return null;
	return { gid: match[1]!, token: match[2]! };
}

function cookieHeader(userCookies: Record<string, string>): string | null {
	const entries = Object.entries(userCookies);
	return entries.length ? entries.map(([key, value]) => `${key}=${value}`).join('; ') : null;
}

function cacheKey(base: string, userCookies: Record<string, string>): string {
	const cookie = cookieHeader(userCookies);
	return cookie ? `${base}:ck:${hashString(cookie)}` : base;
}

function hashString(value: string): string {
	let h = 2166136261;
	for (let i = 0; i < value.length; i += 1) {
		h ^= value.charCodeAt(i);
		h = Math.imul(h, 16777619);
	}
	return (h >>> 0).toString(36);
}

function ttlCache(key: string, ttl: number) {
	return { policy: 'ttl' as const, key, ttl };
}

async function fetchHtml(
	url: string,
	userCookies: Record<string, string>,
	cache?: ReturnType<typeof ttlCache>,
): Promise<Document> {
	const headers: Record<string, string> = { Referer: 'https://e-hentai.org/' };
	const cookie = cookieHeader(userCookies);
	if (cookie) headers.Cookie = cookie;
	const res = await fetchSource(url, { headers, cache });
	if (!res.ok) throw new Error(`E-Hentai: HTTP ${res.status} on ${url}`);
	return new DOMParser().parseFromString(await res.text(), 'text/html');
}

async function postApi(
	body: object,
	userCookies: Record<string, string>,
	cache?: ReturnType<typeof ttlCache>,
): Promise<Record<string, unknown>> {
	const headers: Record<string, string> = { 'Content-Type': 'application/json' };
	const cookie = cookieHeader(userCookies);
	if (cookie) headers.Cookie = cookie;
	const res = await fetchSource(API, {
		headers,
		init: { method: 'POST', body: JSON.stringify(body) },
		cache,
	});
	if (!res.ok) throw new Error(`E-Hentai API: HTTP ${res.status}`);
	return res.json() as Promise<Record<string, unknown>>;
}

async function fetchGdata(gid: string, token: string, userCookies: Record<string, string>): Promise<Record<string, unknown>> {
	const json = await postApi(
		{ method: 'gdata', gidlist: [[parseInt(gid, 10), token]], namespace: 1 },
		userCookies,
		ttlCache(cacheKey(`eh:gdata:v1:${gid}:${token}`, userCookies), GDATA_TTL),
	);
	const meta = (json as { gmetadata?: unknown[] }).gmetadata?.[0];
	if (!meta || typeof meta !== 'object') throw new Error('E-Hentai gdata: no metadata');
	return meta as Record<string, unknown>;
}

function parseThumbPage(doc: Document): Map<number, string> {
	const map = new Map<number, string>();
	for (const el of queryHtmlAll(doc, '#gdt a')) {
		const match = PAGE_HASH_RE.exec(el.getAttribute('href') ?? '');
		if (match) map.set(parseInt(match[2]!, 10), match[1]!);
	}
	return map;
}

function getThumbPage(
	gid: string,
	galleryToken: string,
	thumbPageIdx: number,
	userCookies: Record<string, string>,
): Promise<Map<number, string>> {
	const key = cacheKey(`${gid}/${galleryToken}/${thumbPageIdx}`, userCookies);
	if (!thumbCache.has(key)) {
		const promise = fetchHtml(
			`https://e-hentai.org/g/${gid}/${galleryToken}/?p=${thumbPageIdx}`,
			userCookies,
			ttlCache(cacheKey(`eh:thumb:v1:${gid}:${galleryToken}:${thumbPageIdx}`, userCookies), THUMB_TTL),
		)
			.then(parseThumbPage);
		thumbCache.set(key, promise);
	}
	return thumbCache.get(key)!;
}

async function getHash(
	gid: string,
	galleryToken: string,
	pageNum: number,
	userCookies: Record<string, string>,
): Promise<string | null> {
	const thumbPageIdx = Math.floor((pageNum - 1) / THUMBS_PER_PAGE);
	const map = await getThumbPage(gid, galleryToken, thumbPageIdx, userCookies);
	return map.get(pageNum) ?? null;
}

async function resolveUrl(
	gid: string,
	page: number,
	galleryToken: string,
	showkey: string,
	userCookies: Record<string, string>,
): Promise<string> {
	const hash = await getHash(gid, galleryToken, page, userCookies);
	if (!hash) throw new Error(`E-Hentai: hash missing for page ${page} (gid ${gid})`);

	const json = await postApi(
		{ method: 'showpage', gid: parseInt(gid, 10), page, imgkey: hash, showkey },
		userCookies,
		ttlCache(cacheKey(`eh:showpage:v1:${gid}:${page}:${hash}:${showkey}`, userCookies), SHOWPAGE_TTL),
	);
	const i3Html = (json.i3 as string | undefined) ?? '';
	const match = /id="img"[^>]*src="([^"]+)"/.exec(i3Html);
	if (!match?.[1]) throw new Error(`E-Hentai showpage: no img in i3 for gid ${gid} page ${page}`);
	return match[1];
}

function encodeToken(gid: string, page: number, galleryToken: string, showkey: string): string {
	return `${gid}\x00${page}\x00${galleryToken}\x00${showkey}`;
}

function decodeToken(token: string): { gid: string; page: number; galleryToken: string; showkey: string } | null {
	const parts = token.split('\x00');
	if (parts.length !== 4) return null;
	const page = parseInt(parts[1]!, 10);
	if (!Number.isFinite(page)) return null;
	return { gid: parts[0]!, page, galleryToken: parts[2]!, showkey: parts[3]! };
}

export const ehentaiAdapter: SourceAdapter = {
	async fetchMangaDetail(
		manifest: SourceManifest,
		mangaUrl: string,
		userCookies: Record<string, string>,
	): Promise<MangaDetail> {
		const ids = parseGalleryUrl(mangaUrl);
		if (!ids) throw new Error(`E-Hentai: cannot parse URL: ${mangaUrl}`);

		const meta = await fetchGdata(ids.gid, ids.token, userCookies);
		const title = (meta.title as string | undefined) ?? (meta.title_jpn as string | undefined) ?? '(không tên)';
		const tags = (meta.tags as string[] | undefined) ?? [];
		const langTag = tags.find((tag) => tag.startsWith('language:'));
		const lang = langTag ? langTag.replace('language:', '') : null;
		const pageCount = parseInt(String(meta.filecount ?? '0'), 10);
		const date = meta.posted
			? new Date(parseInt(String(meta.posted), 10) * 1000).toISOString().slice(0, 10)
			: null;

		return {
			id: mangaUrl,
			url: mangaUrl,
			title,
			cover: (meta.thumb as string | undefined) ?? null,
			coverHeaders: manifest.imageHeaders,
			description: (meta.category as string | undefined) ?? null,
			author: (meta.uploader as string | undefined) ?? null,
			status: null,
			genres: null,
			
			chapters: [{
				id: mangaUrl,
				url: mangaUrl,
				number: '1',
				numberNorm: '1',
				label: pageCount > 0 ? `${pageCount} pages` : 'Ch.1',
				title: null,
				date,
				language: lang,
				scanlator: (meta.uploader as string | undefined) ?? null,
			}],
		};
	},

	async fetchChapterPages(
		manifest: SourceManifest,
		chapterUrl: string,
		userCookies: Record<string, string>,
	): Promise<ChapterPages> {
		const ids = parseGalleryUrl(chapterUrl);
		if (!ids) throw new Error(`E-Hentai: cannot parse URL: ${chapterUrl}`);

		const { gid, token: galleryToken } = ids;
		const meta = await fetchGdata(gid, galleryToken, userCookies);
		const pageCount = parseInt(String(meta.filecount ?? '0'), 10);
		if (!pageCount) throw new Error(`E-Hentai: filecount 0 for ${chapterUrl}`);

		const thumbPromise = getThumbPage(gid, galleryToken, 0, userCookies);
		const readerPromise = thumbPromise.then(async (thumbMap) => {
			const hash1 = thumbMap.get(1);
			if (!hash1) throw new Error('E-Hentai: hash missing for page 1');
			return fetchHtml(
				`https://e-hentai.org/s/${hash1}/${gid}-1`,
				userCookies,
				ttlCache(cacheKey(`eh:reader:v1:${gid}:${hash1}:1`, userCookies), READER_TTL),
			);
		});
		const [, readerDoc] = await Promise.all([thumbPromise, readerPromise]);

		let showkey = '';
		for (const script of Array.from(readerDoc.querySelectorAll('script'))) {
			const match = /var showkey="([^"]+)"/.exec(script.textContent ?? '');
			if (match) {
				showkey = match[1]!;
				break;
			}
		}
		if (!showkey) throw new Error('E-Hentai: showkey not found');

		const pages = new Array<string>(pageCount).fill('');
		const tokens = Array.from({ length: pageCount }, (_value, index) =>
			encodeToken(gid, index + 1, galleryToken, showkey),
		);

		return { url: chapterUrl, pages, tokens, pageHeaders: manifest.imageHeaders };
	},

	async resolvePageUrl(
		_manifest: SourceManifest,
		token: string,
		userCookies: Record<string, string>,
	): Promise<string> {
		const decoded = decodeToken(token);
		if (!decoded) throw new Error(`E-Hentai: invalid token "${token}"`);
		return resolveUrl(decoded.gid, decoded.page, decoded.galleryToken, decoded.showkey, userCookies);
	},
};
