import { fetchSource } from '$lib/sourceFetch.svelte';
import { normalizeLang } from '$lib/lang';
import { sourceCache } from '../runtime/cache';
import type { ChapterPages, MangaDetail, SourceManifest } from '../types';
import type { SourceAdapter } from './types';

const GALLERY_RE = /hentaifox\.com\/(?:gallery|g)\/(\d+)/;

const EXT_MAP: Record<string, string> = {
	j: 'jpg',
	p: 'png',
	w: 'webp',
	g: 'gif',
	b: 'bmp',
};

function pickCdn(uniqueId: number): string {
	return uniqueId > 140236 ? 'i3.hentaifox.com' : 'i.hentaifox.com';
}

async function fetchReaderPage(sourceId: string, galleryId: string, userCookies: Record<string, string>): Promise<Document> {
	const headers: Record<string, string> = { Referer: 'https://hentaifox.com/' };
	if (Object.keys(userCookies).length > 0) {
		headers.Cookie = Object.entries(userCookies).map(([key, value]) => `${key}=${value}`).join('; ');
	}
	const res = await fetchSource(`https://hentaifox.com/g/${galleryId}/1/`, {
		headers,
		cache: sourceCache(sourceId, 'chapter', ['reader', galleryId], userCookies),
	});
	if (!res.ok) throw new Error(`HentaiFox reader HTTP ${res.status}`);
	return new DOMParser().parseFromString(await res.text(), 'text/html');
}

function parseGth(doc: Document): Record<string, string> {
	for (const script of Array.from(doc.querySelectorAll('script'))) {
		const text = script.textContent ?? '';
		if (!text.includes('g_th')) continue;
		const match = /g_th\s*=\s*(?:\$\.parseJSON|JSON\.parse)\s*\(\s*['"](\{.+?\})['"]\s*\)/.exec(text);
		if (match?.[1]) {
			try { return JSON.parse(match[1]) as Record<string, string>; } catch { /* try next script */ }
		}
	}
	return {};
}

function inputVal(doc: Document, id: string): string {
	return (doc.querySelector(`input#${id}`) as HTMLInputElement | null)?.value ?? '';
}

async function fetchGalleryPage(sourceId: string, galleryId: string, userCookies: Record<string, string>): Promise<Document> {
	const headers: Record<string, string> = { Referer: 'https://hentaifox.com/' };
	if (Object.keys(userCookies).length > 0) {
		headers.Cookie = Object.entries(userCookies).map(([key, value]) => `${key}=${value}`).join('; ');
	}
	const res = await fetchSource(`https://hentaifox.com/gallery/${galleryId}/`, {
		headers,
		cache: sourceCache(sourceId, 'manga', ['gallery', galleryId], userCookies),
	});
	if (!res.ok) throw new Error(`HentaiFox gallery HTTP ${res.status}`);
	return new DOMParser().parseFromString(await res.text(), 'text/html');
}

// A gallery lists every language it carries (each `<li>` text is "english 152697"
// where the trailing number is a site-wide badge count); take the first word and
// drop pseudo-languages ("translated") via normalizeLang.
function detectLanguages(doc: Document): string[] {
	const codes = Array.from(doc.querySelectorAll('ul.languages li a'))
		.map((a) => normalizeLang((a.textContent ?? '').trim().split(/\s+/)[0]))
		.filter((code): code is string => !!code);
	return [...new Set(codes)];
}

export const hentaifoxAdapter: SourceAdapter = {
	async fetchMangaDetail(
		manifest: SourceManifest,
		mangaUrl: string,
		userCookies: Record<string, string>,
	): Promise<MangaDetail> {
		const galleryId = GALLERY_RE.exec(mangaUrl)?.[1];
		if (!galleryId) throw new Error(`HentaiFox: cannot extract galleryId from: ${mangaUrl}`);

		const doc = await fetchGalleryPage(manifest.id, galleryId, userCookies);
		const title = doc.querySelector('.gallery_title h1')?.textContent?.trim()
			|| doc.querySelector('h1')?.textContent?.trim()
			|| `Gallery #${galleryId}`;
		const cover = doc.querySelector('.cover img')?.getAttribute('src') ?? null;
		const langs = detectLanguages(doc);
		// Reliable only when the gallery carries exactly one language; multi-language
		// or untagged galleries stay null so translation auto-detects per page.
		const lang = langs.length === 1 ? langs[0]! : null;
		const readerUrl = `https://hentaifox.com/g/${galleryId}/1/`;

		return {
			id: mangaUrl,
			url: mangaUrl,
			title,
			cover,
			coverHeaders: manifest.imageHeaders,
			description: null,
			author: null,
			status: 'completed',
			availableLanguages: langs.length > 0 ? langs : null,
			chapters: [{
				id: readerUrl,
				url: readerUrl,
				number: '1',
				numberNorm: '1',
				label: 'Ch.1',
				title: null,
				date: null,
				language: lang,
				scanlator: null,
			}],
		};
	},

	async fetchChapterPages(
		manifest: SourceManifest,
		chapterUrl: string,
		userCookies: Record<string, string>,
	): Promise<ChapterPages> {
		const match = /hentaifox\.com\/(?:gallery|g)\/(\d+)/.exec(chapterUrl);
		if (!match?.[1]) throw new Error(`Cannot extract galleryId from: ${chapterUrl}`);

		const doc = await fetchReaderPage(manifest.id, match[1], userCookies);
		const totalStr = inputVal(doc, 'pages');
		const imageDir = inputVal(doc, 'image_dir');
		const galleryHash = inputVal(doc, 'gallery_id');
		const uniqueId = parseInt(inputVal(doc, 'unique_id') || '0', 10);
		const total = parseInt(totalStr, 10);

		if (!total || !imageDir || !galleryHash) {
			throw new Error(`HentaiFox: missing reader metadata (pages=${totalStr} dir=${imageDir} hash=${galleryHash})`);
		}

		const gth = parseGth(doc);
		const base = `https://${pickCdn(uniqueId)}/${imageDir}/${galleryHash}`;
		const pages: string[] = [];
		for (let i = 1; i <= total; i += 1) {
			const extCode = (gth[String(i)] ?? '').split(',')[0] ?? 'j';
			pages.push(`${base}/${i}.${EXT_MAP[extCode] ?? 'jpg'}`);
		}

		return { url: chapterUrl, pages, pageHeaders: manifest.imageHeaders };
	},
};
