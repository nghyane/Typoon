import { fetchSource } from '$lib/sourceFetch.svelte';
import { sourceCache, type SourceCacheScope } from '../runtime/cache';
import type { ChapterPages, MangaDetail, SourceManifest } from '../types';
import type { SourceAdapter } from './types';

const GALLERY_RE = /nhentai\.net\/g\/(\d+)/;

const LANG_MAP: Record<string, string> = {
	english: 'en',
	japanese: 'ja',
	chinese: 'zh',
};

function cookieHeader(userCookies: Record<string, string>): string | null {
	const entries = Object.entries(userCookies);
	return entries.length ? entries.map(([key, value]) => `${key}=${value}`).join('; ') : null;
}

function headers(userCookies: Record<string, string>): Record<string, string> {
	const out: Record<string, string> = { Referer: 'https://nhentai.net/' };
	const cookie = cookieHeader(userCookies);
	if (cookie) out.Cookie = cookie;
	return out;
}

async function fetchHtml(
	manifest: SourceManifest,
	url: string,
	scope: SourceCacheScope,
	userCookies: Record<string, string>,
): Promise<Document> {
	const res = await fetchSource(url, {
		headers: headers(userCookies),
		cache: sourceCache(manifest.id, scope, [url], userCookies),
	});
	if (!res.ok) throw new Error(`nHentai: HTTP ${res.status} on ${url}`);
	return new DOMParser().parseFromString(await res.text(), 'text/html');
}

function extractGalleryId(url: string): string | null {
	return GALLERY_RE.exec(url)?.[1] ?? null;
}

function text(el: Element | null): string | null {
	const value = el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
	return value || null;
}

function attr(el: Element | null, name: string): string | null {
	return el?.getAttribute(name) ?? null;
}

function names(doc: Document, selector: string): string[] {
	return Array.from(doc.querySelectorAll(selector))
		.map((el) => text(el))
		.filter((value): value is string => !!value);
}

function pageImageFromThumb(src: string | null): string | null {
	if (!src) return null;
	let url: URL;
	try { url = new URL(src); } catch { return null; }

	const parts = url.pathname.split('/');
	const file = parts.at(-1) ?? '';
	const match = /^(\d+)t\.([a-z0-9]+)(?:\.webp)?$/i.exec(file);
	if (!match?.[1] || !match[2]) return null;

	url.hostname = url.hostname.replace(/^t/, 'i');
	parts[parts.length - 1] = `${match[1]}.${match[2].toLowerCase()}`;
	url.pathname = parts.join('/');
	url.search = '';
	return url.href;
}

function pageImages(doc: Document): string[] {
	return Array.from(doc.querySelectorAll('#thumbnail-container a.gallerythumb img'))
		.map((img) => pageImageFromThumb(attr(img, 'src')))
		.filter((value): value is string => !!value);
}

function language(doc: Document): string | null {
	const raw = text(doc.querySelector('#tags a[href^="/language/"] .name'));
	return raw ? (LANG_MAP[raw.toLowerCase()] ?? raw.toLowerCase()) : null;
}

export const nhentaiAdapter: SourceAdapter = {
	async fetchMangaDetail(
		manifest: SourceManifest,
		mangaUrl: string,
		userCookies: Record<string, string>,
	): Promise<MangaDetail> {
		const galleryId = extractGalleryId(mangaUrl);
		if (!galleryId) throw new Error(`nHentai: cannot extract gallery ID from: ${mangaUrl}`);

		const url = `https://nhentai.net/g/${galleryId}/`;
		const doc = await fetchHtml(manifest, url, 'manga', userCookies);
		const pages = pageImages(doc);
		const lang = language(doc);
		const date = attr(doc.querySelector('#tags time[datetime]'), 'datetime')?.slice(0, 10) ?? null;
		const categories = names(doc, '#tags a[href^="/category/"] .name');

		return {
			id: mangaUrl,
			url: mangaUrl,
			title: text(doc.querySelector('h1.title .pretty')) ?? text(doc.querySelector('h1.title')) ?? `Gallery #${galleryId}`,
			cover: attr(doc.querySelector('#cover img'), 'src'),
			coverHeaders: manifest.imageHeaders,
			description: categories.join(', ') || null,
			author: names(doc, '#tags a[href^="/artist/"] .name').join(', ') || null,
			status: categories[0] ?? null,
			
			chapters: [{
				id: url,
				url,
				number: '1',
				numberNorm: '1',
				label: pages.length > 0 ? `${pages.length} pages` : 'Ch.1',
				title: null,
				date,
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
		const galleryId = extractGalleryId(chapterUrl);
		if (!galleryId) throw new Error(`nHentai: cannot extract gallery ID from: ${chapterUrl}`);

		const url = `https://nhentai.net/g/${galleryId}/`;
		const doc = await fetchHtml(manifest, url, 'chapter', userCookies);
		const pages = pageImages(doc);
		if (!pages.length) throw new Error(`nHentai: no page thumbnails found for gallery ${galleryId}`);
		return { url: chapterUrl, pages, pageHeaders: manifest.imageHeaders };
	},
};
