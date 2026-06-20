import { fetchSource } from '$lib/sourceFetch.svelte';
import { sourceCache } from '../runtime/cache';
import type { ChapterPages, SourceManifest } from '../types';
import type { SourceAdapter } from './types';

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

export const hentaifoxAdapter: SourceAdapter = {
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
