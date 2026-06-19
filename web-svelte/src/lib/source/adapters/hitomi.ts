import { fetchSource } from '$lib/sourceFetch.svelte';
import type { BrowseArgs, ChapterPages, MangaDetail, MangaSummary, SourceManifest } from '../types';
import type { SourceAdapter } from './types';

const DOMAIN2 = 'gold-usergeneratedcontent.net';
const LTN = `https://ltn.${DOMAIN2}`;
const PAGE_SIZE = 25;

interface GgData {
	m: (g: number) => number;
	s: (hash: string) => string;
	b: string;
	fetchedAt: number;
}

let ggCache: GgData | null = null;
const GG_TTL = 60 * 60 * 1000;

async function fetchGg(): Promise<GgData> {
	const now = Date.now();
	if (ggCache && now - ggCache.fetchedAt < GG_TTL) return ggCache;

	const res = await fetchSource(`${LTN}/gg.js`, {
		headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
	});
	if (!res.ok) throw new Error(`hitomi gg.js: HTTP ${res.status}`);
	const text = await res.text();

	const b = /\bb:\s*'([^']+)'/.exec(text)?.[1] ?? '';
	const set1 = new Set<number>();
	const switchBody = /switch\s*\(g\)\s*\{([\s\S]+?)\}\s*return/.exec(text)?.[1] ?? '';
	const segments = switchBody.split(/o\s*=\s*(\d)/);
	for (let i = 0; i < segments.length - 1; i += 2) {
		const value = parseInt(segments[i + 1]!, 10);
		if (value !== 1) continue;
		const cases = segments[i]!.match(/case\s+(\d+):/g) ?? [];
		for (const item of cases) {
			set1.add(parseInt(item.replace(/case\s+/, '').replace(':', ''), 10));
		}
	}

	ggCache = {
		m: (g: number) => (set1.has(g) ? 1 : 0),
		s: (hash: string) => {
			const match = /(..)(.)$/.exec(hash);
			return match ? parseInt(match[2]! + match[1]!, 16).toString(10) : '0';
		},
		b,
		fetchedAt: now,
	};
	return ggCache;
}

function subdomainFromHash(hash: string, ext: 'webp' | 'avif', gg: GgData): string {
	const match = /([0-9a-f]{2})([0-9a-f])$/.exec(hash);
	if (!match) return ext === 'avif' ? 'a1' : 'w1';
	const g = parseInt(match[2]! + match[1]!, 16);
	const prefix = ext === 'avif' ? 'a' : 'w';
	return prefix + (1 + gg.m(g));
}

function imageUrl(hash: string, ext: 'webp' | 'avif', gg: GgData): string {
	const subdomain = subdomainFromHash(hash, ext, gg);
	return `https://${subdomain}.${DOMAIN2}/${gg.b}${gg.s(hash)}/${hash}.${ext}`;
}

const GALLERY_ID_RE = /-(\d+)\.html/;

function extractGalleryId(url: string): string | null {
	return GALLERY_ID_RE.exec(url)?.[1] ?? null;
}

interface NozomiEntry {
	promise: Promise<ArrayBuffer>;
	fetchedAt: number;
}

const nozomiCache = new Map<string, NozomiEntry>();
const NOZOMI_TTL = 10 * 60_000;

function getNozomiBuffer(nozomiUrl: string): Promise<ArrayBuffer> {
	const now = Date.now();
	const cached = nozomiCache.get(nozomiUrl);
	if (cached && now - cached.fetchedAt < NOZOMI_TTL) return cached.promise;

	const promise = fetchSource(nozomiUrl, {
		headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
	}).then((res) => {
		if (!res.ok) throw new Error(`hitomi nozomi: HTTP ${res.status}`);
		return res.arrayBuffer();
	});
	nozomiCache.set(nozomiUrl, { promise, fetchedAt: now });
	return promise;
}

async function fetchNozomiPage(nozomiUrl: string, page: number): Promise<number[]> {
	const buffer = await getNozomiBuffer(nozomiUrl);
	const slice = buffer.slice(page * PAGE_SIZE * 4, page * PAGE_SIZE * 4 + PAGE_SIZE * 4);
	const view = new DataView(slice);
	const ids: number[] = [];
	for (let i = 0; i + 3 < slice.byteLength; i += 4) {
		ids.push(view.getInt32(i, false));
	}
	return ids;
}

interface GalleryBlock {
	id: number;
	url: string;
	title: string;
	cover: string | null;
}

const galleryBlockCache = new Map<number, Promise<GalleryBlock>>();

function fetchGalleryBlock(id: number): Promise<GalleryBlock> {
	if (galleryBlockCache.has(id)) return galleryBlockCache.get(id)!;

	const promise = fetchSource(`${LTN}/galleryblock/${id}.html`, {
		headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
	}).then(async (res) => {
		if (!res.ok) throw new Error(`hitomi galleryblock ${id}: HTTP ${res.status}`);
		const doc = new DOMParser().parseFromString(await res.text(), 'text/html');
		const href = doc.querySelector('h1.lillie a')?.getAttribute('href') ?? '';
		const title = doc.querySelector('h1.lillie a')?.textContent?.trim() ?? `#${id}`;
		const img = doc.querySelector('a.lillie img');
		const cover = img?.getAttribute('data-src') ?? img?.getAttribute('src') ?? null;
		return {
			id,
			url: href ? `https://hitomi.la${href}` : `https://hitomi.la/galleries/${id}.html`,
			title,
			cover: cover ? (cover.startsWith('//') ? 'https:' + cover : cover) : null,
		};
	});

	galleryBlockCache.set(id, promise);
	return promise;
}

interface GalleryFile {
	name: string;
	hash: string;
	hasavif?: number;
	haswebp?: number;
}

interface GalleryInfo {
	title: string | null;
	japanese_title: string | null;
	galleryurl: string;
	language: string | null;
	type: string | null;
	artists: Array<{ artist: string }> | null;
	tags: Array<{ tag: string; female?: string; male?: string }> | null;
	files: GalleryFile[];
	datepublished: string | null;
}

async function fetchGalleryInfo(id: string): Promise<GalleryInfo> {
	const res = await fetchSource(`${LTN}/galleries/${id}.js`, {
		headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
	});
	if (!res.ok) throw new Error(`hitomi galleries/${id}.js: HTTP ${res.status}`);
	const text = await res.text();
	const json = text.replace(/^var galleryinfo\s*=\s*/, '').replace(/;\s*$/, '');
	return JSON.parse(json) as GalleryInfo;
}

function nozomiUrlFromFilterState(state: Record<string, string | string[]> | undefined): string {
	const get = (key: string): string | null => {
		const value = state?.[key];
		return typeof value === 'string' && value !== 'all' ? value : null;
	};
	const type = get('type');
	const lang = get('language');

	if (type && lang) return `${LTN}/n/type/${encodeURIComponent(type)}-${encodeURIComponent(lang)}.nozomi`;
	if (type) return `${LTN}/n/type/${encodeURIComponent(type)}-all.nozomi`;
	if (lang) return `${LTN}/n/index-${encodeURIComponent(lang)}.nozomi`;
	return `${LTN}/n/index-all.nozomi`;
}

export const hitomiAdapter: SourceAdapter = {
	async fetchBrowse(
		_manifest: SourceManifest,
		shelfOrSearch: string | { search: true },
		args: BrowseArgs = {},
	): Promise<MangaSummary[]> {
		const page = Math.max(0, (args.page ?? 1) - 1);
		const nozomiUrl = typeof shelfOrSearch !== 'string' && args.q?.trim()
			? `${LTN}/n/${encodeURIComponent(args.q.trim().toLowerCase().replace(/\s+/g, '_'))}-all.nozomi`
			: nozomiUrlFromFilterState(args.filterState);

		const ids = await fetchNozomiPage(nozomiUrl, page);
		if (!ids.length) return [];

		void fetchNozomiPage(nozomiUrl, page + 1).catch(() => {});
		const blocks = await Promise.allSettled(ids.map(fetchGalleryBlock));
		return blocks
			.filter((result): result is PromiseFulfilledResult<GalleryBlock> => result.status === 'fulfilled')
			.map(({ value }) => ({
				id: value.url,
				url: value.url,
				title: value.title,
				cover: value.cover,
			}));
	},

	async fetchMangaDetail(
		_manifest: SourceManifest,
		mangaUrl: string,
		_userCookies: Record<string, string>,
	): Promise<MangaDetail> {
		const id = extractGalleryId(mangaUrl);
		if (!id) throw new Error(`hitomi: cannot extract gallery ID from: ${mangaUrl}`);

		const info = await fetchGalleryInfo(id);
		const block = await fetchGalleryBlock(parseInt(id, 10));
		const langMap: Record<string, string> = {
			japanese: 'ja',
			chinese: 'zh',
			english: 'en',
			korean: 'ko',
		};
		const lang = info.language ? (langMap[info.language] ?? info.language) : null;
		const title = info.title ?? info.japanese_title ?? `Gallery #${id}`;
		const date = info.datepublished ? info.datepublished.slice(0, 10) : null;

		return {
			id: mangaUrl,
			url: mangaUrl,
			title,
			cover: block.cover,
			description: info.type ?? null,
			author: info.artists?.map((artist) => artist.artist).join(', ') ?? null,
			status: info.language ?? null,
			availableLanguages: lang ? [lang] : null,
			chapters: [{
				id: mangaUrl,
				url: mangaUrl,
				number: '1',
				numberNorm: '1',
				label: `${info.files.length} pages`,
				title: null,
				date,
				language: lang,
				scanlator: null,
			}],
		};
	},

	async fetchChapterPages(
		_manifest: SourceManifest,
		chapterUrl: string,
		_userCookies: Record<string, string>,
	): Promise<ChapterPages> {
		const id = extractGalleryId(chapterUrl);
		if (!id) throw new Error(`hitomi: cannot extract gallery ID from: ${chapterUrl}`);

		const [info, gg] = await Promise.all([fetchGalleryInfo(id), fetchGg()]);
		const pages = info.files.map((file) => imageUrl(file.hash, file.hasavif ? 'avif' : 'webp', gg));
		return { url: chapterUrl, pages };
	},
};
