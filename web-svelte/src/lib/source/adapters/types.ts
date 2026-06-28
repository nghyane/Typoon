import type { BrowseArgs, ChapterPages, MangaDetail, MangaSummary, SourceManifest } from '../types';

export interface SourceAdapter {
	fetchChapterPages(
		manifest: SourceManifest,
		chapterUrl: string,
		userCookies: Record<string, string>,
	): Promise<ChapterPages>;

	resolvePageUrl?(
		manifest: SourceManifest,
		token: string,
		userCookies: Record<string, string>,
		/** `refresh` forces signing data (e.g. Hitomi gg.js) past its cache after a page miss. */
		opts?: { refresh?: boolean },
	): Promise<string>;

	fetchMangaDetail?(
		manifest: SourceManifest,
		mangaUrl: string,
		userCookies: Record<string, string>,
	): Promise<MangaDetail>;

	fetchBrowse?(
		manifest: SourceManifest,
		shelfId: string | { search: true },
		args: BrowseArgs,
	): Promise<MangaSummary[]>;
}
