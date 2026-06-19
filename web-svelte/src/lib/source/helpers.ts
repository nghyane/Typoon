import type { MangaChapterRef } from './types';

export function chapterSortKey(chapter: MangaChapterRef): number {
	const value = Number.parseFloat(chapter.numberNorm);
	return Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
}
