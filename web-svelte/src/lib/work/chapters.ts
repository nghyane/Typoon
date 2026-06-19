import type { WorkSource } from '$lib/db';
import type { InstalledSource, MangaChapterRef } from '$lib/source/types';

export interface SourceChapterDetail {
	source: InstalledSource;
	origin: WorkSource;
	refs: MangaChapterRef[];
}

export interface SourceVersion {
	source: InstalledSource;
	origin: WorkSource;
	ref: MangaChapterRef;
	lang: string;
}

export interface MergedChapter {
	numberNorm: string;
	label: string;
	number: string;
	sortKey: number;
	sourceVersions: SourceVersion[];
}

export function mergeChapters(
	sourceChapters: SourceChapterDetail[],
	targetLang: string,
): MergedChapter[] {
	const map = new Map<string, MergedChapter>();
	const target = targetLang.toLowerCase();

	for (const sourceChapter of sourceChapters) {
		for (const ref of sourceChapter.refs) {
			const lang = (ref.language ?? sourceChapter.origin.languages[0] ?? target).toLowerCase();
			const version: SourceVersion = {
				source: sourceChapter.source,
				origin: sourceChapter.origin,
				ref,
				lang,
			};
			const current = map.get(ref.numberNorm);
			if (current) {
				current.sourceVersions.push(version);
			} else {
				map.set(ref.numberNorm, {
					numberNorm: ref.numberNorm,
					label: ref.label,
					number: ref.number || ref.numberNorm,
					sortKey: parseSortKey(ref.numberNorm),
					sourceVersions: [version],
				});
			}
		}
	}

	return [...map.values()];
}

export function rankVersions(chapter: MergedChapter, targetLang: string): SourceVersion[] {
	const target = targetLang.toLowerCase();
	const targetVersions = chapter.sourceVersions.filter((version) => version.lang === target);
	const pool = targetVersions.length > 0 ? targetVersions : chapter.sourceVersions;
	return [...pool].sort(byDateDesc);
}

export function pickBestVersion(chapter: MergedChapter, targetLang: string): SourceVersion | null {
	return rankVersions(chapter, targetLang)[0] ?? null;
}

export function versionKeyOf(version: SourceVersion): string {
	return `${version.source.manifest.id}:${version.ref.id}`;
}

export function sortMergedChapters(chapters: readonly MergedChapter[], newestFirst: boolean): MergedChapter[] {
	const direction = newestFirst ? 1 : -1;
	return [...chapters].sort((a, b) => (b.sortKey - a.sortKey) * direction);
}

export function sortMergedChaptersAsc(chapters: readonly MergedChapter[]): MergedChapter[] {
	return [...chapters].sort((a, b) => a.sortKey - b.sortKey);
}

function byDateDesc(a: SourceVersion, b: SourceVersion): number {
	return (b.ref.date ?? '').localeCompare(a.ref.date ?? '');
}

function parseSortKey(numberNorm: string): number {
	const value = Number.parseFloat(numberNorm);
	return Number.isFinite(value) ? value : 0;
}
