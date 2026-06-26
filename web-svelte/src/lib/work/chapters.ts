import type { WorkSource } from '$lib/db';
import { normalizeLang } from '$lib/lang';
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

export function mergeChapters(sourceChapters: SourceChapterDetail[]): MergedChapter[] {
	const map = new Map<string, MergedChapter>();

	for (const sourceChapter of sourceChapters) {
		// A single-language source's lone declared language is a safe fallback for a
		// chapter ref with no language; multi-language sources must stay '' (auto) so
		// translation auto-detects instead of guessing the wrong source language.
		const declared = sourceChapter.origin.languages ?? [];
		const sourceFallback = declared.length === 1 ? declared[0]! : null;
		for (const ref of sourceChapter.refs) {
			const lang = normalizeLang(ref.language) ?? normalizeLang(sourceFallback) ?? '';
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
	// Readable (non-locked) versions rank above premium-locked ones, then by recency.
	return [...pool].sort((a, b) => {
		const lockDiff = (a.ref.locked ? 1 : 0) - (b.ref.locked ? 1 : 0);
		return lockDiff !== 0 ? lockDiff : byDateDesc(a, b);
	});
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
