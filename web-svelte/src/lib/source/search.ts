// $lib/source/search.ts
// Pure, non-reactive search ranking helpers shared by manga search UIs.
// Kept framework-free so it can be unit-tested in isolation.

import type { InstalledSource, MangaSummary } from './types';

export interface SearchHit {
	source: InstalledSource;
	manga: MangaSummary;
	score: number;
}

/** Max hits kept per source group after ranking. */
export const PER_GROUP_MAX = 8;

function bigrams(value: string): string[] {
	const out: string[] = [];
	for (let i = 0; i < value.length - 1; i += 1) out.push(value.slice(i, i + 2));
	return out;
}

/**
 * Score how well `title` matches the user's `search` term.
 * Returns a value in [0, 1] — exact match scores 1, no overlap scores 0.
 * Combines token overlap with bigram (character-pair) similarity so partial
 * and reordered matches still rank sensibly.
 */
export function fuzzyScore(search: string, title: string): number {
	const q = search.trim().toLowerCase();
	const t = title.trim().toLowerCase();
	if (!q || !t) return 0;
	if (t === q) return 1;
	if (t.startsWith(q)) return 0.95;
	if (t.includes(q)) return 0.85;

	const qTokens = q.split(/\s+/).filter(Boolean);
	const tTokens = t.split(/\s+/).filter(Boolean);
	if (qTokens.length === 0) return 0;

	let matched = 0;
	for (const token of qTokens) {
		if (tTokens.some((item) => item.includes(token))) matched += 1;
	}
	const overlap = matched / qTokens.length;

	const qBigrams = new Set(bigrams(q));
	const tBigrams = bigrams(t);
	if (qBigrams.size === 0 || tBigrams.length === 0) return overlap * 0.7;
	let bigramHit = 0;
	for (const bigram of tBigrams) if (qBigrams.has(bigram)) bigramHit += 1;
	const bigramOverlap = bigramHit / Math.max(qBigrams.size, tBigrams.length);
	return Math.max(overlap * 0.7, bigramOverlap * 0.6);
}

/** Score every item against `search`, sort best-first, cap to PER_GROUP_MAX. */
export function rankAndCap(search: string, source: InstalledSource, items: MangaSummary[]): SearchHit[] {
	const scored = items.map((manga) => ({ source, manga, score: fuzzyScore(search, manga.title) }));
	scored.sort((a, b) => b.score - a.score);
	return scored.slice(0, PER_GROUP_MAX);
}

export function hitKey(hit: SearchHit): string {
	return `${hit.source.manifest.id}::${hit.manga.id}`;
}

export function errorFrom(value: unknown): Error {
	return value instanceof Error ? value : new Error(String(value));
}
