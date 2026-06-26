// $lib/collections.ts — small generic collection helpers.

/**
 * Return a new array with duplicates removed, keeping the first occurrence of
 * each key. Items whose key is empty/nullish are kept as-is (never collapsed).
 *
 * Used wherever a keyed list must stay unique — e.g. browse summaries paged in
 * from a source (a "latest" listing can re-surface the same series on a later
 * page) before a keyed `{#each}` renders them.
 */
export function dedupeBy<T>(items: readonly T[], key: (item: T) => string | null | undefined): T[] {
	const seen = new Set<string>();
	const out: T[] = [];
	for (const item of items) {
		const k = key(item);
		if (k) {
			if (seen.has(k)) continue;
			seen.add(k);
		}
		out.push(item);
	}
	return out;
}
