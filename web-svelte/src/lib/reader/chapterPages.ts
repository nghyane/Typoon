// Cached chapter page-list fetches. The page-list (resolving a chapter URL into
// its image URLs/tokens through the source gateway) is the dominant wait when
// entering the reader — ~0.5–1s of gateway latency before the first image can
// even start. Routing it through the query cache lets the work-detail page
// PREFETCH it on chapter hover/tap, so by the time the reader mounts the list is
// already resolved and the first page loads immediately.

import { queryClient } from '$lib/queryClient';
import { fetchChapterPages } from '$lib/source/runtime/endpoints';
import type { SourceManifest } from '$lib/source/types';

const STALE_MS = 10 * 60_000;

function key(sourceId: string, chapterUrl: string) {
	return ['chapter-pages', sourceId, chapterUrl] as const;
}

/** Resolve a chapter's page list, served from cache when warm (e.g. prefetched). */
export function fetchChapterPagesCached(manifest: SourceManifest, chapterUrl: string) {
	return queryClient.ensureQueryData({
		queryKey: key(manifest.id, chapterUrl),
		queryFn: () => fetchChapterPages(manifest, chapterUrl),
		staleTime: STALE_MS,
	});
}

/** Warm the page list ahead of navigation (hover/tap on a chapter row). Idempotent:
 *  a fresh/in-flight entry is a no-op, so spamming hovers costs nothing. */
export function prefetchChapterPages(manifest: SourceManifest, chapterUrl: string): void {
	void queryClient.prefetchQuery({
		queryKey: key(manifest.id, chapterUrl),
		queryFn: () => fetchChapterPages(manifest, chapterUrl),
		staleTime: STALE_MS,
	});
}
