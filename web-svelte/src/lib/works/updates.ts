// Chapter-update detection. No third-party source pushes us — we fingerprint the
// source's current chapter list (newest chapter + count) and compare it to what
// was stored. To keep it cheap (each check is a gateway round-trip), checks are
// TTL-gated and run through a small concurrency pool, off the render path.

import { db, type Work, type WorkUpdate } from '$lib/db';
import { getSource } from '$lib/source/registry';
import { fetchMangaDetail } from '$lib/source/runtime/endpoints';
import { queryClient } from '$lib/queryClient';
import { mergeChapters, sortMergedChapters, type MergedChapter, type SourceChapterDetail } from '$lib/work/chapters';
import { listLibraryWorks } from '$lib/works/repo';

const DEFAULT_TTL_MS = 6 * 60 * 60 * 1000; // don't re-poll a source more than ~4×/day
const MANGA_DETAIL_STALE_MS = 5 * 60 * 1000; // reuse a recent detail-page fetch
const MAX_CONCURRENT = 4; // be polite to the rate-limited gateway

function nowIso(): string {
	return new Date().toISOString();
}

export async function getWorkUpdate(workId: string): Promise<WorkUpdate | null> {
	return (await db().workUpdates.get(workId)) ?? null;
}

export async function getWorkUpdatesMap(workIds: string[]): Promise<Record<string, WorkUpdate>> {
	if (!workIds.length) return {};
	const rows = await db().workUpdates.bulkGet(workIds);
	const map: Record<string, WorkUpdate> = {};
	for (const row of rows) if (row) map[row.work_id] = row;
	return map;
}

async function fetchMergedChapters(work: Work, fresh: boolean): Promise<MergedChapter[]> {
	const results = await Promise.allSettled(
		work.sources.map(async (origin): Promise<SourceChapterDetail> => {
			const source = getSource(origin.source);
			if (!source) throw new Error(`Nguồn ${origin.source} không khả dụng.`);
			const detail = await queryClient.fetchQuery({
				queryKey: ['manga-detail', origin.source, origin.upstream_ref] as const,
				queryFn: () => fetchMangaDetail(source.manifest, origin.upstream_ref),
				// A forced check must hit the network even if a recent detail-page
				// fetch is still cached; a background (TTL) check can reuse it.
				staleTime: fresh ? 0 : MANGA_DETAIL_STALE_MS,
			});
			return { source, origin, refs: detail.chapters };
		}),
	);
	const ok = results.filter((r): r is PromiseFulfilledResult<SourceChapterDetail> => r.status === 'fulfilled');
	return mergeChapters(ok.map((r) => r.value));
}

function newestChapterDate(chapter: MergedChapter): string | null {
	const dates = chapter.sourceVersions.map((v) => v.ref.date).filter((d): d is string => !!d);
	if (!dates.length) return null;
	return [...dates].sort().at(-1) ?? null;
}

/** Fetch the source's current chapter list, compare to the stored fingerprint,
 *  and persist. Returns whether a NEW chapter appeared (false on first seed and
 *  on no-change). Returns null on a fetch failure so we never clobber good data. */
export async function refreshWorkUpdate(work: Work, opts: { fresh?: boolean } = {}): Promise<{ record: WorkUpdate; hasNew: boolean } | null> {
	const merged = await fetchMergedChapters(work, !!opts.fresh);
	if (!merged.length) return null;
	const newest = sortMergedChapters(merged, true)[0];
	if (!newest) return null;

	const prev = await getWorkUpdate(work.id);
	const latestNorm = newest.numberNorm;
	const count = merged.length;
	const changed = !prev || prev.latest_norm !== latestNorm || prev.chapter_count !== count;
	const hasNew = !!prev && changed; // first seed establishes a baseline, not "new"

	const record: WorkUpdate = {
		work_id: work.id,
		latest_norm: latestNorm,
		latest_label: newest.number || latestNorm,
		chapter_count: count,
		updated_at: changed ? (newestChapterDate(newest) ?? nowIso()) : prev!.updated_at,
		checked_at: nowIso(),
	};
	await db().workUpdates.put(record);
	return { record, hasNew };
}

export interface UpdateCheckResult {
	checked: number;
	withNew: number;
}

export async function checkLibraryUpdates(opts: { force?: boolean; ttlMs?: number; concurrency?: number } = {}): Promise<UpdateCheckResult> {
	return checkWorksUpdates(await listLibraryWorks(), opts);
}

export async function checkWorksUpdates(
	works: Work[],
	opts: { force?: boolean; ttlMs?: number; concurrency?: number } = {},
): Promise<UpdateCheckResult> {
	const ttl = opts.ttlMs ?? DEFAULT_TTL_MS;
	const now = Date.now();
	const map = await getWorkUpdatesMap(works.map((w) => w.id));
	const due = works.filter((w) => opts.force || !map[w.id] || now - Date.parse(map[w.id]!.checked_at) > ttl);

	let withNew = 0;
	await pool(due, opts.concurrency ?? MAX_CONCURRENT, async (work) => {
		const res = await refreshWorkUpdate(work, { fresh: opts.force }).catch(() => null);
		if (res?.hasNew) withNew += 1;
	});
	return { checked: due.length, withNew };
}

async function pool<T>(items: readonly T[], limit: number, fn: (item: T) => Promise<void>): Promise<void> {
	let cursor = 0;
	const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
		while (cursor < items.length) {
			const item = items[cursor++]!;
			await fn(item);
		}
	});
	await Promise.all(workers);
}
