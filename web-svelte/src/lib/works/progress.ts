import { db, type ReadProgress } from '$lib/db';

export async function getProgress(workId: string): Promise<ReadProgress | null> {
	return (await db().progress.get(workId)) ?? null;
}

// Mark a chapter as opened: it becomes the resume point and joins the read set.
// Wrapped in a transaction so rapid chapter-to-chapter navigation can't drop
// entries through a read-modify-write race.
export async function recordRead(workId: string, chapterNorm: string): Promise<void> {
	if (!workId || !chapterNorm) return;
	const now = new Date().toISOString();
	await db().transaction('rw', db().progress, async () => {
		const current = await db().progress.get(workId);
		const read = current?.read ?? [];
		// Switching to a different chapter resets the saved scroll offset (it belongs
		// to the chapter named by last_chapter); reopening the same one keeps it.
		const sameChapter = current?.last_chapter === chapterNorm;
		await db().progress.put({
			work_id: workId,
			last_chapter: chapterNorm,
			last_read_at: now,
			read: read.includes(chapterNorm) ? read : [...read, chapterNorm],
			last_scroll_top: sameChapter ? current?.last_scroll_top ?? 0 : 0,
		});
	});
}

// Save the scroll offset within the current resume chapter so "đọc tiếp" returns
// to where the reader left off — even after a reload or a fresh navigation, not
// just browser back/forward. Only writes when the chapter is still the resume
// point, so it never clobbers another chapter's saved offset.
export async function recordScroll(workId: string, chapterNorm: string, scrollTop: number): Promise<void> {
	if (!workId || !chapterNorm) return;
	await db().transaction('rw', db().progress, async () => {
		const current = await db().progress.get(workId);
		if (!current || current.last_chapter !== chapterNorm) return;
		await db().progress.put({ ...current, last_scroll_top: Math.max(0, Math.round(scrollTop)) });
	});
}
