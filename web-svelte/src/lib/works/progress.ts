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
		await db().progress.put({
			work_id: workId,
			last_chapter: chapterNorm,
			last_read_at: now,
			read: read.includes(chapterNorm) ? read : [...read, chapterNorm],
		});
	});
}
