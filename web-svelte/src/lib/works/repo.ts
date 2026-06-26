import { nanoid } from 'nanoid';
import { db, type Work, type WorkSource } from '$lib/db';
import { deriveSourceKeys, sourceKey, type LibraryStatus } from '$lib/db';
import type { MangaDetail, MangaSummary, SourceManifest } from '$lib/source/types';
import { workIdFromSourceRef } from './id';

export async function listLibraryWorks(): Promise<Work[]> {
	return db().works
		.orderBy('updated_at')
		.reverse()
		.filter((work) => work.in_library && !work.deleted)
		.toArray();
}

export async function listRecentWorks(limit = 24): Promise<Work[]> {
	const works = await db().works.orderBy('last_opened_at').reverse().limit(limit).toArray();
	return works.filter((work) => !work.deleted);
}

export async function getWork(id: string): Promise<Work | null> {
	const work = await db().works.get(id);
	return work && !work.deleted ? work : null;
}

export async function listWorksBySourceRefs(
	source: string,
	upstreamRefs: readonly string[],
): Promise<Map<string, Work>> {
	const refs = [...new Set(upstreamRefs)];
	if (refs.length === 0) return new Map();

	const refSet = new Set(refs);
	const keys = refs.map((ref) => sourceKey(source, ref));
	const works = await db().works.where('sourceKey').anyOf(keys).toArray();
	const out = new Map<string, Work>();

	for (const work of works) {
		if (work.deleted) continue;
		for (const item of work.sources) {
			if (item.source === source && refSet.has(item.upstream_ref)) {
				out.set(item.upstream_ref, work);
			}
		}
	}

	return out;
}

export async function touchWork(id: string): Promise<void> {
	const now = new Date().toISOString();
	await db().works.update(id, { last_opened_at: now, updated_at: now });
}

export async function addWorkToLibrary(
	workId: string,
	status: LibraryStatus = 'reading',
): Promise<Work | null> {
	const current = await getWork(workId);
	if (!current) return null;
	const now = new Date().toISOString();
	const next: Work = {
		...current,
		in_library: true,
		library_status: current.library_status ?? status,
		library_added_at: current.library_added_at ?? now,
		updated_at: now,
	};
	await db().works.put(next);
	return next;
}

export async function removeWorkFromLibrary(workId: string): Promise<Work | null> {
	const current = await getWork(workId);
	if (!current) return null;
	const next: Work = {
		...current,
		in_library: false,
		library_status: null,
		updated_at: new Date().toISOString(),
	};
	await db().works.put(next);
	return next;
}

export async function setWorkLibraryStatus(workId: string, status: LibraryStatus): Promise<Work | null> {
	const current = await getWork(workId);
	if (!current) return null;
	const now = new Date().toISOString();
	const next: Work = {
		...current,
		in_library: true,
		library_status: status,
		library_added_at: current.library_added_at ?? now,
		updated_at: now,
	};
	await db().works.put(next);
	return next;
}

export async function createBlankWork(input: {
	title: string;
	target_lang?: string;
	nsfw?: boolean;
}): Promise<Work> {
	const now = new Date().toISOString();
	const work: Work = {
		id: nanoid(12),
		title: input.title,
		cover_url: null,
		target_lang: input.target_lang ?? 'vi',
		nsfw: !!input.nsfw,
		sources: [],
		sourceKey: [],
		in_library: false,
		library_status: null,
		library_added_at: null,
		last_opened_at: now,
		created_at: now,
		updated_at: now,
	};
	await db().works.put(work);
	return work;
}

function syncIdentityToPrimary(
	current: Work,
	sources: WorkSource[],
): Partial<Pick<Work, 'title' | 'cover_url'>> {
	const primary = sources[0];
	if (!primary) return {};

	const patch: Partial<Pick<Work, 'title' | 'cover_url'>> = {};
	if (!current.title_overridden) {
		const title = primary.title.trim();
		if (title && title !== current.title) patch.title = title;
	}
	if (!current.cover_overridden && primary.cover_url !== current.cover_url) {
		patch.cover_url = primary.cover_url;
	}
	return patch;
}

export async function attachSource(workId: string, source: WorkSource): Promise<Work> {
	const current = await getWork(workId);
	if (!current) throw new Error('Work không tồn tại.');

	const exists = current.sources.some((item) =>
		item.source === source.source && item.upstream_ref === source.upstream_ref,
	);
	if (exists) return current;

	const sources = [...current.sources, source];
	const next: Work = {
		...current,
		...syncIdentityToPrimary(current, sources),
		sources,
		sourceKey: deriveSourceKeys(sources),
		updated_at: new Date().toISOString(),
	};
	await db().works.put(next);
	return next;
}

export async function ensureWorkFromSource(manifest: SourceManifest, manga: MangaSummary | MangaDetail): Promise<Work> {
	const key = sourceKey(manifest.id, manga.url);
	const newId = workIdFromSourceRef(manifest.id, manga.url);

	const existing = (await db().works.where('sourceKey').equals(key).toArray())
		.find((work) => !work.deleted);
	const now = new Date().toISOString();

	if (existing) {
		if (existing.id !== newId) {
			await db().works.delete(existing.id);
			const migrated: Work = { ...existing, id: newId, last_opened_at: now, updated_at: now };
			await db().works.put(migrated);
			return migrated;
		}
		await db().works.update(existing.id, { last_opened_at: now, updated_at: now });
		return { ...existing, last_opened_at: now, updated_at: now };
	}

	const languages = [...(('availableLanguages' in manga && manga.availableLanguages) || manifest.languages || [])];
	const sources = [{
		source: manifest.id,
		upstream_ref: manga.url,
		title: manga.title,
		cover_url: manga.cover,
		languages,
		added_at: now,
	}];
	const work: Work = {
		id: newId,
		title: manga.title,
		cover_url: manga.cover,
		target_lang: 'vi',
		nsfw: !!manifest.nsfw,
		sources,
		sourceKey: deriveSourceKeys(sources),
		in_library: false,
		library_status: null,
		library_added_at: null,
		last_opened_at: now,
		created_at: now,
		updated_at: now,
	};
	await db().works.put(work);
	return work;
}
