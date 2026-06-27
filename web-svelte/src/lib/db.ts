import Dexie, { type EntityTable } from 'dexie';

export interface WorkSource {
	source: string;
	upstream_ref: string;
	title: string;
	cover_url: string | null;
	languages: string[];
	added_at: string;
}

export type LibraryStatus = 'reading' | 'plan' | 'done' | 'dropped';

export interface Work {
	id: string;
	title: string;
	title_overridden?: boolean;
	cover_url: string | null;
	cover_overridden?: boolean;
	target_lang: string;
	nsfw: boolean;
	sources: WorkSource[];
	sourceKey: string[];
	in_library: boolean;
	library_status: LibraryStatus | null;
	library_added_at: string | null;
	last_opened_at: string;
	created_at: string;
	updated_at: string;
	deleted?: boolean;
}

// Per-work reading progress. Kept in its own table (not on Work) so the frequent
// "mark chapter read" writes never race with the full-object puts that update the
// library state of a Work.
export interface ReadProgress {
	work_id: string;
	last_chapter: string; // numberNorm of the most recently opened chapter
	last_read_at: string;
	read: string[]; // numberNorms the reader has opened
}

export class TypoonDb extends Dexie {
	works!: EntityTable<Work, 'id'>;
	progress!: EntityTable<ReadProgress, 'work_id'>;

	constructor() {
		super('typoon-v3-svelte');
		this.version(1).stores({
			works: '&id, in_library, last_opened_at, updated_at, *sourceKey',
		});
		this.version(2).stores({
			works: '&id, in_library, last_opened_at, updated_at, *sourceKey',
			progress: '&work_id',
		});
	}
}

let instance: TypoonDb | null = null;

export function db(): TypoonDb {
	if (!instance) {
		if (typeof indexedDB === 'undefined') {
			throw new Error('IndexedDB is not available (SSR or unsupported browser)');
		}
		instance = new TypoonDb();
	}
	return instance;
}

export function sourceKey(source: string, upstreamRef: string): string {
	return `${source}:${upstreamRef}`;
}

export function deriveSourceKeys(sources: WorkSource[]): string[] {
	return sources.map((source) => sourceKey(source.source, source.upstream_ref));
}
