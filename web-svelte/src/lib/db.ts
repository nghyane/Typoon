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

export class TypoonDb extends Dexie {
	works!: EntityTable<Work, 'id'>;

	constructor() {
		super('typoon-v3-svelte');
		this.version(1).stores({
			works: '&id, in_library, last_opened_at, updated_at, *sourceKey',
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
