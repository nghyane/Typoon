// $lib/library/addManga.svelte.ts
// Reactive state machine for the "Add manga to library" modal.
//
// All business logic lives here so AddMangaModal.svelte is pure markup. The
// component constructs one of these, drives it via `open`/`query`, and renders
// derived getters. Effects are owned by this class (started in an $effect.root
// or via the component's lifecycle) and are written to be self-cancelling.

import { untrack } from 'svelte';
import { goto } from '$app/navigation';
import { addWorkToLibrary, createBlankWork, ensureWorkFromSource } from '$lib/works/repo';
import { fetchBrowse, fetchMangaDetail } from '$lib/source/runtime/endpoints';
import { hasSearch } from '$lib/source/runtime/metadata';
import { listSources, setSourceEnabled } from '$lib/source/registry';
import { isUrlLike, matchSourceUrl, type SourceUrlMatch } from '$lib/source/urlMatch';
import { errorFrom, hitKey, rankAndCap, type SearchHit } from '$lib/source/search';
import type { InstalledSource, MangaDetail } from '$lib/source/types';

const SEARCH_DEBOUNCE_MS = 250;
const MIN_QUERY_LEN = 2;
const INITIAL_PREVIEW = 3;
const PER_GROUP_MAX = 8;
const BLANK_TITLE_FALLBACK = 'Manga mới';

export interface SearchFailure {
	sourceId: string;
	error: Error;
}

export interface ResultGroup {
	source: InstalledSource;
	hits: SearchHit[];
}

/** Discriminated UI state for the URL-paste lane, so the view renders one branch. */
export type UrlPhase =
	| { kind: 'idle' }
	| { kind: 'unsupported'; url: string }
	| { kind: 'disabled'; match: SourceUrlMatch }
	| { kind: 'loading'; match: SourceUrlMatch }
	| { kind: 'error'; match: SourceUrlMatch; message: string }
	| { kind: 'preview'; match: SourceUrlMatch; detail: MangaDetail };

export class AddMangaController {
	// ── Inputs (set by the component) ────────────────────────────
	#open = $state(false);
	// Read lazily so the latest prop callback is always used, not a stale one
	// captured at construction time.
	#getOnClose: () => () => void;

	// ── Core state ───────────────────────────────────────────────
	query = $state('');
	#debouncedQuery = $state('');
	#sources = $state<InstalledSource[]>([]);
	#hits = $state<SearchHit[]>([]);
	#failures = $state<SearchFailure[]>([]);
	#searching = $state(false);
	#pendingKey = $state<string | null>(null);
	#error = $state('');
	scopeId = $state<string | null>(null);
	#expandedBySource = $state<Record<string, boolean>>({});

	// ── URL paste lane ───────────────────────────────────────────
	#urlPreviewRef = $state('');
	#urlPreview = $state<MangaDetail | null>(null);
	#urlPreviewLoading = $state(false);
	#urlError = $state('');
	#urlReloadKey = $state(0);
	#urlFetchToken = $state('');

	constructor(getOnClose: () => () => void) {
		this.#getOnClose = getOnClose;
	}

	#close(): void {
		this.#getOnClose()();
	}

	// ── Derived view model ───────────────────────────────────────
	get sources() { return this.#sources; }
	get error() { return this.#error; }
	get failures() { return this.#failures; }
	get searching() { return this.#searching; }
	get pendingKey() { return this.#pendingKey; }
	get busy() { return this.#pendingKey !== null; }
	get trimmed() { return this.query.trim(); }
	get isUrl() { return isUrlLike(this.trimmed); }
	get debouncedQuery() { return this.#debouncedQuery; }

	get enabledSources() { return this.#sources.filter((s) => s.enabled); }
	get searchableSources() { return this.enabledSources.filter((s) => hasSearch(s.manifest)); }

	/** Match a pasted URL against ALL sources (enabled + disabled). */
	get urlMatch(): SourceUrlMatch | null {
		return this.isUrl ? matchSourceUrl(this.trimmed, this.#sources) : null;
	}

	get hits() { return this.#hits; }
	get scopedHits() {
		return this.scopeId === null ? this.#hits : this.#hits.filter((h) => h.source.manifest.id === this.scopeId);
	}
	get visibleSources() {
		return this.scopeId === null
			? this.searchableSources
			: this.searchableSources.filter((s) => s.manifest.id === this.scopeId);
	}
	get hitCounts() {
		const counts = new Map<string, number>();
		for (const hit of this.#hits) {
			const id = hit.source.manifest.id;
			counts.set(id, (counts.get(id) ?? 0) + 1);
		}
		return counts;
	}
	get sourcesWithHits() {
		return this.searchableSources.filter((s) => this.hitCounts.has(s.manifest.id));
	}
	get singleSourceResults() { return this.visibleSources.length === 1; }

	get resultGroups(): ResultGroup[] {
		const by = new Map<string, ResultGroup>();
		for (const hit of this.scopedHits) {
			const id = hit.source.manifest.id;
			if (!by.has(id)) by.set(id, { source: hit.source, hits: [] });
			by.get(id)!.hits.push(hit);
		}
		for (const group of by.values()) group.hits.sort((a, b) => b.score - a.score);
		return this.visibleSources
			.map((s) => by.get(s.manifest.id))
			.filter((g): g is ResultGroup => !!g);
	}

	/** Hits a source group shows before "see more" is clicked. */
	visibleHitsFor(group: ResultGroup): SearchHit[] {
		const capped = group.hits.slice(0, PER_GROUP_MAX);
		const expanded = this.singleSourceResults || this.#expandedBySource[group.source.manifest.id];
		return expanded ? capped : capped.slice(0, INITIAL_PREVIEW);
	}

	moreCountFor(group: ResultGroup): number {
		return group.hits.slice(0, PER_GROUP_MAX).length - this.visibleHitsFor(group).length;
	}

	/** Single discriminated phase for the URL lane — keeps the view declarative. */
	get urlPhase(): UrlPhase {
		const match = this.urlMatch;
		if (!this.isUrl) return { kind: 'idle' };
		if (!match) return { kind: 'unsupported', url: this.trimmed };
		if (match.disabled) return { kind: 'disabled', match };
		if (this.#urlError) return { kind: 'error', match, message: this.#urlError };
		if (this.#urlPreview && this.#urlPreviewRef === match.upstreamRef) {
			return { kind: 'preview', match, detail: this.#urlPreview };
		}
		return { kind: 'loading', match };
	}

	// ── Effects: call once from the component (inside $effect via .start()) ──

	/** Wire up all reactive effects. Returns a disposer. Must run in component
	 *  init so `$effect`/`$effect.root` has an owner. */
	start(): () => void {
		const root = $effect.root(() => {
			this.#resetOnOpen();
			this.#debounceQuery();
			this.#runSearch();
			this.#runUrlPreview();
		});
		return root;
	}

	setOpen(value: boolean): void {
		this.#open = value;
	}

	#resetOnOpen(): void {
		$effect(() => {
			if (!this.#open) return;
			untrack(() => this.#reset());
		});
	}

	#reset(): void {
		this.#sources = listSources();
		this.query = '';
		this.#debouncedQuery = '';
		this.#hits = [];
		this.#failures = [];
		this.#searching = false;
		this.#pendingKey = null;
		this.#error = '';
		this.scopeId = null;
		this.#expandedBySource = {};
		this.#urlPreviewRef = '';
		this.#urlPreview = null;
		this.#urlPreviewLoading = false;
		this.#urlError = '';
		this.#urlReloadKey = 0;
		this.#urlFetchToken = '';
	}

	#debounceQuery(): void {
		$effect(() => {
			if (!this.#open) return;
			const value = this.query;
			const timer = window.setTimeout(() => { this.#debouncedQuery = value.trim(); }, SEARCH_DEBOUNCE_MS);
			return () => window.clearTimeout(timer);
		});
	}

	#runSearch(): void {
		$effect(() => {
			const q = this.#debouncedQuery.trim();
			if (!this.#open || isUrlLike(q) || q.length < MIN_QUERY_LEN) {
				this.#hits = [];
				this.#failures = [];
				this.#searching = false;
				return;
			}

			let cancelled = false;
			this.#searching = true;
			this.#error = '';
			const targets = this.searchableSources;

			Promise.allSettled(
				targets.map(async (source) => ({
					source,
					items: await fetchBrowse(source.manifest, { search: true as const }, { page: 1, q }),
				})),
			).then((results) => {
				if (cancelled) return;
				const nextHits: SearchHit[] = [];
				const nextFailures: SearchFailure[] = [];
				for (let i = 0; i < results.length; i++) {
					const result = results[i]!;
					if (result.status === 'fulfilled') {
						nextHits.push(...rankAndCap(q, result.value.source, result.value.items));
					} else {
						nextFailures.push({ sourceId: targets[i]!.manifest.id, error: errorFrom(result.reason) });
					}
				}
				this.#hits = nextHits;
				this.#failures = nextFailures;
			}).catch((err: Error) => {
				if (!cancelled) this.#error = err.message;
			}).finally(() => {
				if (!cancelled) this.#searching = false;
			});

			return () => { cancelled = true; };
		});
	}

	#runUrlPreview(): void {
		$effect(() => {
			const match = this.urlMatch;
			const reload = this.#urlReloadKey;
			if (!this.#open || !match || match.disabled) {
				this.#urlPreviewRef = '';
				this.#urlPreview = null;
				this.#urlPreviewLoading = false;
				this.#urlError = '';
				return;
			}

			const ref = match.upstreamRef;
			const token = `${ref}::${reload}`;
			// untrack the dedup read so writing #urlFetchToken doesn't re-trigger
			// this effect and cancel its own in-flight fetch (the original bug).
			if (untrack(() => this.#urlFetchToken) === token) return;
			this.#urlFetchToken = token;

			let cancelled = false;
			this.#urlPreviewRef = ref;
			this.#urlPreview = null;
			this.#urlError = '';
			this.#urlPreviewLoading = true;

			fetchMangaDetail(match.source.manifest, ref)
				.then((detail) => { if (!cancelled) this.#urlPreview = detail; })
				.catch((err) => { if (!cancelled) this.#urlError = errorFrom(err).message; })
				.finally(() => { if (!cancelled) this.#urlPreviewLoading = false; });

			return () => { cancelled = true; };
		});
	}

	// ── Commands (user actions) ──────────────────────────────────

	setQuery(value: string): void {
		this.query = value;
		this.scopeId = null;
		this.#expandedBySource = {};
		this.#error = '';
		this.#urlError = '';
	}

	setScope(id: string | null): void {
		this.scopeId = id;
	}

	expandSource(id: string): void {
		this.#expandedBySource = { ...this.#expandedBySource, [id]: true };
	}

	#refreshSources(): void {
		this.#sources = listSources();
	}

	toggleSource(source: InstalledSource): void {
		if (!hasSearch(source.manifest)) return;
		setSourceEnabled(source.manifest.id, !source.enabled);
		this.#refreshSources();
	}

	/** Enable a matched-but-disabled source so its preview can load. */
	enableMatchedSource(): void {
		const match = this.urlMatch;
		if (!match) return;
		setSourceEnabled(match.source.manifest.id, true);
		this.#refreshSources();
	}

	retryUrlPreview(): void {
		this.#urlError = '';
		this.#urlReloadKey += 1;
	}

	isImportingUrl(): boolean {
		return this.#pendingKey?.startsWith('url::') ?? false;
	}

	async importPreview(): Promise<void> {
		const match = this.urlMatch;
		const detail = this.#urlPreview;
		if (!match || !detail || this.#pendingKey) return;
		const key = `url::${match.upstreamRef}`;
		await this.#runImport(key, () =>
			ensureWorkFromSource(match.source.manifest, { ...detail, id: match.upstreamRef, url: match.upstreamRef }),
		);
	}

	async importHit(hit: SearchHit): Promise<void> {
		if (this.#pendingKey) return;
		const key = hitKey(hit);
		await this.#runImport(key, async () => {
			const resolved = await fetchMangaDetail(hit.source.manifest, hit.manga.url).catch(() => null);
			return ensureWorkFromSource(hit.source.manifest, resolved ?? hit.manga);
		});
	}

	async importBlank(title: string): Promise<void> {
		if (this.#pendingKey) return;
		this.#pendingKey = 'blank';
		this.#error = '';
		try {
			const work = await createBlankWork({ title: title.trim() || BLANK_TITLE_FALLBACK });
			await addWorkToLibrary(work.id);
			this.#close();
			await goto(`/w/${work.id}`);
		} catch (err) {
			this.#error = errorFrom(err).message;
		} finally {
			if (this.#pendingKey === 'blank') this.#pendingKey = null;
		}
	}

	/** Shared import pipeline: dedupe by key, ensure work, add to library, close. */
	async #runImport(key: string, ensure: () => Promise<{ id: string; in_library: boolean }>): Promise<void> {
		this.#pendingKey = key;
		this.#error = '';
		try {
			const work = await ensure();
			if (!work.in_library) await addWorkToLibrary(work.id);
			this.#close();
		} catch (err) {
			this.#error = errorFrom(err).message;
		} finally {
			if (this.#pendingKey === key) this.#pendingKey = null;
		}
	}
}
