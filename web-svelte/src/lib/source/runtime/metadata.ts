// Manifest metadata — sync inspection of source capability.
// Called by Explore UI before issuing browse/detail fetches.
import { getAdapter } from '../adapters';
import type { SourceManifest, Shelf, Filter } from '../types';

export interface ShelfDescriptor {
	id: string;
	label: string;
	hint?: string;
	paginated: boolean;
}

function isInternal(m: SourceManifest): boolean {
	return m.kind === 'internal';
}

function shelfPaginated(shelf: Shelf): boolean {
	const p = shelf.endpoint.pagination;
	return p != null && p.type !== 'cursor';
}

export function getShelves(manifest: SourceManifest): ShelfDescriptor[] {
	if (isInternal(manifest)) return [];
	if (manifest.adapter) {
		const adapter = getAdapter(manifest.adapter);
		if (adapter?.fetchBrowse) {
			const declared = manifest.endpoints?.shelves ?? [];
			return declared.length > 0
				? declared.map((s) => ({ id: s.id, label: s.label, hint: s.hint, paginated: shelfPaginated(s) }))
				: [{ id: 'latest', label: 'Mới nhất', paginated: true }];
		}
	}
	return (manifest.endpoints?.shelves ?? []).map((s) => ({
		id: s.id, label: s.label, hint: s.hint, paginated: shelfPaginated(s),
	}));
}

export function hasSearch(manifest: SourceManifest): boolean {
	if (isInternal(manifest)) return false;
	if (manifest.adapter) {
		const adapter = getAdapter(manifest.adapter);
		if (adapter?.fetchBrowse) return true;
	}
	return !!manifest.endpoints?.search;
}

export function shelfPageSize(manifest: SourceManifest, shelfId: string): number {
	if (isInternal(manifest)) return 24;
	if (manifest.adapter) {
		const adapter = getAdapter(manifest.adapter);
		if (adapter?.fetchBrowse) {
			const shelf = manifest.endpoints?.shelves.find((s) => s.id === shelfId);
			const p = shelf?.endpoint.pagination;
			return (p && p.type !== 'cursor') ? p.pageSize : 25;
		}
	}
	const shelf = manifest.endpoints?.shelves.find((s) => s.id === shelfId);
	return shelf?.endpoint.pagination?.pageSize ?? Infinity;
}

export function searchPageSize(manifest: SourceManifest): number {
	if (manifest.adapter) {
		const adapter = getAdapter(manifest.adapter);
		if (adapter?.fetchBrowse) return 25;
	}
	return manifest.endpoints?.search?.pagination?.pageSize ?? Infinity;
}

/**
 * Active filters for the current browse target. Shelf-level filters (when the
 * shelf declares them) take precedence over source-level `manifest.filters`;
 * `search` and shelves without their own filters fall back to source-level.
 */
export function getFilters(manifest: SourceManifest, shelfId?: string): Filter[] {
	if (shelfId) {
		const shelf = manifest.endpoints?.shelves.find((s) => s.id === shelfId);
		if (shelf?.filters) return shelf.filters;
	}
	return manifest.filters ?? [];
}

export function getDefaultFilterState(manifest: SourceManifest): Record<string, string | string[]> {
	return { ...(manifest.defaults ?? {}) };
}

export interface AssembledFilters {
	/** `&a&b` query fragments for `inject: 'param'` filters (`{filterParams}`). */
	params: string;
	/** single path segment for an `inject: 'path'` filter (`{filterPath}`). */
	path: string;
	/** extra `q` terms for `inject: 'query'` filters, space-joined. */
	query: string;
}

/** Splits the selected options of `filters` into per-injection-mode request pieces. */
export function assembleFilters(
	filters: Filter[],
	state: Record<string, string | string[]>,
): AssembledFilters {
	const params: string[] = [];
	const queryTerms: string[] = [];
	let path = '';
	for (const filter of filters) {
		const selected = state[filter.id];
		if (selected == null) continue;
		const ids = Array.isArray(selected) ? selected : [selected];
		for (const id of ids) {
			const option = filter.options.find((item) => item.id === id);
			if (!option?.param) continue;
			if (filter.inject === 'path') path = option.param;
			else if (filter.inject === 'query') queryTerms.push(option.param);
			else params.push(option.param);
		}
	}
	return {
		params: params.length > 0 ? '&' + params.join('&') : '',
		path,
		query: queryTerms.join(' '),
	};
}

export function assembleFilterState(
	filters: Filter[],
	state: Record<string, string | string[]>,
): Record<string, string | string[]> {
	const out: Record<string, string | string[]> = {};
	for (const filter of filters) {
		const selected = state[filter.id];
		if (selected == null) continue;
		const ids = Array.isArray(selected) ? selected : [selected];
		const values = ids
			.map((id) => {
				const option = filter.options.find((item) => item.id === id);
				return option ? (option.value ?? option.id) : null;
			})
			.filter((v): v is string => v !== null);
		if (values.length > 0) {
			out[filter.id] = filter.type === 'select' ? values[0]! : values;
		}
	}
	return out;
}
