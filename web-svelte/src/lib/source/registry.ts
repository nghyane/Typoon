import type { InstalledSource, SourceManifest } from './types';

const bundled: Record<string, { default: SourceManifest }> = import.meta.glob(
	'../../../../packages/manga-sources/*.json',
	{ eager: true },
);

const storageKey = 'typoon.sources.v7.svelte';
const defaultEnabledSourceIds = new Set(['baozimh', 'naver-webtoon', 'truyenqq']);

function isManifest(value: { default: unknown }): value is { default: SourceManifest } {
	const manifest = value.default as Partial<SourceManifest> | undefined;
	return !!manifest?.id && !!manifest?.name;
}

export const bundledManifests: SourceManifest[] = Object.values(bundled)
	.filter(isManifest)
	.map((entry) => entry.default);

function readEnabled(): Record<string, boolean> {
	if (typeof localStorage === 'undefined') return {};
	try {
		return JSON.parse(localStorage.getItem(storageKey) ?? '{}') as Record<string, boolean>;
	} catch {
		return {};
	}
}

function writeEnabled(enabled: Record<string, boolean>): void {
	if (typeof localStorage === 'undefined') return;
	localStorage.setItem(storageKey, JSON.stringify(enabled));
}

let enabledCache: Record<string, boolean> | null = null;
// The assembled InstalledSource[] is derived purely from the (large) bundled
// manifests + the enabled snapshot, so it's stable until toggled. Cache it:
// listSources()/getSource() are called per WorkCard, and rebuilding 20 manifest
// objects on each call showed up as avoidable churn in the browse grids.
let sourcesCache: InstalledSource[] | null = null;
let sourceById: Map<string, InstalledSource> | null = null;

function getEnabledSnapshot(): Record<string, boolean> {
	if (!enabledCache) enabledCache = readEnabled();
	return enabledCache;
}

function invalidateCache(): void {
	enabledCache = null;
	sourcesCache = null;
	sourceById = null;
}

export function listSources(): InstalledSource[] {
	if (sourcesCache) return sourcesCache;
	const enabled = getEnabledSnapshot();
	sourcesCache = bundledManifests.map((manifest) => ({
		manifest,
		origin: 'bundled' as const,
		installedAt: 0,
		enabled: enabled[manifest.id] ?? defaultEnabledSourceIds.has(manifest.id),
	}));
	return sourcesCache;
}

export function listEnabledSources(): InstalledSource[] {
	return listSources().filter((source) => source.enabled);
}

export function setSourceEnabled(id: string, value: boolean): void {
	const enabled = readEnabled();
	enabled[id] = value;
	writeEnabled(enabled);
	invalidateCache();
}

export function enableDefaultSources(): void {
	const enabled = readEnabled();
	for (const manifest of bundledManifests) enabled[manifest.id] = defaultEnabledSourceIds.has(manifest.id);
	writeEnabled(enabled);
	invalidateCache();
}

export function getSource(id: string): InstalledSource | null {
	if (!sourceById) sourceById = new Map(listSources().map((source) => [source.manifest.id, source]));
	const source = sourceById.get(id);
	return source?.enabled ? source : null;
}
