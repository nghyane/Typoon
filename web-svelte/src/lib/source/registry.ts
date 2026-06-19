import type { InstalledSource, SourceManifest } from './types';

const bundled: Record<string, { default: SourceManifest }> = import.meta.glob(
	'../../../../packages/manga-sources/*.json',
	{ eager: true },
);

const storageKey = 'typoon.sources.v7.svelte';

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

function getEnabledSnapshot(): Record<string, boolean> {
	if (!enabledCache) enabledCache = readEnabled();
	return enabledCache;
}

function invalidateCache(): void {
	enabledCache = null;
}

export function listSources(): InstalledSource[] {
	const enabled = getEnabledSnapshot();
	return bundledManifests.map((manifest) => ({
		manifest,
		origin: 'bundled' as const,
		installedAt: 0,
		enabled: enabled[manifest.id] ?? true,
	}));
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

export function getSource(id: string): InstalledSource | null {
	return listEnabledSources().find((source) => source.manifest.id === id) ?? null;
}
