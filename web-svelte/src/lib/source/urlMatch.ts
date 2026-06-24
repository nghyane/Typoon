// $lib/source/urlMatch.ts
// Shared helpers for detecting and resolving pasted manga URLs against installed
// sources. Used by AddMangaModal (and any other paste entry point) so the
// detection logic lives in one place.

import type { InstalledSource } from './types';

/** Tracking / share query params that should be stripped from a pasted URL
 *  before it becomes the stored upstream reference. */
const TRACKING_PARAMS = [
	/^utm_/i,
	/^ref$/i,
	/^ref_/i,
	/^fbclid$/i,
	/^gclid$/i,
	/^igshid$/i,
	/^spm$/i,
];

export interface SourceUrlMatch {
	/** The source whose host matches the URL. */
	source: InstalledSource;
	/** Cleaned URL to use as the upstream reference. */
	upstreamRef: string;
	/** True when a matching source exists but is currently disabled. */
	disabled: boolean;
}

/** Heuristic: does this input look like an http(s) URL the user pasted? */
export function isUrlLike(input: string): boolean {
	return /^https?:\/\//i.test(input.trim());
}

/** Remove tracking params and trailing fragments, keeping meaningful query. */
export function cleanUrl(raw: string): string {
	let parsed: URL;
	try {
		parsed = new URL(raw.trim());
	} catch {
		return raw.trim();
	}
	for (const key of [...parsed.searchParams.keys()]) {
		if (TRACKING_PARAMS.some((re) => re.test(key))) parsed.searchParams.delete(key);
	}
	parsed.hash = '';
	return parsed.href;
}

function hostMatches(urlHost: string, manifestHost: string): boolean {
	const host = urlHost.toLowerCase();
	const target = manifestHost.toLowerCase();
	return host === target || host.endsWith(`.${target}`);
}

/**
 * Resolve a pasted URL against the full source list (enabled + disabled).
 *
 * Searches `allSources` so callers can distinguish "no source manages this
 * site" (returns null) from "a source exists but is disabled" (returns a match
 * with `disabled: true`), enabling a "turn it on" prompt instead of a dead end.
 */
export function matchSourceUrl(raw: string, allSources: InstalledSource[]): SourceUrlMatch | null {
	let parsed: URL;
	try {
		parsed = new URL(raw.trim());
	} catch {
		return null;
	}
	const host = parsed.host;
	// Prefer exact host match, fall back to subdomain match.
	const source = allSources.find((item) => item.manifest.host.toLowerCase() === host.toLowerCase())
		?? allSources.find((item) => hostMatches(host, item.manifest.host));
	if (!source) return null;
	return { source, upstreamRef: cleanUrl(raw), disabled: !source.enabled };
}
