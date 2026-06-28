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

/** Lowercase a host and drop the common `www.` / `m.` (mobile) prefix so the
 *  bare apex, the `www.` form and the mobile form all compare equal. */
function normalizeHost(host: string): string {
	return host.toLowerCase().replace(/^(?:www|m)\./u, '');
}

/** Pull the host out of an absolute URL, or null when it isn't one. */
function hostOf(url: string | undefined): string | null {
	if (!url) return null;
	try {
		return new URL(url).host;
	} catch {
		return null;
	}
}

/**
 * Every host a source can be recognised at, normalized and de-duplicated.
 *
 * Combines the manifest `host` (often the API/canonical origin) with the
 * user-facing `homepage` host, because for API-backed sources they differ and
 * users paste the homepage one: MangaDex `api.mangadex.org` ↔ `mangadex.org`,
 * OTruyen `otruyenapi.com` ↔ `otruyen.cc`.
 */
function sourceHosts(source: InstalledSource): string[] {
	const hosts = [source.manifest.host, hostOf(source.manifest.homepage)]
		.filter((host): host is string => !!host)
		.map(normalizeHost);
	return [...new Set(hosts)];
}

/** Does the pasted host belong to a source? Matches the apex and any subdomain
 *  in either direction so `mangadex.org` ↔ `api.mangadex.org` both resolve. */
function hostMatchesSource(pastedHost: string, source: InstalledSource): boolean {
	const host = normalizeHost(pastedHost);
	return sourceHosts(source).some(
		(candidate) =>
			host === candidate || host.endsWith(`.${candidate}`) || candidate.endsWith(`.${host}`),
	);
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
	// Prefer an exact host hit (precise wins over fuzzy), then fall back to the
	// normalized homepage/subdomain match that covers www./m. and API origins.
	const source = allSources.find((item) => item.manifest.host.toLowerCase() === host.toLowerCase())
		?? allSources.find((item) => hostMatchesSource(host, item));
	if (!source) return null;
	return { source, upstreamRef: cleanUrl(raw), disabled: !source.enabled };
}
