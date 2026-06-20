import type { SourceCacheOptions } from '$lib/sourceFetch.svelte';

export type SourceCacheScope = 'browse' | 'manga' | 'chapters' | 'chapter' | 'metadata' | 'session' | 'page';

export const SOURCE_CACHE_TTL: Record<SourceCacheScope, number> = {
	browse: 60 * 60,
	manga: 60 * 60,
	chapters: 60 * 60,
	chapter: 30 * 24 * 60 * 60,
	metadata: 30 * 24 * 60 * 60,
	session: 15 * 60,
	page: 5 * 60,
};

export function sourceCache(
	sourceId: string,
	scope: SourceCacheScope,
	parts: readonly unknown[],
	cookies: Record<string, string> = {},
	ttl = SOURCE_CACHE_TTL[scope],
): SourceCacheOptions {
	const cookie = cookieSignature(cookies);
	const suffix = cookie ? `:ck:${hashString(cookie)}` : '';
	return {
		policy: 'ttl',
		key: `src:${sourceId}:${scope}:v1:${hashString(stableStringify(parts))}${suffix}`,
		ttl,
	};
}

export function cookieSignature(cookies: Record<string, string>): string {
	return Object.entries(cookies)
		.sort(([a], [b]) => a.localeCompare(b))
		.map(([key, value]) => `${key}=${value}`)
		.join('; ');
}

export function hashString(value: string): string {
	let h = 2166136261;
	for (let i = 0; i < value.length; i += 1) {
		h ^= value.charCodeAt(i);
		h = Math.imul(h, 16777619);
	}
	return (h >>> 0).toString(36);
}

function stableStringify(value: unknown): string {
	if (value == null || typeof value !== 'object') return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
	const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
	return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`).join(',')}}`;
}
