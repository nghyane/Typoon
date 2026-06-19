/**
 * Generate a deterministic, URL-safe work ID from (source, upstream_ref).
 * Same input always produces the same output — enables DB lookup + migration.
 */
export function workIdFromSourceRef(source: string, upstreamRef: string): string {
	const key = `${source}:${upstreamRef}`;
	let h = 0;
	for (let i = 0; i < key.length; i += 1) {
		h = (h << 5) - h + key.charCodeAt(i);
		h |= 0;
	}
	return Math.abs(h).toString(36).padStart(8, '0');
}
