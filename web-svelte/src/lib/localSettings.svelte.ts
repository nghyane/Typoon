import { browser } from '$app/environment';

export type ReaderMode = 'webtoon';
export type ThemeMode = 'system' | 'light' | 'dark';
export type SourcePref = { kind: 'auto' } | { kind: 'raw'; versionKey: string };

export const READER_PAGE_WIDTH_MIN = 600;
export const READER_PAGE_WIDTH_MAX = 1600;

export interface LocalSettings {
	theme: ThemeMode;
	reader_mode: ReaderMode;
	reader_page_width: number;
	reader_source_prefs: Record<string, SourcePref>;
	default_target_lang: string;
	updated_at: string;
}

const storageKey = 'typoon.localSettings.v1.svelte';

const defaults: LocalSettings = {
	theme: 'system',
	reader_mode: 'webtoon',
	reader_page_width: 1040,
	reader_source_prefs: {},
	default_target_lang: 'vi',
	updated_at: new Date(0).toISOString(),
};

class LocalSettingsStore {
	state = $state<LocalSettings>(defaults);
	#loaded = false;

	load(): void {
		if (this.#loaded || !browser) return;
		this.#loaded = true;
		try {
			const raw = localStorage.getItem(storageKey);
			if (!raw) return;
			this.state = normalize(JSON.parse(raw) as Partial<LocalSettings>);
		} catch {
			this.state = defaults;
		}
	}

	update(patch: Partial<Omit<LocalSettings, 'updated_at'>>): void {
		this.load();
		this.state = normalize({ ...this.state, ...patch, updated_at: new Date().toISOString() });
		if (browser) localStorage.setItem(storageKey, JSON.stringify(this.state));
	}
}

function normalize(value: Partial<LocalSettings>): LocalSettings {
	const rawReaderMode = (value as { reader_mode?: unknown }).reader_mode;
	const readerMode = rawReaderMode === 'pager' || rawReaderMode === 'strip' ? 'webtoon' : rawReaderMode;
	return {
		...defaults,
		...value,
		reader_mode: isReaderMode(readerMode) ? readerMode : defaults.reader_mode,
		reader_page_width: clamp(value.reader_page_width, READER_PAGE_WIDTH_MIN, READER_PAGE_WIDTH_MAX, defaults.reader_page_width),
		reader_source_prefs: normalizeSourcePrefs(value.reader_source_prefs),
		theme: isThemeMode(value.theme) ? value.theme : defaults.theme,
		default_target_lang: value.default_target_lang ?? defaults.default_target_lang,
		updated_at: value.updated_at ?? defaults.updated_at,
	};
}

function isReaderMode(value: unknown): value is ReaderMode {
	return value === 'webtoon';
}

function isThemeMode(value: unknown): value is ThemeMode {
	return value === 'system' || value === 'light' || value === 'dark';
}

function normalizeSourcePrefs(value: unknown): Record<string, SourcePref> {
	if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
	const out: Record<string, SourcePref> = {};
	for (const [key, pref] of Object.entries(value as Record<string, unknown>)) {
		if (!pref || typeof pref !== 'object') continue;
		const candidate = pref as Partial<SourcePref>;
		if (candidate.kind === 'auto') out[key] = { kind: 'auto' };
		else if (candidate.kind === 'raw' && typeof candidate.versionKey === 'string') out[key] = { kind: 'raw', versionKey: candidate.versionKey };
	}
	return out;
}

function clamp(value: unknown, min: number, max: number, fallback: number): number {
	const n = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(n)) return fallback;
	return Math.min(max, Math.max(min, Math.round(n)));
}

export const localSettings = new LocalSettingsStore();
