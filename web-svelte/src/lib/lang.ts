const LANG_LABEL: Record<string, string> = {
	en: 'English',
	vi: 'Tiếng Việt',
	ja: 'Tiếng Nhật',
	ko: 'Tiếng Hàn',
	zh: 'Tiếng Trung',
	'zh-hk': '中文 (HK)',
	'pt-br': 'Português (BR)',
	pt: 'Português',
	es: 'Español',
	'es-la': 'Español (LA)',
	fr: 'Français',
	de: 'Deutsch',
	it: 'Italiano',
	ru: 'Русский',
	id: 'Indonesia',
	th: 'ภาษาไทย',
	ar: 'العربية',
	tr: 'Türkçe',
	pl: 'Polski',
	nl: 'Nederlands',
	fa: 'فارسی',
	hi: 'हिन्दी',
	bn: 'বাংলা',
	he: 'עברית',
	hu: 'Magyar',
	ms: 'Bahasa Melayu',
	tl: 'Filipino',
	el: 'Ελληνικά',
	ro: 'Română',
	uk: 'Українська',
	cs: 'Čeština',
	sv: 'Svenska',
	kk: 'Қазақша',
	la: 'Latīna',
	ka: 'ქართული',
};

// English language *names* (as tagged by gallery sources like nhentai/hitomi/e-hentai)
// → ISO 639-1 codes. Keeps multi-language detection accurate instead of leaking raw
// names like "spanish" downstream.
const NAME_TO_CODE: Record<string, string> = {
	english: 'en',
	japanese: 'ja',
	chinese: 'zh',
	cantonese: 'zh-hk',
	korean: 'ko',
	vietnamese: 'vi',
	spanish: 'es',
	'spanish (latam)': 'es-la',
	french: 'fr',
	german: 'de',
	italian: 'it',
	russian: 'ru',
	portuguese: 'pt',
	'portuguese (br)': 'pt-br',
	brazilian: 'pt-br',
	indonesian: 'id',
	thai: 'th',
	arabic: 'ar',
	turkish: 'tr',
	polish: 'pl',
	dutch: 'nl',
	persian: 'fa',
	farsi: 'fa',
	hindi: 'hi',
	bengali: 'bn',
	hebrew: 'he',
	hungarian: 'hu',
	malay: 'ms',
	filipino: 'tl',
	tagalog: 'tl',
	greek: 'el',
	romanian: 'ro',
	ukrainian: 'uk',
	czech: 'cs',
	swedish: 'sv',
	kazakh: 'kk',
	latin: 'la',
	georgian: 'ka',
};

// Pseudo "languages" that gallery sources tag alongside the real one (a translated
// English doujin is tagged both "english" and "translated"). Never a chapter language.
export const PSEUDO_LANGS = new Set(['translated', 'rewrite', 'speechless', 'n/a', 'other']);

/**
 * Normalize a raw language value (an English name, an ISO code, or a "language:xx"
 * tag) into a canonical ISO 639-1 code. Returns `null` when the value is unknown,
 * a pseudo-language, or an "all/multi/auto" marker — callers must treat `null` as
 * "auto-detect", never hardcode a fallback language.
 */
export function normalizeLang(raw: string | null | undefined): string | null {
	if (!raw) return null;
	let s = raw.trim().toLowerCase();
	const colon = s.lastIndexOf(':'); // strip "language:english" style prefixes
	if (colon >= 0) s = s.slice(colon + 1).trim();
	if (!s || s === 'multi' || s === 'all' || s === 'auto') return null;
	if (PSEUDO_LANGS.has(s)) return null;
	if (s in NAME_TO_CODE) return NAME_TO_CODE[s]!;
	if (s in LANG_LABEL) return s; // already a known ISO code
	if (/^[a-z]{2}(-[a-z]{2,4})?$/.test(s)) return s; // generic ISO-ish passthrough
	return null; // unknown → auto
}

export function languageName(code: string): string {
	return LANG_LABEL[code] ?? LANG_LABEL[code.toLowerCase()] ?? code.toUpperCase();
}

export function languageCode(code: string): string {
	return code.toUpperCase();
}

export function languageSummary(codes: readonly string[]): string {
	if (codes.length === 0) return '';
	if (codes.length === 1) return languageName(codes[0]!);
	if (codes.length <= 3) return codes.map(languageCode).join(' · ');
	return `${codes.length} ngôn ngữ`;
}
