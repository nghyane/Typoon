import type { ChapterNumberNorm } from './types';

export type { ChapterNumberNorm };

export const DEFAULT_CHAPTER_NUMBER_NORM: ChapterNumberNorm = {
	input: 'number',
	patterns: [
		'Ch(?:apter|ương|\\.|)\\s*:?\\s*([0-9]+(?:\\.[0-9]+)?)',
		'第\\s*([0-9]+(?:\\.[0-9]+)?)\\s*[话話回]?',
		'([0-9]+(?:\\.[0-9]+)?)\\s*화',
		'([0-9]+(?:\\.[0-9]+)?)',
	],
	default: 'slug',
	postprocess: ['stripLeadingZeros', 'lowercase'],
};

interface CompiledNorm {
	input: 'number' | 'label';
	patterns: RegExp[];
	default: 'slug' | 'empty' | 'verbatim';
	postprocess: ('lowercase' | 'trim' | 'stripLeadingZeros')[];
}

const cache = new Map<ChapterNumberNorm, CompiledNorm>();

export function compileChapterNumberNorm(
	spec: ChapterNumberNorm | undefined | null,
): CompiledNorm {
	const effective = spec ?? DEFAULT_CHAPTER_NUMBER_NORM;
	const cached = cache.get(effective);
	if (cached) return cached;

	const compiled: CompiledNorm = {
		input: effective.input ?? 'number',
		default: effective.default ?? 'slug',
		patterns: (effective.patterns ?? []).map(compilePattern),
		postprocess: effective.postprocess ?? ['stripLeadingZeros', 'lowercase'],
	};
	cache.set(effective, compiled);
	return compiled;
}

function compilePattern(pattern: string): RegExp {
	try {
		return new RegExp(pattern, 'i');
	} catch (err) {
		throw new Error(
			`chapterNumberNorm pattern invalid: ${JSON.stringify(pattern)} (${(err as Error).message})`,
		);
	}
}

export function applyChapterNumberNorm(
	compiled: CompiledNorm,
	raw: { number: string; label: string | null },
): string {
	const rawStr = compiled.input === 'label' ? (raw.label ?? '') : raw.number;

	let extracted: string | null = null;
	for (const re of compiled.patterns) {
		const m = re.exec(rawStr);
		if (m) {
			extracted = (m[1] ?? m[0]) ?? null;
			if (extracted != null) break;
		}
	}

	let out: string;
	if (extracted != null) {
		out = extracted;
	} else {
		switch (compiled.default) {
			case 'slug': out = rawStr; break;
			case 'empty': out = ''; break;
			case 'verbatim': out = rawStr; break;
			default: out = rawStr;
		}
	}

	for (const step of compiled.postprocess) {
		switch (step) {
			case 'lowercase': out = out.toLowerCase(); break;
			case 'trim': out = out.trim(); break;
			case 'stripLeadingZeros': out = out.replace(/^0+(?=\d)/, ''); break;
		}
	}

	return out.trim();
}
