import type { TranslationContext } from '../domain/context';
import type { TranslatedUnit, TranslationUnit } from '../domain/translation';
import type { Translator } from '../translators/translator';
export declare function translateSegment(args: {
    units: readonly TranslationUnit[];
    translator: Translator;
    sourceLang?: string | null;
    targetLang: string;
    context?: TranslationContext;
    signal?: AbortSignal;
}): Promise<readonly TranslatedUnit[]>;
