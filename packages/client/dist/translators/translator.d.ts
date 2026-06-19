import type { TranslationContext } from '../domain/context';
import type { TranslatedUnit, TranslationUnit } from '../domain/translation';
export interface TranslateUnitsArgs {
    units: readonly TranslationUnit[];
    sourceLang: string | null;
    targetLang: string;
    context?: TranslationContext;
    signal?: AbortSignal;
}
export interface Translator {
    readonly name: string;
    translateUnits(args: TranslateUnitsArgs): Promise<readonly TranslatedUnit[]>;
}
