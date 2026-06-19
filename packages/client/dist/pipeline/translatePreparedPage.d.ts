/** Build translation inputs/results for one PreparedPage. */
import type { Translator } from '../translators/translator';
import type { TextRegion } from '../domain/regions';
import type { RecognizedTextPage, TextUnit } from '../domain/text';
import type { TranslationUnit, TranslatedUnit } from '../domain/translation';
import type { PreparedPageHandle } from '../domain/prepared';
export interface PreparedTextResult {
    readonly recognized: RecognizedTextPage;
    readonly textUnits: readonly TextUnit[];
    readonly translationUnits: readonly TranslationUnit[];
}
export interface PreparedTranslationResult extends PreparedTextResult {
    readonly translations: readonly TranslatedUnit[];
}
export declare function preparedTextFromRecognition(args: {
    readonly handle: PreparedPageHandle;
    readonly recognized: RecognizedTextPage;
}): PreparedTextResult;
export declare function textFromRecognition(args: {
    readonly pageIndex: number;
    readonly recognized: RecognizedTextPage;
    readonly regions?: readonly TextRegion[] | null;
}): PreparedTextResult;
export declare function translatePreparedText(args: {
    readonly text: PreparedTextResult;
    readonly translator: Translator;
    readonly sourceLanguage: string | null;
    readonly targetLanguage: string;
    readonly signal?: AbortSignal;
}): Promise<PreparedTranslationResult>;
