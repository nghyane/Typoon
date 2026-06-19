import type { TranslatedUnit } from '../../domain/translation';
import type { Translator } from '../translator';
export interface GoogleTranslateWebOptions {
    readonly endpoint?: string;
    readonly maxBatchChars?: number;
    readonly maxConcurrency?: number;
}
export declare class GoogleTranslateWeb implements Translator {
    readonly name = "google-translate-web";
    private readonly endpoint;
    private readonly maxBatchChars;
    private readonly limiter;
    constructor(options?: GoogleTranslateWebOptions);
    translateUnits({ units, sourceLang, targetLang, signal }: Parameters<Translator['translateUnits']>[0]): Promise<readonly TranslatedUnit[]>;
}
