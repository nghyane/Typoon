import type { TranslatedUnit, TranslationUnit } from '../../domain/translation';
export declare function batchUnits(units: readonly TranslationUnit[], maxBatchChars?: number): TranslationUnit[][];
export declare function serializeBatch(units: readonly TranslationUnit[]): string;
export declare function parseTranslatedBatch(text: string, expectedCount: number): string[];
export declare function toTranslatedUnit(unit: TranslationUnit, targetText: string): TranslatedUnit;
