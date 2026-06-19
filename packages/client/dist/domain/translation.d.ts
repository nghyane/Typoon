import type { ImagePixels } from './image';
import type { TextRole, TextPlacement } from './planning';
import type { TextUnit } from './text';
export type TranslationKind = 'dialogue' | 'sfx' | 'skip';
export type RenderPhase = 'text' | 'layout';
export interface TranslationUnit {
    readonly id: string;
    readonly pageIndex: number;
    readonly blockIds: readonly string[];
    readonly sourceText: string;
    readonly kind: TranslationKind;
    readonly role: TextRole;
}
export interface TranslatedUnit {
    readonly unitId: string;
    readonly pageIndex: number;
    readonly kind: TranslationKind;
    readonly role: TextRole;
    readonly sourceText: string;
    readonly targetText: string;
}
export interface RenderedPage {
    readonly phase: RenderPhase;
    readonly image: ImagePixels;
    readonly pageIndex: number;
    readonly pageSize: readonly [number, number];
    readonly detectedLanguage: string | null;
    readonly textUnits: readonly TextUnit[];
    readonly translationUnits: readonly TranslationUnit[];
    readonly placements: readonly TextPlacement[];
    readonly translations: readonly TranslatedUnit[];
    readonly timingMs: Record<string, number>;
}
