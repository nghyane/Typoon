import type { CanvasPage } from '../domain/canvas';
import type { TextPlacement } from '../domain/planning';
import type { RecognizedTextPage } from '../domain/text';
import type { RenderPhase, RenderedPage, TranslatedUnit, TranslationUnit } from '../domain/translation';
import type { TextUnit } from '../domain/text';
export declare function materializeRenderedPage(args: {
    readonly phase: RenderPhase;
    readonly canvas: CanvasPage;
    readonly recognizedText: RecognizedTextPage;
    readonly textUnits: readonly TextUnit[];
    readonly translationUnits: readonly TranslationUnit[];
    readonly placements: readonly TextPlacement[];
    readonly translations: readonly TranslatedUnit[];
    readonly timingMs?: Record<string, number>;
}): RenderedPage;
