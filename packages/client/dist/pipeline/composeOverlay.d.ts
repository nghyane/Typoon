/**
 * Compose placements + translations into one or more PageOverlay objects.
 *
 * This is where the OCR/ONNX fork merges — ONNX regions are preferred
 * when available, falling back to OCR-only placement.
 *
 * Background expansion (SafeMargins) is pre-computed here from the image
 * so the renderer never needs raw pixel access.
 */
import type { TextRegion } from '../domain/regions';
import type { PageOverlay } from '../domain/overlay';
import type { PreparedPageHandle } from '../domain/prepared';
import type { TranslatedUnit } from '../domain/translation';
import type { RecognizedTextPage, TextUnit } from '../domain/text';
import type { TextPlacement } from '../domain/planning';
import type { ImagePixels } from '../domain/image';
import type { SafeMarginsDebug } from '../render/backgroundFit';
export interface ComposeOverlayArgs {
    readonly handle: PreparedPageHandle;
    readonly recognized: RecognizedTextPage;
    readonly textUnits: readonly TextUnit[];
    readonly translations: readonly TranslatedUnit[];
    readonly regions: readonly TextRegion[] | null;
    readonly placements?: readonly TextPlacement[];
    readonly placementMargins?: readonly SafeMarginsDebug[];
    /** Optional fallback for legacy callers that still compute margins on main. */
    readonly sourceImage?: ImagePixels;
}
export declare function buildOverlayPlacements(args: {
    readonly recognized: RecognizedTextPage;
    readonly textUnits: readonly TextUnit[];
    readonly regions: readonly TextRegion[] | null;
}): readonly TextPlacement[];
export declare function composeAndProjectOverlays(args: ComposeOverlayArgs): readonly PageOverlay[];
