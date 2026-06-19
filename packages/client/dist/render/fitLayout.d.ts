import type { TextPlacement, TextRole } from '../domain/planning';
import { type SafeMarginsDebug } from './backgroundFit';
import { type LineLayoutCandidate } from './lineComposer';
import { type FitRect } from './fitGeometry';
import type { FontProfile } from './font';
import type { DomMeasurer } from './textMeasure';
import { type RenderLanguageContext } from './languageProfile';
export type TypesetDirection = 'horizontal' | 'vertical';
type FontIntentReason = 'role-standard' | 'source' | 'fallback-role-median' | 'fallback-geometry';
export interface FitResult {
    readonly text: string;
    readonly fontSizePx: number;
    readonly lineHeightPx: number;
    readonly paddingXPx: number;
    readonly paddingYPx: number;
    readonly overflow: boolean;
    readonly rect: FitRect;
    readonly baseRect: FitRect;
    readonly maxDomFitPx: number;
    readonly capReason: string;
    readonly sourceFontPx: number | null;
    readonly roleMedianFontPx: number | null;
    readonly targetFontPx: number;
    readonly fontIntentReason: FontIntentReason;
    readonly fitReason: string;
    readonly direction: TypesetDirection;
    readonly directionReason: string;
    readonly layoutCandidate: LineLayoutCandidate;
    readonly lineCount: number;
    readonly lineScore: number;
    readonly maxFill: number;
    readonly edgeGuardPx: number;
    readonly fontShortSideRatio: number;
    readonly expansion: SafeMarginsDebug | null;
    readonly safeShapeUsed: boolean;
}
interface PageFontContext {
    readonly roleMedians: ReadonlyMap<TextRole, number>;
    readonly allMedianPx: number | null;
    readonly pageMaxPx: number;
    readonly preserveSourceScale: boolean;
}
export declare function fitLayout(placement: TextPlacement, text: string, sourceText: string | undefined, context: PageFontContext, font: FontProfile, measurer: DomMeasurer, preMargin: SafeMarginsDebug | null, languageContext?: RenderLanguageContext): FitResult;
export declare function pageFontContext(placements: readonly TextPlacement[], pageWidth: number, languageContext?: RenderLanguageContext): PageFontContext;
export {};
