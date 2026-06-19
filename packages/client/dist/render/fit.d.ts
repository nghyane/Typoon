import type { TextPlacement } from '../domain/planning';
import { type SafeMarginsDebug } from './backgroundFit';
import { type FitRect } from './fitGeometry';
import { type LineLayoutCandidate } from './lineComposer';
import type { FontProfile } from './font';
import { type TypesetDirection } from './fitLayout';
import type { RenderLanguageContext } from './languageProfile';
export type { TypesetDirection };
type FontIntentReason = 'role-standard' | 'source' | 'fallback-role-median' | 'fallback-geometry';
export interface CssFitInput {
    readonly placement: TextPlacement;
    readonly text: string;
    readonly sourceText?: string;
}
export interface CssFitResult {
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
    readonly desiredFontSizePx: number;
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
export declare function fitPageText(items: readonly CssFitInput[], pageSize: readonly [number, number], font: FontProfile, placementMargins?: readonly SafeMarginsDebug[], fontContextPlacements?: readonly TextPlacement[], languageContext?: RenderLanguageContext): CssFitResult[];
