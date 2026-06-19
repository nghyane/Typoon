import type { TextRole } from '../domain/planning';
import type { BubbleShapeProfile } from './bubbleShape';
import type { FontProfile } from './font';
import type { DomMeasurer } from './textMeasure';
export type LineLayoutCandidate = 'baseline' | 'vertical';
export interface LineComposition {
    readonly text: string;
    readonly lines: readonly string[];
    readonly candidate: LineLayoutCandidate;
    readonly lineCount: number;
    readonly heightPx: number;
    readonly overflowHeight: boolean;
    readonly overflowWidth: boolean;
    readonly fits: boolean;
    readonly score: number;
    /** Max ratio of line width / line limit across all lines. >0.85 = too dense. */
    readonly maxFill: number;
}
export declare function composeLines(args: {
    readonly text: string;
    readonly width: number;
    readonly height: number;
    readonly fontSizePx: number;
    readonly font: FontProfile;
    readonly fontWeight: string;
    readonly role: TextRole;
    readonly direction?: 'horizontal' | 'vertical';
    readonly shapeProfile?: BubbleShapeProfile;
    readonly sourceLineCount?: number;
    readonly measurer: DomMeasurer;
}): LineComposition;
