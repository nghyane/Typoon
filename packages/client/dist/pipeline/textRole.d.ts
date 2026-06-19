import type { TextRole } from '../domain/planning';
import type { TextBlock } from '../domain/text';
export interface TextRoleContext {
    /** Median font size across the page — the "body" text size. */
    readonly bodyFontPx?: number;
}
export declare function textRoleContext(blocks: readonly TextBlock[]): TextRoleContext;
export declare function classifyTextBlockRole(block: TextBlock, context?: TextRoleContext): TextRole;
export declare function blockSourceFontPx(block: TextBlock): number;
