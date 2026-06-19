import type { TextRole } from '../domain/planning';
import { type TextScript } from './textScript';
export interface RenderLanguageContext {
    readonly sourceLanguage?: string | null;
    readonly targetLanguage?: string | null;
}
export interface TextRenderProfile {
    readonly sourceFamily: LanguageFamily;
    readonly targetFamily: LanguageFamily;
    readonly targetScript: TextScript;
    readonly fontScale: number;
    readonly minReadableFontPx: number;
    readonly innerPadXEm: number;
    readonly innerPadYEm: number;
    readonly geometryGrowXEm: number;
    readonly geometryGrowYEm: number;
    readonly geometryGrowWidthRatio: number;
    readonly geometryGrowHeightRatio: number;
    readonly pageMaxFraction: number;
    readonly hierarchyMaxFraction: number;
}
export type LanguageFamily = 'latin' | 'hangul' | 'han' | 'kana' | 'mixed-cjk' | 'unknown';
export declare function textRenderProfile(text: string, context: RenderLanguageContext | undefined, role: TextRole, sourceText?: string | null): TextRenderProfile;
export declare function pageRenderProfile(context: RenderLanguageContext | undefined): Pick<TextRenderProfile, 'pageMaxFraction' | 'hierarchyMaxFraction'>;
