export interface FontMetrics {
    readonly unitsPerEm: number;
    readonly ascent: number;
    readonly descent: number;
    readonly lineGap: number;
}
export interface FontProfile {
    readonly family: string;
    readonly cssFamily: string;
    readonly metrics: FontMetrics;
    readonly lineHeightRatio: number;
}
export declare const MANGA_FONT_PROFILE: FontProfile;
export declare const MANGA_FONT_FAMILY: string;
export declare function ensureMangaFontLoaded(sizePx?: number): Promise<void>;
