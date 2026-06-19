import type { TextPlacement } from './planning';
import type { TranslatedUnit, TranslationUnit } from './translation';
export interface SegmentRequest {
    readonly pageIndexes: readonly number[];
    readonly sourceLang: string | null;
    readonly targetLang: string;
    /** Request LLM post-edit after machine translation. */
    readonly postEdit: boolean;
    readonly sessionId?: string;
}
export interface SegmentScript {
    /** Reading-order translation units for the whole segment. */
    readonly units: readonly TranslationUnit[];
}
export interface TranslationVersion {
    readonly id: string;
    readonly method: 'machine' | 'post_edit';
    readonly baseId?: string;
    readonly units: readonly TranslatedUnit[];
}
export interface LayoutPlan {
    readonly pages: readonly PageLayout[];
}
export interface PageLayout {
    readonly pageIndex: number;
    readonly placements: readonly TextPlacement[];
}
