import type { LayoutPlan, SegmentRequest, SegmentScript, TranslationVersion } from '../domain/segment';
import type { RecognizedTextPage } from '../domain/text';
export interface TranslationPostEditor {
    readonly name: string;
    postEdit(args: {
        readonly request: SegmentRequest;
        readonly transcript: readonly RecognizedTextPage[];
        readonly script: SegmentScript;
        readonly base: TranslationVersion;
        readonly layout: LayoutPlan;
        readonly signal?: AbortSignal;
    }): Promise<TranslationVersion>;
}
