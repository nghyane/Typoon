import type { TranslationRequest } from '../domain/run';
type TranslationScope = NonNullable<TranslationRequest['scope']>;
export interface SourcePlan {
    readonly statusPages: readonly number[];
    readonly loadOrder: readonly number[];
    readonly prepareOrder: readonly number[];
}
/**
 * Build source orders for a run.
 *
 * loadOrder can prioritize the current page.
 * prepareOrder is allowed to differ because continuous prepare needs
 * source-natural input order even if loading is prioritized.
 */
export declare function buildSourcePlan(args: {
    readonly pageCount: number;
    readonly scope: TranslationScope;
    readonly priority?: TranslationRequest['priority'];
    readonly preparation: TranslationRequest['preparation'];
}): SourcePlan;
export {};
