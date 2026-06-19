import type { TextLayoutHint, TextPlacement } from '../domain/planning';
export declare function withTextLayoutHints(placements: readonly TextPlacement[], pageSize: readonly [number, number]): TextPlacement[];
export declare function defaultLayoutHint(): TextLayoutHint;
