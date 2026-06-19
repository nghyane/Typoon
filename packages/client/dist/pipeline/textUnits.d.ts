import type { TextBlock, TextUnit } from '../domain/text';
export declare function textUnitsFromBlocks(blocks: readonly TextBlock[], pageIndex: number): TextUnit[];
export declare function blockUnitId(pageIndex: number, blockIndex: number): string;
