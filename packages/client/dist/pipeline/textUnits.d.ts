import type { TextBlock, TextUnit } from '../domain/text';
import { type TextRoleContext } from './textRole';
export declare function textUnitsFromBlocks(blocks: readonly TextBlock[], pageIndex: number, roleContext?: TextRoleContext): TextUnit[];
export declare function blockUnitId(pageIndex: number, blockIndex: number): string;
