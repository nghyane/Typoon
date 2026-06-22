import type { TextPlacement } from '../domain/planning';
import type { TextRegion } from '../domain/regions';
import type { RecognizedTextPage, TextUnit } from '../domain/text';
import { type TextRoleContext } from './textRole';
export declare function textPlacementsFromRecognition(recognized: RecognizedTextPage, units: readonly TextUnit[], roleContext?: TextRoleContext): TextPlacement[];
export declare function layoutPlacementsFromRegions(recognized: RecognizedTextPage, units: readonly TextUnit[], regions: readonly TextRegion[], roleContext?: TextRoleContext): TextPlacement[];
