import type { TextPlacement } from '../domain/planning';
import type { TextRegion } from '../domain/regions';
import type { RecognizedTextPage, TextUnit } from '../domain/text';
export declare function textPlacementsFromRecognition(recognized: RecognizedTextPage, units: readonly TextUnit[]): TextPlacement[];
export declare function layoutPlacementsFromRegions(recognized: RecognizedTextPage, units: readonly TextUnit[], regions: readonly TextRegion[]): TextPlacement[];
