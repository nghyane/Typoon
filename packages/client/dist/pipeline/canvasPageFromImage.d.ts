import type { CanvasPage } from '../domain/canvas';
import type { ImageInput } from '../image/input';
export declare function canvasPageFromImage(input: ImageInput, options: {
    readonly pageIndex: number;
}): Promise<CanvasPage>;
