export interface DomPageOptions {
    readonly imageSelector?: string;
    readonly hostSelector?: string;
}
export declare function imagesIn(container: HTMLElement, imageSelector?: string): HTMLImageElement[];
export declare function imageAt(container: HTMLElement, index: number, imageSelector?: string): HTMLImageElement;
export declare function pageHostForImage(image: HTMLImageElement, hostSelector: string | undefined, index: number): HTMLElement;
