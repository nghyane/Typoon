import type { RenderedPage } from '../domain/translation';
import { type OverlayOptions } from '../render/overlay';
import { type DomPageOptions } from './domShared';
export type DomOverlayTargetOptions = DomPageOptions;
export declare class DomOverlayTarget {
    private readonly container;
    private readonly imageSelector;
    private readonly hostSelector?;
    constructor(container: HTMLElement, options?: DomOverlayTargetOptions);
    pageHost(index: number): HTMLElement;
    attach(index: number, page: RenderedPage, options?: OverlayOptions): HTMLElement;
}
