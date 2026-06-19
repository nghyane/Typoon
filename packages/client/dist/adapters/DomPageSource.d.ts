import type { PageSource } from '../domain/source';
import { type DomPageOptions } from './domShared';
export type DomPageSourceOptions = DomPageOptions;
export declare class DomPageSource implements PageSource {
    private readonly container;
    private readonly imageSelector;
    constructor(container: HTMLElement, options?: DomPageSourceOptions);
    get pageCount(): number;
    loadPage(index: number): HTMLImageElement;
}
