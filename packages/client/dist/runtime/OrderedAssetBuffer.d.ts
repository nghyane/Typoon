import type { PageAsset } from '../domain/source';
type OrderedAsset = Pick<PageAsset, 'index'>;
/**
 * Buffers loaded assets and emits them to the preparation stream
 * in the expected preparation order.
 *
 * Assets may load out of order (priority queue, network variance),
 * but preparation may require a different order. Identity preparation can use
 * priority order, while continuous-strip preparation uses natural source order.
 */
export declare class OrderedAssetBuffer<T extends OrderedAsset = PageAsset> {
    /** How many assets have been emitted so far. */
    private emitted;
    private readonly buffer;
    private readonly skipped;
    private readonly prepareOrder;
    constructor(prepareOrder: readonly number[]);
    /**
     * Push a loaded asset.  Returns zero or more assets ready for
     * preparation, in source-index order.
     */
    push(index: number, asset: T): T[];
    /** Mark a failed page as non-blocking and emit any later ready assets. */
    skip(index: number): T[];
    private drainReady;
}
export {};
