/* tslint:disable */
/* eslint-disable */

/**
 * Called once when the WASM module is loaded. Routes panics through
 * `console.error` instead of the cryptic `RuntimeError: unreachable`.
 */
export function __start(): void;

/**
 * Character budget for a bubble. Returns `[chars_per_line, n_lines, font_size_px]`.
 */
export function char_budget(bubble_w: number, bubble_h: number, page_width: number, src_font_size_px: number, src_line_count: number): Uint32Array;

/**
 * Render a page (RGBA-native, single-canvas).
 *
 * Input is the inpaint output (clean page) as RGBA `[H, W, 4]`. The
 * `polygons` array carries the **drawable** region per bubble — already
 * the inner text area as decided by the grouper. No border scanning.
 *
 * Output is `{ width, height, rgba: Uint8Array, bubbles: [...] }`.
 * JS encodes the returned RGBA straight via `@jsquash/png`.
 */
export function render_page(request: any, clean_rgba: Uint8Array): any;

/**
 * Stitch RGB pages into a vertical strip.
 *
 * `pages_meta` is `{ pages: [{ width, height }, ...] }`. `pages_data` is
 * the concatenation of each page's RGB buffer in declaration order; the
 * helper sums `page.width * page.height * 3` to know where each page
 * starts.
 */
export function stitch_pages(pages_meta: any, pages_data: Uint8Array): any;

export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __start: () => void;
    readonly char_budget: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly render_page: (a: any, b: number, c: number) => [number, number, number];
    readonly stitch_pages: (a: any, b: number, c: number) => [number, number, number];
    readonly version: () => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
