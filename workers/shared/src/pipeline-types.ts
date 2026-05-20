/**
 * Pipeline-specific job types shared between typoon-pipeline and queue consumers.
 * chapter_id is `number` (D1 INTEGER) throughout \u2014 not string.
 */

/** R2 key conventions \u2014 single source of truth across all workers. */
export const K = {
  raw:        (c: number)            => `raw/${c}/source.zip`,
  prepared:   (c: number, i: number) => `prepared/${c}/${String(i).padStart(4, "0")}.jpg`,
  meta:       (c: number)            => `prepared/${c}/meta.json`,
  scan:       (c: number, i: number) => `scan/${c}/${String(i).padStart(4, "0")}.msgpack`,
  scanMeta:   (c: number)            => `scan/${c}/meta.msgpack`,
  mask:       (c: number, i: number) => `mask/${c}/${String(i).padStart(4, "0")}.bin`,
  storyboard: (c: number, n: number) => `storyboard/${c}/${String(n).padStart(2, "0")}.jpg`,
  inpaint:    (c: number, i: number) => `inpaint/${c}/${String(i).padStart(4, "0")}.png`,
  typeset:    (c: number, i: number) => `typeset/${c}/${String(i).padStart(4, "0")}.png`,
  archive:    (c: number)            => `render/${c}.bnl`,
} as const;

/** Message enqueued to typoon-inpaint-queue per page. */
export interface InpaintJob {
  workflow_id:  string;
  chapter_id:   number;
  page_index:   number;
  total_pages:  number;
  prepared_key: string;
  scan_key:     string;
  mask_key:     string;
}

/** Payload sent via instance.sendEvent("inpaints-done", ...) */
export interface InpaintsDonePayload {
  inpaint_keys: string[];
}
