/**
 * Shared types between pipeline workflow and queue consumers.
 * Kept narrow — only what flows across worker boundaries.
 */

export interface PreparedPage {
  index:  number;
  width:  number;
  height: number;
}

export interface PreparedChapterMeta {
  chapter_id:  string;
  strategy:    "one_to_one" | "stitch";
  is_color:    boolean;
  color_ratio: number;
  pages:       PreparedPage[];
  groups:      number[][];
  raw_count:   number;
}

/** Message enqueued to typoon-scan-queue per page. */
export interface ScanJob {
  workflow_id:  string;
  chapter_id:   string;
  page_index:   number;
  total_pages:  number;
  prepared_key: string;
  is_color:     boolean;
}

/** Message enqueued to typoon-inpaint-queue per page. */
export interface InpaintJob {
  workflow_id:  string;
  chapter_id:   string;
  page_index:   number;
  total_pages:  number;
  prepared_key: string;
  scan_key:     string;
  mask_key:     string;
}

/** Payload sent via instance.sendEvent("scans-done", ...) */
export interface ScansDonePayload {
  scan_keys: string[];
  mask_keys: string[];
}

/** Payload sent via instance.sendEvent("inpaints-done", ...) */
export interface InpaintsDonePayload {
  inpaint_keys: string[];
}

/** R2 key conventions — single source of truth. */
export const K = {
  raw:        (c: string)            => `raw/${c}/source.zip`,
  prepared:   (c: string, i: number) => `prepared/${c}/${String(i).padStart(4, "0")}.jpg`,
  meta:       (c: string)            => `prepared/${c}/meta.json`,
  scan:       (c: string, i: number) => `scan/${c}/${String(i).padStart(4, "0")}.msgpack`,
  scanMeta:   (c: string)            => `scan/${c}/meta.msgpack`,
  mask:       (c: string, i: number) => `mask/${c}/${String(i).padStart(4, "0")}.bin`,
  storyboard: (c: string, n: number) => `storyboard/${c}/${String(n).padStart(2, "0")}.jpg`,
  inpaint:    (c: string, i: number) => `inpaint/${c}/${String(i).padStart(4, "0")}.png`,
  typeset:    (c: string, i: number) => `typeset/${c}/${String(i).padStart(4, "0")}.png`,
  archive:    (c: string)            => `render/${c}.bnl`,
} as const;
