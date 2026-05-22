/**
 * Pipeline-shared job types.
 * `job_id` is `number` (D1 INTEGER) throughout — not string.
 */

/** R2 key conventions — single source of truth across workers. */
export const K = {
  raw:        (j: number)            => `raw/${j}/source.zip`,
  prepared:   (j: number, i: number) => `prepared/${j}/${String(i).padStart(4, "0")}.jpg`,
  meta:       (j: number)            => `prepared/${j}/meta.json`,
  scan:       (j: number, i: number) => `scan/${j}/${String(i).padStart(4, "0")}.msgpack`,
  scanMeta:   (j: number)            => `scan/${j}/meta.msgpack`,
  mask:       (j: number, i: number) => `mask/${j}/${String(i).padStart(4, "0")}.bin`,
  storyboard: (j: number, n: number) => `storyboard/${j}/${String(n).padStart(2, "0")}.jpg`,
  inpaint:    (j: number, i: number) => `inpaint/${j}/${String(i).padStart(4, "0")}.png`,
  typeset:    (j: number, i: number) => `typeset/${j}/${String(i).padStart(4, "0")}.png`,
  archive:    (j: number)            => `render/${j}.bnl`,
} as const;
