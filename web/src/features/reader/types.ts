// Reader IR — shape every concrete reader source converts to before
// the UI touches it. Keeps the page-render components (`PageImage`,
// `ContinuousView`, `SinglePageView`) agnostic of where the pixels
// came from (BNL archive, manifest stream, future formats).

import type { ReactNode } from 'react'


/** One page in the reader. `url` is null while the page is still
 *  being fetched; the slot still reserves its aspect-ratio box so
 *  the layout never jumps. `width`/`height` are 0 when unknown —
 *  raw sources rarely ship dimensions until the <img> loads. */
export interface ReaderPage {
  /** Stable index. Streaming sources (BNL) deliver out of order; the
   *  index is the authoritative position regardless of arrival. */
  index:  number
  url:    string | null
  width:  number
  height: number
}


/** Display metadata for the toolbar / Discord presence / history. */
export interface ReaderMeta {
  workId:      number
  workTitle:   string
  /** Free-form chapter display string ("Ch.64", "第106话"). */
  chapterText: string
  chapterSub:  string | null
  /** target_lang for translated, source_lang for raw. */
  lang:        string | null
}


/** A neighbour the toolbar's prev/next buttons should navigate to.
 *  Resolved by `useReaderContext` once per chapter switch — the
 *  Reader component just renders a Link to whatever target this
 *  describes, no source-kind branching. */
export type ReaderNavTarget = {
  workId:     number
  numberNorm: string
  /** Optional preferred reading-lang. Carried so the URL preserves
   *  the user's choice across chapter jumps. */
  lang?:      string
  /** Optional active source. Same intent as `lang`. */
  src?:       number
}


export interface ReaderNav {
  prev: ReaderNavTarget | null
  next: ReaderNavTarget | null
}


/** Unified shape returned by the source-resolving hook. Toolbar +
 *  page list both read from this. */
export interface ReaderSource {
  pages:   ReaderPage[]
  /** Streaming source's blob-URL map (BNL). When present the page
   *  list prefers `urls.get(index)` over `pages[i].url`. */
  urls?:   ReadonlyMap<number, string>
  meta:    ReaderMeta
  nav:     ReaderNav
  loading: boolean
  error:   string | null
  toolbarExtra?: ReactNode
}
