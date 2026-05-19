/** Cross-worker types (kept tiny and stable). */

export interface PreparedPageMeta {
  index:  number;
  width:  number;
  height: number;
}

export interface PreparedChapterMeta {
  chapter_id:  string;
  strategy:    "one_to_one" | "stitch";
  is_color:    boolean;
  color_ratio: number;
  pages:       PreparedPageMeta[];
  groups:      number[][];   // per output page → list of raw indices that contributed
  raw_count:   number;
}

export type ReadingOrderRule = "rtl_columns" | "vertical_scroll" | "ltr_topdown";
export type BlockClass = "dialogue" | "sfx" | "narration";
export type ShapeKind = "dialogue" | "burst";
export type TextDirection = "horizontal" | "vertical";

export interface WordBox { text: string; bbox: [number, number, number, number]; }
export interface LineBox { text: string; bbox: [number, number, number, number]; rotation_deg: number; }

/** Wiki: derived from Lens word-bbox shorter side median.
 *  Drives the Rust fitter's hi_bound for binary font-size search. */
export interface TypesettingHint {
  font_size_px:       number;        // median(min(word_w, word_h))
  line_count:         number;
  avg_chars_per_line: number;
  text_direction:     TextDirection;
}

/** One bubble emitted by the grouper.
 *
 *  Polygon is the drawable area — already padded; render fits text
 *  into it directly. erase_polygons are tight per-member stripes
 *  (rasterized to the page mask PNG; orchestrator does flood-fill +
 *  bucket from there).
 *
 *  Invariant maintained by `group/`: polygon ⊇ word_union ⊇ erase. */
export interface BubbleGroup {
  idx:            number;           // reading-order on the page
  page_index:     number;
  source_text:    string;
  confidence:     number;

  polygon:        [number, number][];          // ≥ 4 vertices; ellipse = 24-gon
  bbox:           [number, number, number, number];   // polygon AABB (page-clipped)
  shape_kind:     ShapeKind;
  rotation_deg:   number;
  text_direction: TextDirection;
  class:          BlockClass;                  // sfx | narration | dialogue (translate routing)

  typesetting:    TypesettingHint | null;
  erase_polygons: [number, number][][];        // per-member OBB stripes

  used_hint:      boolean;                     // SOURCE vs FALLBACK (debug)
}

export interface RejectedBlock {
  bbox:         [number, number, number, number];
  text:         string;
  rotation_deg: number;
  is_vertical:  boolean;
  reason:       string;
}

export interface ScanPageResult {
  page_index:         number;
  page_size:          [number, number];
  detected_language:  string | null;
  reading_order_rule: ReadingOrderRule;
  groups:             BubbleGroup[];
  rejected:           RejectedBlock[];
  page_body_ratio:    number;          // 0 = fewer than 3 SOURCE samples
  tile_count:         number;
  timing_ms:          Record<string, number>;
}

export interface TranslationOp {
  key:        string;
  page_index: number;
  block_idx:  number;
  kind:       "dialogue" | "sfx" | "skip";
  class:      BlockClass;
  text:       string;
}

export interface TranslateResult {
  chapter_id:   string;
  translations: TranslationOp[];
  missing:      string[];
  windows:      { num: number; keys: string[]; char_count: number; latency_ms: number; usage?: any }[];
  errors?:      string[];
}

// ── Brief contracts ────────────────────────────────────────────────────
//
// Produced by `brief-worker`, consumed by `translate-worker` (everything)
// and the typeset stage (`noise.json` for the "kind=skip" predicate).
//
// Split across multiple R2 keys to give cache granularity: re-running
// translate with a tweaked prompt should not invalidate the vision-pass
// outputs, and series-level memory writers (future) only need to merge
// `glossary.json` / `characters.json` across chapters.

export interface BriefCharacter {
  name:        string;
  target_name: string;
  gender:      "male" | "female" | "unknown";
  role:        string;
  voice:       string;
}

export interface BriefAddressPair {
  speaker:  string;
  listener: string;
  pair:     string;
}

/** Index file pointing at the other artefacts; the only key consumers
 *  need to remember. Drives Cache API entries (key = `briefIndexKey`,
 *  value = ETag-like hash of the chunk count + chapter_id). */
export interface BriefIndex {
  chapter_id:  string;
  version:     1;
  chunk_count: number;            // how many vision calls ran
  has_prose:   boolean;
  has_chars:   boolean;
  has_address: boolean;
  has_gloss:   boolean;
  has_notes:   boolean;
  has_noise:   boolean;
  /** Total bubbles classified as noise (deterministic + vision). */
  noise_count: number;
  /** Pages where every bubble is noise. Whole-page chrome. */
  noise_pages: number[];
  timing_ms:   Record<string, number>;
}
