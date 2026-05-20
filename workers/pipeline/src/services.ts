/** Service binding RPC interfaces — what the workflow expects from each
 *  downstream service. Kept here so pipeline.ts can stay focused on the
 *  workflow shape. */

import type { PreparedChapterMeta } from "./types";

export interface MediaService {
  prepareChapter(args: {
    chapter_id: number;
    zip_key:    string;
    strategy?:  string;
  }): Promise<PreparedChapterMeta>;
}

export interface ScanService {
  scanChapter(args: {
    chapter_id:   number;
    workflow_id:  string;
    pages:        { page_index: number; prepared_key: string; is_color: boolean }[];
    lang_hint?:   string;
    total_pages?: number;
  }): Promise<{
    scan_keys:       string[];
    mask_keys:       string[];
    storyboard_keys: string[];
    timings_ms:      Record<string, number>;
  }>;
}

export interface BriefService {
  briefChapter(args: {
    chapter_id:      number;
    source_lang:     string;
    target_lang:     string;
    is_color:        boolean;
    strategy:        string;
    scan_keys:       string[];
    storyboard_keys: string[];
  }): Promise<{
    index_key:   string;
    chunk_count: number;
    noise_count: number;
    noise_pages: number[];
    timing_ms:   Record<string, number>;
  }>;
}

export interface TranslateService {
  translateChapter(args: {
    chapter_id:  number;
    scan_keys:   string[];
    source_lang: string;
    target_lang: string;
    use_brief?:  boolean;
  }): Promise<{
    output_key:   string;
    translations: number;
    missing:      number;
    errors?:      string[];
  }>;
}

export interface TypesetPackService {
  typesetAndPack(args: {
    chapter_id:    number;
    pages:         { page_index: number; inpaint_key: string; scan_key: string; page_width: number }[];
    translate_key: string;
  }): Promise<{
    archive_key: string;
    size_bytes:  number;
    pages:       number;
    timings_ms:  Record<string, number>;
  }>;
}

export interface ApiCallbackService {
  finalize(args: {
    chapter_id:  number;
    draft_id:    number;
    archive_key: string;
    page_count:  number;
    scan_keys:   string[];
    mask_keys:   string[];
  }): Promise<void>;
  notifyProgress(args: {
    draft_id: number;
    stage:    string;
    index?:   number;
    total?:   number;
  }): Promise<void>;
  notifyError(args: {
    draft_id: number;
    stage:    string;
    message:  string;
  }): Promise<void>;
}

/** Brand to keep RPC stubs away from raw object exports. */
export type Service<T> = T & { connect?: never };
