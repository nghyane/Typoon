/** Service binding RPC interfaces — what the workflow expects from each
 *  downstream service. Kept here so pipeline.ts can stay focused on the
 *  workflow shape. */

import type { PreparedJobMeta } from "./types";

export interface MediaService {
  prepareChapter(args: {
    job_id:    number;
    zip_key:   string;
    strategy?: string;
  }): Promise<PreparedJobMeta>;
}

export interface ScanService {
  scanChapter(args: {
    job_id:       number;
    pages:        { page_index: number; prepared_key: string; is_color: boolean }[];
    lang_hint?:   string;
    total_pages?: number;
  }): Promise<{
    scan_keys:       string[];
    storyboard_keys: string[];
    timings_ms:      Record<string, number>;
  }>;
}

export interface BriefService {
  briefJob(args: {
    job_id:          number;
    source_lang:     string;
    target_lang:     string;
    is_color:        boolean;
    strategy:        string;
    scan_keys:       string[];
    storyboard_keys: string[];
    /** Optional R2 key of seed WorkContext (gzip+JSON) from client. */
    context_in_key?: string;
  }): Promise<{
    index_key:       string;
    /** R2 key of merged WorkContext (gzip+JSON), for the client to download. */
    context_out_key: string;
    chunk_count:     number;
    noise_count:     number;
    noise_pages:     number[];
    timing_ms:       Record<string, number>;
  }>;
}

export interface TranslateService {
  translateChapter(args: {
    job_id:      number;
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

export interface InpaintService {
  inpaintChapter(args: {
    job_id:       number;
    page_indices: number[];
    concurrency?: number;
  }): Promise<{
    results: {
      page_index:  number;
      output_key?: string;
      bubbles?:    number;
      tiles_shape?: string[];
      error?:      string;
    }[];
    wall_total_ms:    number;
    concurrency_used: number;
  }>;
}

export interface TypesetPackService {
  typesetAndPack(args: {
    job_id:        number;
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
    job_id:           number;
    /** Omitted when kind='analyze'. */
    archive_key?:     string;
    page_count:       number;
    /** R2 key of merged WorkContext, surfaced to client via /jobs/:id. */
    context_out_key?: string;
  }): Promise<void>;
  notifyProgress(args: {
    job_id:  number;
    stage:   string;
    index?:  number;
    total?:  number;
  }): Promise<void>;
  notifyError(args: {
    job_id:  number;
    stage:   string;
    message: string;
  }): Promise<void>;
}

/** Brand to keep RPC stubs away from raw object exports. */
export type Service<T> = T & { connect?: never };
