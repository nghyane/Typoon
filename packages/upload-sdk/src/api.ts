// Shared types for the upload-init / upload-finalize handshake. Mirror
// the FastAPI schemas in `typoon/api/routes/upload.py`. When the engine
// adds a field, sync this file by hand (same drift policy as the rest
// of the engine API types).

export interface UploadInitBody {
  byte_size: number
}

export interface InitPart {
  number: number
  url:    string
}

export interface UploadInitOut {
  /** Material chapters will be created against. Per-material init
   *  echoes back the path arg; per-work init returns the
   *  server-resolved (lazy-created) upload-origin material id. */
  material_id: number
  tmp_id:      string
  upload_id:   string
  parts:       InitPart[]
  part_size:   number
  expires_in:  number
}

export interface FinalizePart {
  number: number
  etag:   string
}

export interface UploadFinalizeBody {
  tmp_id:    string
  upload_id: string
  parts:     FinalizePart[]
  /** Free-form chapter label (e.g. "Chương 040", "第106话",
   *  "Extra: Volume Cover"). Stored verbatim for display. */
  label?:    string
  /** Source manifest URL for this chapter. When provided, the engine
   *  dedups against existing chapter rows (same material × upstream_url)
   *  so multiple users uploading the same source chapter share one row. */
  upstream_url?: string
  /** Canonical chapter key (work_chapters.number_norm). Drives the
   *  work_chapter row that dedups chapters across sources of the
   *  same Work. Engine generates a sequential fallback when omitted. */
  number_norm?: string
  /** BCP-47 of the pixels in this upload. Server stamps it on the
   *  chapter row so spawn-translate uses it as the source language
   *  instead of falling back to `material.languages[0]` (wrong for
   *  any material that hosts chapters in multiple languages). */
  source_lang?: string
}

export interface UploadAbortBody {
  tmp_id:    string
  upload_id: string
}

/** Engine-returned chapter row. Kept loose so consumers can narrow as
 *  needed; full type lives in the web SPA's `ApiChapter`. */
export interface ApiChapterLike {
  id:           number
  material_id:  number
  number:       string
  page_count:   number
  // …other fields the consumer may project; keep this loose so the SDK
  // doesn't pin the full API shape.
}

/** Minimal client surface the SDK depends on. The web SPA's `api.ts`
 *  and the extension's `TypoonClient` both implement this shape. */
export interface UploadHttpClient {
  uploadInit(materialId: number, body: UploadInitBody): Promise<UploadInitOut>
  uploadFinalize(
    materialId: number, body: UploadFinalizeBody,
  ): Promise<ApiChapterLike>
  uploadAbort(materialId: number, body: UploadAbortBody): Promise<void>
}

/** Extension of `UploadHttpClient` for the per-work convenience
 *  route. The SPA's `api.ts` implements both shapes; the extension
 *  has no use for the work-level flow and may omit it. */
export interface WorkUploadHttpClient extends UploadHttpClient {
  workUploadInit(workId: number, body: UploadInitBody): Promise<UploadInitOut>
}
