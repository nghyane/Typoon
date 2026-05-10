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
  tmp_id:     string
  upload_id:  string
  parts:      InitPart[]
  part_size:  number
  expires_in: number
}

export interface FinalizePart {
  number: number
  etag:   string
}

export interface UploadFinalizeBody {
  tmp_id:    string
  upload_id: string
  parts:     FinalizePart[]
  number?:   string
  title?:    string
}

export interface UploadAbortBody {
  tmp_id:    string
  upload_id: string
}

/** Engine-returned chapter row (kept loose — consumers narrow as needed). */
export interface ApiChapterLike {
  chapter_id: number
  project_id: number
  number:     string
  state:      string
  page_count: number
  // ...other fields the consumer may project; keep this loose so the SDK
  // doesn't pin the full API shape.
}

/** Minimal client surface the SDK depends on. The web SPA's `api.ts`
 *  and the extension's `TypoonClient` both implement this shape. */
export interface UploadHttpClient {
  uploadInit(projectId: number, body: UploadInitBody): Promise<UploadInitOut>
  uploadFinalize(projectId: number, body: UploadFinalizeBody): Promise<ApiChapterLike>
  uploadAbort(projectId: number, body: UploadAbortBody): Promise<void>
}
