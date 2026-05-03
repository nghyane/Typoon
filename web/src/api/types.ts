// Mirror of Pydantic API models — keep in sync with typoon/api/models.py

export interface Progress {
  stage: string
  page_index: number
  page_total: number
}

export interface ChapterOut {
  chapter_id: number
  project_id: number
  idx: number
  state: 'idle' | 'pending' | 'running' | 'error' | 'done'
  stage: 'scan' | 'translate' | 'render' | ''
  page_count: number
  error: string
  progress: Progress | null
}

export interface ProjectOut {
  project_id: number
  slug: string
  title: string
  source_lang: string
  target_lang: string
}

// API request bodies
export interface ImportBody {
  folder: string
  title: string
  source_lang?: string
  target_lang?: string
}

// Pagination params — ready for when API supports it
export interface ChapterParams {
  state?: ChapterOut['state']
  page?: number
  limit?: number
}

// SSE event types — mirror runs/events.py
export type SSEEventType =
  | 'StageStarted'
  | 'StageDone'
  | 'StageFailed'
  | 'PageDone'
  | 'ChapterDownloaded'
  | 'ChapterFailed'
  | 'LLMCall'
  | 'LLMResponse'

export interface SSEEvent {
  type: SSEEventType
  chapter_id?: number
  stage?: string
  page_index?: number
  page_total?: number
  error?: string
  [key: string]: unknown
}
