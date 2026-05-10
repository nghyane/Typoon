// Stable error so callers (popup toasts, queue retry) can branch without
// parsing strings. Mirrors web/src/shared/api/api.ts.
export class BackendUnavailableError extends Error {
  constructor() {
    super('Máy chủ tạm thời không phản hồi. Có thể đang bảo trì hoặc khởi động lại.')
    this.name = 'BackendUnavailableError'
  }
}

/** 401 from the engine — token revoked or wrong. UI re-runs Setup. */
export class UnauthorizedError extends Error {
  constructor() {
    super('Token không hợp lệ hoặc đã bị thu hồi.')
    this.name = 'UnauthorizedError'
  }
}

/** 429 — quota exhausted. Surface specifically so the UI can word it
 *  ("Đã hết quota chương hôm nay") instead of a generic error. */
export class QuotaExceededError extends Error {
  constructor(message?: string) {
    super(message ?? 'Đã đạt giới hạn upload, thử lại sau.')
    this.name = 'QuotaExceededError'
  }
}
