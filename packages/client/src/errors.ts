export type BrowserSdkErrorCode =
  | 'WEBGPU_UNAVAILABLE'
  | 'MODEL_UNAVAILABLE'
  | 'MODEL_NOT_DOWNLOADED'
  | 'MODEL_DOWNLOAD_FAILED'
  | 'IMAGE_CORS_TAINTED'
  | 'IMAGE_DECODE_FAILED'
  | 'TEXT_RECOGNIZER_UNAVAILABLE'
  | 'OUT_OF_MEMORY'

export class BrowserSdkError extends Error {
  readonly code: BrowserSdkErrorCode

  constructor(code: BrowserSdkErrorCode, message: string, cause?: unknown) {
    super(message)
    this.name = 'BrowserSdkError'
    this.code = code
    if (cause !== undefined) this.cause = cause
  }
}
