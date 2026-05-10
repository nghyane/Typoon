// Public surface of `core/typoon`. Anything not re-exported here is
// considered internal — callers in `shell/` / `entrypoints/` import only
// from the package root.

export { TypoonClient, type TypoonClientOpts } from './client'
export {
  BackendUnavailableError,
  QuotaExceededError,
  UnauthorizedError,
} from './errors'
export type {
  ApiChapter,
  ApiMe,
  ApiMeProject,
  ApiProject,
  ApiTokenInfo,
  ChapterState,
  CreateProjectOpts,
} from './types'
