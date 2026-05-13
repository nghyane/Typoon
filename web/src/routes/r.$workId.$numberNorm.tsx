// Unified reader route — /r/$workId/$numberNorm
//
// One URL pattern for both translated and raw reading. The route
// layer is thin: parses the URL, hands off to `useReader`, renders
// the shell. Source kind is resolved CLIENT-SIDE from the cached
// Work payload (`useWork`) — no extra round-trip.
//
// Search params:
//   • lang  — preferred reading lang. Override of viewerEntry.target_lang.
//             Kept in URL so deep links land on the right version.
//   • src   — active source material id. Drives the manifest fetch
//             when the picked version is raw; ignored for translations.
//
// Status branches:
//   loading           spinner
//   not-found         EmptyState
//   pending-render    "đang render xong"
//   no-source         "plugin chưa cài"
//   error             EmptyState
//   ready             toolbar + body

import { createFileRoute, redirect as routerRedirect } from '@tanstack/react-router'
import { useEffect } from 'react'
import { AlertTriangle } from 'lucide-react'

import { api, WorkRedirectedError } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'

import { ReaderToolbar, type ViewMode } from '@features/reader/ReaderToolbar'
import { ReaderBody } from '@features/reader/Reader'
import { useReader } from '@features/reader/useReader'


interface SearchParams {
  page?: number
  mode?: ViewMode
  lang?: string
  src?:  number
}


function ReaderPage() {
  const { workId: workIdStr, numberNorm } = Route.useParams()
  const { page = 0, mode = 'continuous', lang, src } = Route.useSearch()
  const nav = Route.useNavigate()
  const workId = Number(workIdStr)
  const validWorkId = Number.isInteger(workId) && workId > 0

  const setPage = (p: number) =>
    nav({ search: (s) => ({ ...s, page: p > 0 ? p : undefined }) })
  const setMode = (m: ViewMode) =>
    nav({ search: (s) => ({ ...s, mode: m === 'continuous' ? undefined : m }) })

  const reader = useReader({
    workId:     validWorkId ? workId : 0,
    numberNorm,
    lang,
    src,
  })

  // Reset scroll to top whenever the chapter changes. Without this,
  // jumping from a long chapter to a short one starts the new chapter
  // halfway down the page (or below the bottom) — the reader's
  // continuous view feels broken.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' })
  }, [workIdStr, numberNorm])
  if (!validWorkId) {
    return (
      <div className="px-4 py-16">
        <EmptyState icon={AlertTriangle} title="Đường dẫn không hợp lệ" />
      </div>
    )
  }

  return (
    <div className="min-h-dvh bg-bg">
      <ReaderToolbar
        workTitle={reader.meta.workTitle || 'Chương'}
        chapterText={reader.meta.chapterText}
        chapterSub={reader.meta.chapterSub}
        prev={reader.nav.prev}
        next={reader.nav.next}
        page={page}
        totalPages={reader.pages.length}
        mode={mode}
        onModeChange={setMode}
        onBack={() => nav({
          // Always return to the Work page, not the previous history
          // entry. History "back" lands on the prior chapter when the
          // user got here via next/prev — confusing and not what
          // clicking the manga title implies.
          to:     '/w/$workId',
          params: { workId: workIdStr },
          search: { src },
        })}
      />

      <ReaderContent
        reader={reader}
        numberNorm={numberNorm}
        page={page}
        mode={mode}
        onChange={setPage}
      />
    </div>
  )
}


/** Renders the body — pages OR a status placeholder. The toolbar
 *  above stays mounted regardless of status so the user can navigate
 *  back / to a sibling chapter while the current one is loading or
 *  errored. */
function ReaderContent({
  reader, numberNorm, page, mode, onChange,
}: {
  reader:     ReturnType<typeof useReader>
  numberNorm: string
  page:       number
  mode:       ViewMode
  onChange:   (p: number) => void
}) {
  if (reader.status === 'loading') {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (reader.status === 'not-found') {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Không tìm thấy chương"
          hint={`Chương "${numberNorm}" không có trong manga này.`}
        />
      </div>
    )
  }
  if (reader.status === 'pending-render') {
    return (
      <div className="px-6 py-16 text-center max-w-md mx-auto">
        <p className="text-sm font-medium text-text">Chương chưa render xong</p>
        <p className="text-xs text-text-subtle mt-1">
          Bản dịch đang được hệ thống tạo ra — quay lại sau ít phút.
        </p>
      </div>
    )
  }
  if (reader.status === 'no-source') {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Nguồn chưa được cài"
          hint="Vào Cài đặt → Nguồn để cài plugin tương ứng."
        />
      </div>
    )
  }
  if (reader.status === 'error') {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được chương"
          hint={reader.error ?? 'Selector hoặc archive có thể đã hỏng.'}
        />
      </div>
    )
  }
  return (
    <ReaderBody
      source={reader}
      mode={mode}
      page={page}
      onChange={onChange}
    />
  )
}


export const Route = createFileRoute('/r/$workId/$numberNorm')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    page: typeof s.page === 'number' && s.page > 0 ? s.page : undefined,
    mode: s.mode === 'single' ? 'single' : undefined,
    lang: typeof s.lang === 'string' ? s.lang : undefined,
    src:  typeof s.src  === 'number' ? s.src  : undefined,
  }),
  // Intercept Work merge redirect BEFORE rendering. Same pattern as
  // `/w/$workId`: ensure-fetch surfaces `WorkRedirectedError` and we
  // throw a router redirect carrying every search param so the
  // reader resumes on the same chapter under the canonical Work id.
  beforeLoad: async ({ params, search, context }) => {
    const workId = Number(params.workId)
    if (!Number.isInteger(workId) || workId <= 0) return
    try {
      await context.queryClient.ensureQueryData({
        queryKey: qk.work.byId(workId),
        queryFn:  () => api.getWork(workId),
      })
    } catch (err) {
      if (err instanceof WorkRedirectedError) {
        throw routerRedirect({
          to:     '/r/$workId/$numberNorm',
          params: {
            workId:     String(err.newId),
            numberNorm: params.numberNorm,
          },
          search,
          replace: true,
        })
      }
    }
  },
  component: ReaderPage,
  staticData: { chrome: 'bare' },
})
