import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState, useMemo, useEffect } from 'react'
import { useHeaderStore } from '../store/header'
import { useProjectEvents } from '@shared/lib/events'
import { api, type ApiChapter } from '@shared/api/api'
import { useDelayedFlag } from '@shared/lib/useDelayedFlag'
import { ProjectHero } from '@features/project-detail/ProjectHero'
import { ChapterList } from '@features/project-detail/ChapterList'
import { SelectionBar } from '@features/project-detail/SelectionBar'
import { useChapterMutations } from '@features/project-detail/mutations'
import { TabBar } from '@features/project-detail/TabBar'
import { GlossaryPanel } from '@features/project-detail/GlossaryPanel'
import { SettingsPanel } from '@features/project-detail/SettingsPanel'
import { UploadChapterDialog } from '@features/project-detail/UploadChapterDialog'
import { chapterStats } from '@features/project-detail/chapter'
import { matchFilter, type Filter, type Tab } from '@features/project-detail/filter'

interface SearchParams {
  tab?:    Tab
  filter?: Filter
  q?:      string
}

function ProjectDetailPage() {
  const { projectId } = Route.useParams()
  const { tab = 'chapters', filter = 'all', q = '' } = Route.useSearch()
  const nav = Route.useNavigate()
  // Route params are strings; coerce once. The comparisons below treat
  // anything non-positive (NaN, 0) as "not yet ready" so route-level
  // checks don't pile up.
  const id = Number(projectId)
  const validId = Number.isInteger(id) && id > 0

  // Live updates for this project — opens an SSE subscription scoped
  // to project <id>. Closes when the user navigates away.
  useProjectEvents(id)

  // URL is the state of truth. setX = navigate with new search params.
  const setTab    = (next: Tab)    => nav({ search: (s) => ({ ...s, tab:    next }) })
  const setFilter = (next: Filter) => nav({ search: (s) => ({ ...s, filter: next }) })
  const setQ      = (next: string) => nav({ search: (s) => ({ ...s, q: next || undefined }) })

  const [sel,        setSel]        = useState<Set<number>>(new Set())
  const [uploadOpen, setUploadOpen] = useState(false)

  // ChapterList already constructs its own copy of these mutations.
  // We instantiate again at the route level only because SelectionBar
  // needs `startMany`, which is unrelated to the per-row actions. The
  // hook is cheap (just registers React Query subscriptions) so a
  // second copy is fine; keeping the bar's wiring at the same level as
  // `sel` state avoids drilling a callback through ChapterList.
  const mutations = useChapterMutations(id)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)

  const { data: project, isError: pErr, isPending: pLoad } = useQuery({
    queryKey: ['projects', id],
    queryFn:  () => api.getProject(id),
    enabled:  validId,
  })

  const { data: chapters = [], isPending: cLoad } = useQuery({
    queryKey: ['projects', id, 'chapters'],
    queryFn:  () => api.listChapters(id),
    enabled:  validId,
    // SSE drives most updates, but the stream can drop (network blip,
    // backgrounded tab) and a worker crash leaves no event to listen
    // for. Poll while anything is in flight so the list eventually
    // reflects reality even with no events.
    refetchInterval: (q) => {
      const data = q.state.data as ApiChapter[] | undefined
      const busy = data?.some((c) => c.state === 'running' || c.state === 'pending')
      return busy ? 5000 : false
    },
  })

  const showHeroSkeleton = useDelayedFlag(pLoad && !project, 250)

  const sorted = useMemo(
    () => [...chapters].sort((a, b) => a.position - b.position),
    [chapters],
  )

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return sorted.filter((c) => {
      if (!matchFilter(c, filter)) return false
      if (!needle) return true
      return (
        c.number.toLowerCase().includes(needle) ||
        (c.title?.toLowerCase().includes(needle) ?? false)
      )
    })
  }, [sorted, filter, q])

  const stats        = chapterStats(chapters)
  const allChecked   = sel.size === filtered.length && filtered.length > 0
  const existingNums = useMemo(() => new Set(chapters.map((c) => c.number)), [chapters])

  useEffect(() => {
    if (project) setHeader(project.title, [{ label: 'Dự án', to: '/projects' }])
    return () => clearHeader()
  }, [project, setHeader, clearHeader])

  const toggleOne = (cid: number) =>
    setSel((prev) => {
      const next = new Set(prev)
      if (next.has(cid)) next.delete(cid); else next.add(cid)
      return next
    })

  const toggleAll = () =>
    sel.size === filtered.length && filtered.length > 0
      ? setSel(new Set())
      : setSel(new Set(filtered.map((c) => c.chapter_id)))

  if (pErr || (!pLoad && !project)) return <NotFound />

  return (
    <div>
      {project ? (
        <ProjectHero
          project={project}
          stats={stats}
          isOwner={project.is_owner}
          onAddChapters={() => setUploadOpen(true)}
        />
      ) : showHeroSkeleton ? (
        <HeroSkeleton />
      ) : (
        <div className="h-[124px]" />  /* reserve space, no skeleton flash */
      )}

      <TabBar value={tab} onChange={setTab} />

      <div className="px-4 sm:px-6 py-4 sm:py-5">
        {tab === 'chapters' && (
          <ChapterList
            projectId={id}
            isOwner={project?.is_owner ?? false}
            chapters={filtered}
            loading={cLoad}
            stats={stats}
            filter={filter} setFilter={setFilter}
            q={q}           setQ={setQ}
            sel={sel}       toggleOne={toggleOne} toggleAll={toggleAll}
            allChecked={allChecked}
            onPull={() => setUploadOpen(true)}
          />
        )}
        {tab === 'glossary' && <GlossaryPanel projectId={id} />}
        {tab === 'settings' && project && <SettingsPanel project={project} />}
      </div>

      {sel.size > 0 && (
        <SelectionBar
          count={sel.size}
          onClear={() => setSel(new Set())}
          onStart={() => {
            mutations.startMany.mutate(Array.from(sel), {
              onSettled: () => setSel(new Set()),
            })
          }}
          pending={mutations.startMany.isPending}
        />
      )}

      {project && (
        <UploadChapterDialog
          open={uploadOpen}
          onClose={() => setUploadOpen(false)}
          project={project}
          existing={existingNums}
        />
      )}
    </div>
  )
}

function HeroSkeleton() {
  return (
    <div className="px-6 pt-6 pb-5 flex items-start gap-4 animate-pulse">
      <div className="w-20 aspect-[2/3] rounded-md bg-surface shrink-0" />
      <div className="flex-1 min-w-0 space-y-2 pt-1">
        <div className="h-7 w-72 rounded bg-surface" />
        <div className="h-4 w-96 rounded bg-surface" />
      </div>
    </div>
  )
}

function NotFound() {
  return (
    <div className="p-6">
      <p className="text-sm text-error-text font-medium">Không tìm thấy dự án.</p>
      <Link to="/projects" className="text-sm text-text-subtle underline mt-1 inline-block">← Quay lại</Link>
    </div>
  )
}

const TABS: Tab[] = ['chapters', 'glossary', 'settings']
const FILTERS: Filter[] = ['all', 'idle', 'running', 'done', 'error']

export const Route = createFileRoute('/projects/$projectId/')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    tab:    TABS.includes(s.tab as Tab) ? (s.tab as Tab) : undefined,
    filter: FILTERS.includes(s.filter as Filter) ? (s.filter as Filter) : undefined,
    q:      typeof s.q === 'string' && s.q.length > 0 ? s.q : undefined,
  }),
  component: ProjectDetailPage,
})
