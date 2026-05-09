import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState, useMemo, useEffect } from 'react'
import { useHeaderStore } from '../store/header'
import { useProjectInterest } from '../store/interest'
import { api, type ApiChapter } from '@shared/api/api'
import { useDelayedFlag } from '@shared/lib/useDelayedFlag'
import { ProjectHero } from '@features/project-detail/ProjectHero'
import { ChapterList } from '@features/project-detail/ChapterList'
import { SelectionBar } from '@features/project-detail/SelectionBar'
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
  const id = Number(projectId)

  // Tell the SSE hook this tab cares about events for this project.
  useProjectInterest(isNaN(id) ? null : id)

  // URL is the state of truth. setX = navigate with new search params.
  const setTab    = (next: Tab)    => nav({ search: (s) => ({ ...s, tab:    next }) })
  const setFilter = (next: Filter) => nav({ search: (s) => ({ ...s, filter: next }) })
  const setQ      = (next: string) => nav({ search: (s) => ({ ...s, q: next || undefined }) })

  const [sel,        setSel]        = useState<Set<number>>(new Set())
  const [uploadOpen, setUploadOpen] = useState(false)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)

  const { data: project, isError: pErr, isPending: pLoad } = useQuery({
    queryKey: ['projects', id],
    queryFn:  () => api.getProject(id),
    enabled:  !isNaN(id),
  })

  const { data: chapters = [], isPending: cLoad } = useQuery({
    queryKey: ['projects', id, 'chapters'],
    queryFn:  () => api.listChapters(id),
    enabled:  !isNaN(id),
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

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return chapters.filter((c) => {
      if (!matchFilter(c, filter)) return false
      if (!needle) return true
      return (
        String(c.idx).includes(needle) ||
        (c.title?.toLowerCase().includes(needle) ?? false)
      )
    })
  }, [chapters, filter, q])

  const stats        = chapterStats(chapters)
  const allChecked   = sel.size === filtered.length && filtered.length > 0
  const existingNums = useMemo(() => new Set(chapters.map((c) => c.idx)), [chapters])

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

      <div className="px-6 py-5">
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
          onRedo={() => { alert(`Dịch ${sel.size} chương`); setSel(new Set()) }}
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
