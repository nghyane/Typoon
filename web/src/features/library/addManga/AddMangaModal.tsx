import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BookmarkPlus, Search, Link as LinkIcon, AlertTriangle,
  Loader2, Wand2, CheckCircle2,
} from 'lucide-react'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { Cover } from '@shared/ui/Cover'
import { input, label, Tag } from '@shared/ui/primitives'
import { toast } from '@shared/ui/Toaster'
import { cn } from '@shared/lib/cn'
import { api, type LibraryStatus } from '@shared/api/api'
import { useEnabledSources } from '@features/browse/sources'
import { fetchMangaDetail, hasSearch } from '@features/browse/manifest/runtime'
import { proxify } from '@features/browse/proxy'
import type { InstalledSource, MangaSummary } from '@features/browse/manifest/types'
import { isUrlLike, matchSource } from './parseUrl'
import { useFanoutSearch, type SearchHit } from './fanoutSearch'
import { ManualCreateForm } from './ManualCreateForm'
import { Favicon } from './SourceBits'
import { SourceSidebar } from './SourceSidebar'

// =============================================================================
// AddMangaModal — Library entry point.
//
// Three modes, one modal:
//
//   ① http(s)://…   URL paste. Match against manifest.host across all
//                   enabled sources. Unsupported host → manual create.
//   ② "naruto"      Search. Sidebar scopes to one source or fanout to
//                   every searchable manifest. Results group by source.
//   ③ empty input   Hint card explaining the scope.
//
// Layout: search mode uses a 2-col split (sidebar + body); picked /
// manual modes drop the sidebar for a wider form.
// =============================================================================

interface Props {
  open:    boolean
  onClose: () => void
}

interface Picked {
  source:      InstalledSource
  upstreamRef: string
  title:       string
  cover:       string | null
  description: string | null
  author:      string | null
  status:      string | null
  languages:   string[]
  nsfw:        boolean
}

type Mode = 'search' | 'picked' | 'manual'


export function AddMangaModal({ open, onClose }: Props) {
  const allSources = useEnabledSources()
  const searchableIds = useMemo(
    () => new Set(allSources.filter((s) => hasSearch(s.manifest))
                            .map((s) => s.manifest.id)),
    [allSources],
  )

  const [query,            setQuery]            = useState('')
  const [picked,           setPicked]           = useState<Picked | null>(null)
  const [manualSeed,       setManualSeed]       = useState<string | null>(null)
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null)

  const [targetLang, setTargetLang] = useState('vi')
  const [autoTr,     setAutoTr]     = useState(false)
  const [status,     setStatus]     = useState<LibraryStatus>('reading')

  const urlMatch = useMemo(
    () => isUrlLike(query) ? matchSource(query, allSources) : null,
    [query, allSources],
  )
  const lockedSourceId = urlMatch?.source.manifest.id ?? null

  useEffect(() => {
    if (!open) return
    setQuery('')
    setPicked(null)
    setManualSeed(null)
    setSelectedSourceId(null)
    setTargetLang('vi')
    setAutoTr(false)
    setStatus('reading')
  }, [open])

  useEffect(() => {
    if (!picked) return
    const native = picked.languages[0]
    setTargetLang('vi')
    setAutoTr(!(native && native === 'vi'))
  }, [picked])

  const mode: Mode = manualSeed !== null ? 'manual'
                   : picked !== null     ? 'picked'
                                          : 'search'

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Thêm manga vào thư viện"
      size={mode === 'search' ? 'lg' : 'md'}
      footerLeft={<FooterLeft
        mode={mode}
        picked={picked}
        sources={allSources}
        searchableIds={searchableIds}
      />}
      footer={mode === 'picked' && picked ? (
        <ConfirmActions
          picked={picked}
          targetLang={targetLang}
          autoTr={autoTr}
          status={status}
          onCancel={() => setPicked(null)}
          onDone={onClose}
        />
      ) : (
        <Button variant="ghost" onClick={onClose}>Huỷ</Button>
      )}
    >
      {mode === 'manual' ? (
        <div className="px-5 py-4">
          <ManualCreateForm
            initialTitle={manualSeed ?? ''}
            onCancel={() => setManualSeed(null)}
            onCreated={onClose}
          />
        </div>
      ) : mode === 'picked' && picked ? (
        <div className="px-5 py-4">
          <PickedDetail
            picked={picked}
            targetLang={targetLang}
            setTargetLang={setTargetLang}
            autoTr={autoTr}
            setAutoTr={setAutoTr}
            status={status}
            setStatus={setStatus}
            onChangePick={() => setPicked(null)}
          />
        </div>
      ) : (
        <SearchSplit
          query={query}
          setQuery={setQuery}
          sources={allSources}
          searchableIds={searchableIds}
          selectedSourceId={selectedSourceId}
          setSelectedSourceId={setSelectedSourceId}
          lockedSourceId={lockedSourceId}
          urlMatch={urlMatch}
          onPick={setPicked}
          onManualCreate={(seed) => setManualSeed(seed)}
        />
      )}
    </Modal>
  )
}


// ── Search split layout ─────────────────────────────────────────────

function SearchSplit({
  query, setQuery, sources, searchableIds,
  selectedSourceId, setSelectedSourceId,
  lockedSourceId, urlMatch,
  onPick, onManualCreate,
}: {
  query:                string
  setQuery:             (s: string) => void
  sources:              InstalledSource[]
  searchableIds:        Set<string>
  selectedSourceId:     string | null
  setSelectedSourceId:  (id: string | null) => void
  lockedSourceId:       string | null
  urlMatch:             ReturnType<typeof matchSource>
  onPick:               (p: Picked) => void
  onManualCreate:       (seed: string) => void
}) {
  const isUrl     = isUrlLike(query)
  const effective = lockedSourceId ?? selectedSourceId

  const scopedSources = useMemo(() => {
    if (effective === null) {
      return sources.filter((s) => searchableIds.has(s.manifest.id))
    }
    return sources.filter((s) => s.manifest.id === effective)
  }, [sources, searchableIds, effective])

  return (
    <div className="flex min-h-[420px] max-h-[60vh]">
      <SourceSidebar
        sources={sources}
        searchableIds={searchableIds}
        value={selectedSourceId}
        onChange={setSelectedSourceId}
        lockedSourceId={lockedSourceId}
      />

      <section className="flex-1 min-w-0 flex flex-col overflow-hidden">
        <div className="px-5 pt-4 pb-3 border-b border-border-soft shrink-0">
          <InputRow
            query={query}
            setQuery={setQuery}
            isUrl={isUrl}
            urlMatch={urlMatch}
          />
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {isUrl ? (
            urlMatch
              ? <UrlImportCard match={urlMatch} onPick={onPick} />
              : <UnsupportedUrlCard url={query} onManualCreate={onManualCreate} />
          ) : query.trim().length < 2 ? (
            <ScopeHint
              selectedSourceId={effective}
              sources={sources}
              searchableCount={searchableIds.size}
            />
          ) : (
            <Results
              query={query}
              searchableSources={scopedSources}
              onPick={onPick}
              onManualCreate={onManualCreate}
            />
          )}
        </div>
      </section>
    </div>
  )
}


// ── Input + URL badge ───────────────────────────────────────────────

function InputRow({
  query, setQuery, isUrl, urlMatch,
}: {
  query:    string
  setQuery: (s: string) => void
  isUrl:    boolean
  urlMatch: ReturnType<typeof matchSource>
}) {
  const Icon = isUrl ? LinkIcon : Search
  return (
    <div className="relative">
      <Icon
        size={14}
        className="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none"
      />
      <input
        autoFocus
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Tìm tên truyện hoặc dán đường dẫn manga"
        className={cn(input, 'pl-9 h-10', isUrl && (urlMatch ? 'pr-36' : 'pr-32'))}
      />
      {isUrl && <UrlBadge urlMatch={urlMatch} />}
    </div>
  )
}


function UrlBadge({
  urlMatch,
}: {
  urlMatch: ReturnType<typeof matchSource>
}) {
  const cls =
    'absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center gap-1 ' +
    'h-6 px-2 rounded-xs text-[11px] font-medium pointer-events-none'
  if (urlMatch) {
    return (
      <span className={cn(cls, 'bg-success/15 text-success-text')}>
        <CheckCircle2 size={10} />
        {urlMatch.source.manifest.name}
      </span>
    )
  }
  return (
    <span className={cn(cls, 'bg-warning/15 text-warning-text')}>
      <AlertTriangle size={10} />
      Chưa hỗ trợ
    </span>
  )
}


// ── Empty-state hint ────────────────────────────────────────────────

function ScopeHint({
  selectedSourceId, sources, searchableCount,
}: {
  selectedSourceId: string | null
  sources:          InstalledSource[]
  searchableCount:  number
}) {
  const sel = selectedSourceId
    ? sources.find((s) => s.manifest.id === selectedSourceId)
    : null
  return (
    <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-5 py-8">
      <div className="flex items-center gap-3">
        <span className="size-10 rounded-sm bg-bg/40 flex items-center justify-center shrink-0">
          {sel
            ? <Favicon host={sel.manifest.host} size={20} />
            : <Search size={18} className="text-text-subtle" />
          }
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">
            {sel
              ? `Tìm trên ${sel.manifest.name}`
              : `Tìm trên ${searchableCount} nguồn cùng lúc`
            }
          </p>
          <p className="text-[11px] text-text-subtle mt-1">
            Hoặc dán đường dẫn manga vào ô trên để thêm trực tiếp.
          </p>
        </div>
      </div>
    </div>
  )
}


// ── URL paste flows ─────────────────────────────────────────────────

function UnsupportedUrlCard({
  url, onManualCreate,
}: {
  url: string; onManualCreate: (seed: string) => void
}) {
  return (
    <div className="rounded-md bg-warning/10 border border-warning/20 px-4 py-3">
      <div className="flex items-start gap-2.5">
        <AlertTriangle size={14} className="text-warning-text shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Không có nguồn quản lý site này</p>
          <p className="text-[11px] text-text-subtle mt-0.5 break-all line-clamp-2">
            {url}
          </p>
          <button
            type="button"
            onClick={() => onManualCreate('')}
            className="mt-2.5 inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm bg-surface-2 text-[12px] text-text hover:bg-hover cursor-pointer transition-colors"
          >
            <Wand2 size={12} />
            Tạo thủ công thay
          </button>
        </div>
      </div>
    </div>
  )
}


function UrlImportCard({
  match, onPick,
}: {
  match: NonNullable<ReturnType<typeof matchSource>>
  onPick: (p: Picked) => void
}) {
  const { source, upstreamRef } = match
  const manifest = source.manifest
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setErr(null)
    fetchMangaDetail(manifest, upstreamRef)
      .then((d) => {
        if (cancelled) return
        onPick({
          source, upstreamRef,
          title:       d.title,
          cover:       d.cover,
          description: d.description,
          author:      d.author,
          status:      d.status,
          languages:   d.availableLanguages ?? manifest.languages,
          nsfw:        !!manifest.nsfw,
        })
      })
      .catch((e) => {
        if (cancelled) return
        setErr(e instanceof Error ? e.message : 'Không tải được trang truyện')
      })
    return () => { cancelled = true }
  }, [manifest, upstreamRef, source, onPick])

  if (err) {
    return (
      <div className="rounded-md bg-error/10 border border-error/20 px-4 py-3">
        <div className="flex items-start gap-2.5">
          <AlertTriangle size={14} className="text-error-text shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-text">Không tải được từ {manifest.name}</p>
            <p className="text-[11px] text-error-text mt-0.5 line-clamp-2">{err}</p>
          </div>
        </div>
      </div>
    )
  }
  return (
    <div className="rounded-md bg-surface-2 px-4 py-3 flex items-center gap-2.5">
      <Loader2 size={14} className="text-info-text animate-spin shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text">Đang tải từ {manifest.name}…</p>
        <p className="text-[11px] text-text-subtle truncate mt-0.5">{upstreamRef}</p>
      </div>
    </div>
  )
}


// ── Results — fanout grouped by source ──────────────────────────────

function Results({
  query, searchableSources, onPick, onManualCreate,
}: {
  query:             string
  searchableSources: InstalledSource[]
  onPick:            (p: Picked) => void
  onManualCreate:    (seed: string) => void
}) {
  const { hits, loading, failures } = useFanoutSearch(query, searchableSources)

  const groups = useMemo(() => {
    const by: Record<string, { source: InstalledSource; hits: SearchHit[] }> = {}
    for (const h of hits) {
      const id = h.source.manifest.id
      if (!by[id]) by[id] = { source: h.source, hits: [] }
      by[id]!.hits.push(h)
    }
    return searchableSources
      .map((s) => by[s.manifest.id])
      .filter((g): g is { source: InstalledSource; hits: SearchHit[] } => !!g)
  }, [hits, searchableSources])

  const singleSource = searchableSources.length === 1

  if (loading && hits.length === 0) {
    return (
      <div className="flex items-center gap-2.5 px-4 py-3 rounded-md bg-surface-2">
        <Loader2 size={14} className="text-info-text animate-spin shrink-0" />
        <p className="text-sm text-text-muted">
          Đang tìm trên {searchableSources.length} nguồn…
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {groups.length > 0 && (
        <div className="flex items-center justify-between gap-2 px-0.5">
          <p className="text-[11px] uppercase tracking-wider text-text-subtle">
            {hits.length} kết quả
            {loading && <span className="ml-1.5 normal-case">· đang tìm thêm…</span>}
          </p>
          {failures.length > 0 && (
            <span className="text-[11px] text-warning-text inline-flex items-center gap-1">
              <AlertTriangle size={10} />
              {failures.length} nguồn lỗi
            </span>
          )}
        </div>
      )}

      {groups.map(({ source, hits: groupHits }) => (
        <SourceGroup
          key={source.manifest.id}
          source={source}
          hits={groupHits}
          onPick={onPick}
          hideHeader={singleSource}
        />
      ))}

      <ManualCreateRow
        query={query}
        hits={hits.length}
        onManualCreate={onManualCreate}
      />
    </div>
  )
}


function SourceGroup({
  source, hits, onPick, hideHeader,
}: {
  source:     InstalledSource
  hits:       SearchHit[]
  onPick:     (p: Picked) => void
  hideHeader: boolean
}) {
  const manifest = source.manifest
  return (
    <section>
      {!hideHeader && (
        <header className="flex items-center gap-2 px-1 mb-1.5">
          <Favicon host={manifest.host} size={14} />
          <span className="text-[12px] font-medium text-text">
            {manifest.name}
          </span>
          <span className="text-[11px] text-text-subtle">{hits.length}</span>
        </header>
      )}
      <ul className="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
        {hits.slice(0, 8).map(({ manga }) => (
          <ResultRow
            key={`${manifest.id}::${manga.id}`}
            source={source}
            manga={manga}
            onPick={onPick}
          />
        ))}
      </ul>
    </section>
  )
}


function ResultRow({
  source, manga, onPick,
}: {
  source: InstalledSource; manga: MangaSummary; onPick: (p: Picked) => void
}) {
  const manifest = source.manifest
  const [resolving, setResolving] = useState(false)

  const pick = async () => {
    setResolving(true)
    try {
      const d = await fetchMangaDetail(manifest, manga.url)
      onPick({
        source, upstreamRef: manga.url,
        title:       d.title || manga.title,
        cover:       d.cover ?? manga.cover,
        description: d.description,
        author:      d.author,
        status:      d.status,
        languages:   d.availableLanguages ?? manifest.languages,
        nsfw:        !!manifest.nsfw,
      })
    } catch {
      onPick({
        source, upstreamRef: manga.url,
        title:       manga.title,
        cover:       manga.cover,
        description: null, author: null, status: null,
        languages:   manifest.languages,
        nsfw:        !!manifest.nsfw,
      })
    } finally {
      setResolving(false)
    }
  }

  return (
    <li>
      <button
        type="button"
        onClick={pick}
        disabled={resolving}
        className={cn(
          'w-full flex items-center gap-3 px-3 py-2 text-left',
          'hover:bg-hover transition-colors cursor-pointer',
          resolving && 'opacity-60 cursor-wait',
        )}
      >
        <Cover
          src={manga.cover ? proxify(manga.cover) : null}
          title={manga.title}
          className="w-10 aspect-[2/3] rounded-xs shrink-0"
          fontSize="text-[10px]"
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text truncate">{manga.title}</p>
          {manifest.languages.length > 0 && (
            <p className="text-[11px] text-text-subtle mt-0.5 uppercase">
              {manifest.languages.slice(0, 3).join('/')}
            </p>
          )}
        </div>
        {resolving && (
          <Loader2 size={14} className="text-text-subtle animate-spin shrink-0" />
        )}
      </button>
    </li>
  )
}


function ManualCreateRow({
  query, hits, onManualCreate,
}: {
  query: string; hits: number; onManualCreate: (seed: string) => void
}) {
  const seed = query.trim()
  return (
    <button
      type="button"
      onClick={() => onManualCreate(seed)}
      className={cn(
        'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-md',
        'text-left transition-colors cursor-pointer',
        hits === 0
          ? 'bg-accent/10 border border-accent/20 hover:bg-accent/15'
          : 'bg-surface-2 hover:bg-hover',
      )}
    >
      <span className={cn(
        'inline-flex items-center justify-center size-8 rounded-sm shrink-0',
        hits === 0 ? 'bg-accent text-accent-fg' : 'bg-surface text-text-muted',
      )}>
        <Wand2 size={13} />
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text">
          {hits === 0
            ? `Không tìm thấy. Tạo "${seed}" thủ công?`
            : `Không thấy "${seed}"? Tạo thủ công`
          }
        </p>
        <p className="text-[11px] text-text-subtle mt-0.5">
          Manga không thuộc nguồn nào · tải chương từ file zip/cbz
        </p>
      </div>
    </button>
  )
}


// ── Picked → confirm form ────────────────────────────────────────────

function PickedDetail({
  picked, targetLang, setTargetLang,
  autoTr, setAutoTr, status, setStatus,
  onChangePick,
}: {
  picked:        Picked
  targetLang:    string
  setTargetLang: (s: string) => void
  autoTr:        boolean
  setAutoTr:     (b: boolean) => void
  status:        LibraryStatus
  setStatus:     (s: LibraryStatus) => void
  onChangePick:  () => void
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 p-3 rounded-md bg-surface-2">
        <Cover
          src={picked.cover ? proxify(picked.cover) : null}
          title={picked.title}
          className="w-14 aspect-[2/3] rounded-xs shrink-0"
          fontSize="text-xs"
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-text line-clamp-2">{picked.title}</p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <Tag tone="info" size="sm">{picked.source.manifest.name}</Tag>
            {picked.languages[0] && (
              <Tag tone="neutral" size="sm" uppercase>
                {picked.languages.slice(0, 3).join('/')}
              </Tag>
            )}
            {picked.nsfw && <Tag tone="error" size="sm" uppercase>NSFW</Tag>}
          </div>
          {picked.author && (
            <p className="text-[11px] text-text-subtle mt-1.5 truncate">
              {picked.author}{picked.status && ` · ${picked.status}`}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onChangePick}
          className="text-xs text-text-subtle hover:text-text shrink-0 underline-offset-2 hover:underline cursor-pointer"
        >
          Đổi
        </button>
      </div>

      <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-3">
        <div>
          <label className={label}>Đọc bằng</label>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            className={input}
          >
            <option value="vi">Tiếng Việt</option>
            <option value="en">English</option>
            <option value="ja">日本語</option>
            <option value="ko">한국어</option>
            <option value="zh">中文</option>
          </select>
        </div>
        <div>
          <label className={label}>Tình trạng</label>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as LibraryStatus)}
            className={input}
          >
            <option value="reading">Đang đọc</option>
            <option value="plan">Kế hoạch</option>
            <option value="on_hold">Tạm dừng</option>
            <option value="done">Đã xong</option>
          </select>
        </div>

        <div className="col-span-2 flex items-center gap-2.5">
          <input
            id="auto-translate"
            type="checkbox"
            checked={autoTr}
            onChange={(e) => setAutoTr(e.target.checked)}
            className="size-4 cursor-pointer accent-accent"
          />
          <label
            htmlFor="auto-translate"
            className="text-sm text-text-muted cursor-pointer select-none"
          >
            Tự động dịch chương mới sang {targetLang.toUpperCase()}
            <span className="text-[11px] text-text-subtle ml-1">
              · tốn quota dịch cho mỗi chương mới
            </span>
          </label>
        </div>
      </div>
    </div>
  )
}


// ── Footer ───────────────────────────────────────────────────────────

function FooterLeft({
  mode, picked, sources, searchableIds,
}: {
  mode:          Mode
  picked:        Picked | null
  sources:       InstalledSource[]
  searchableIds: Set<string>
}) {
  if (mode === 'manual') return <span>Tạo manga không thuộc nguồn nào</span>
  if (mode === 'picked' && picked) {
    return (
      <span className="inline-flex items-center gap-2 truncate">
        <span className="truncate">{picked.title}</span>
        <Tag tone="info" size="sm">{picked.source.manifest.name}</Tag>
      </span>
    )
  }
  if (sources.length !== searchableIds.size) {
    return (
      <span>
        {sources.length} nguồn đã cài
        <span className="text-text-subtle"> · {searchableIds.size} hỗ trợ tìm</span>
      </span>
    )
  }
  return <span>{sources.length} nguồn đã cài</span>
}


function ConfirmActions({
  picked, targetLang, autoTr, status, onCancel, onDone,
}: {
  picked:     Picked
  targetLang: string
  autoTr:     boolean
  status:     LibraryStatus
  onCancel:   () => void
  onDone:     () => void
}) {
  const qc = useQueryClient()
  const m = useMutation({
    mutationFn: async () => {
      const material = await api.importMaterial({
        source:       picked.source.manifest.id,
        upstream_ref: picked.upstreamRef,
        title:        picked.title,
        cover_url:    picked.cover,
        description:  picked.description,
        author:       picked.author,
        status:       picked.status,
        languages:    picked.languages,
        nsfw:         picked.nsfw,
      })
      return await api.createLibraryEntry({
        material_id:    material.id,
        title:          picked.title,
        cover_url:      picked.cover ?? null,
        target_lang:    targetLang,
        auto_translate: autoTr,
        status,
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['library'] })
      toast.success(`Đã thêm "${picked.title}" vào thư viện`)
      onDone()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <>
      <Button variant="ghost" onClick={onCancel} disabled={m.isPending}>
        Quay lại
      </Button>
      <Button
        variant="primary"
        onClick={() => m.mutate()}
        disabled={m.isPending}
      >
        <BookmarkPlus size={14} />
        Thêm vào thư viện
      </Button>
    </>
  )
}
