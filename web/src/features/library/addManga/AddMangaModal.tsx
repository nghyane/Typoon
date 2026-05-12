import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BookmarkPlus, Search, Link as LinkIcon, AlertTriangle,
  Globe, Loader2,
} from 'lucide-react'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { Cover } from '@shared/ui/Cover'
import { input, label, Tag } from '@shared/ui/primitives'
import { toast } from '@shared/ui/Toaster'
import { cn } from '@shared/lib/cn'
import { api, type LibraryStatus } from '@shared/api/api'
import { useEnabledSources } from '@features/browse/sources'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { proxify } from '@features/browse/proxy'
import type { InstalledSource, MangaSummary } from '@features/browse/manifest/types'
import { isUrlLike, matchSource } from './parseUrl'
import { useFanoutSearch } from './fanoutSearch'

// =============================================================================
// AddMangaModal — entry point to /library.
//
// One modal, two intents:
//
//   ① Paste URL    user has a specific manga page open elsewhere
//   ② Type name    user wants to discover something fanned-out
//
// Mode auto-detects from the input: anything starting with http(s)://
// flips to URL mode; anything else falls back to search results. The
// switch is invisible to the user — no tabs, no segment control.
//
// After picking a result, the form expands to ask for `target_lang`
// + `auto_translate`. Confirming POSTs /api/material/import then
// /api/library/entry — both calls are idempotent for the resolved
// material, so a retry after a partial failure is safe.
// =============================================================================

interface Props {
  open:    boolean
  onClose: () => void
}

interface Picked {
  source:    InstalledSource
  /** Manga URL on the upstream source (also the material upstream_ref). */
  upstreamRef: string
  /** Resolved display title from manifest. Fed to /material/import so
   *  the snapshot is right on first write. */
  title:     string
  cover:     string | null
  /** Detail fields only populated after we hit fetchMangaDetail. Until
   *  then we render the summary-level data the search row carried. */
  description: string | null
  author:      string | null
  status:      string | null
  /** Manifest-declared languages — drives the target_lang default. */
  languages:   string[]
  nsfw:        boolean
}


export function AddMangaModal({ open, onClose }: Props) {
  const sources = useEnabledSources()

  const [query, setQuery] = useState('')
  const [picked, setPicked] = useState<Picked | null>(null)

  // Status + target_lang form, only shown after Picked.
  const [targetLang, setTargetLang] = useState('vi')
  const [autoTr, setAutoTr]         = useState(false)
  const [status, setStatus]         = useState<LibraryStatus>('reading')

  useEffect(() => {
    if (!open) return
    setQuery('')
    setPicked(null)
    setTargetLang('vi')
    setAutoTr(false)
    setStatus('reading')
  }, [open])

  // Default target_lang follows the picked manga's *first* native
  // language only when it differs from VI — preserves "I want VN
  // translation" as the casual default while still being smart about
  // already-VN sources.
  useEffect(() => {
    if (!picked) return
    const native = picked.languages[0]
    if (native && native === 'vi') setTargetLang('vi')
    else setTargetLang('vi')
    // auto_translate defaults to TRUE only when target ≠ native
    setAutoTr(!(native && native === 'vi'))
  }, [picked])

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Thêm manga vào thư viện"
      size="md"
      footerLeft={picked
        ? <FooterContext picked={picked} />
        : <SearchHint query={query} sources={sources} />
      }
      footer={picked
        ? <ConfirmActions
            picked={picked}
            targetLang={targetLang}
            autoTr={autoTr}
            status={status}
            onCancel={() => setPicked(null)}
            onDone={onClose}
          />
        : <Button variant="ghost" onClick={onClose}>Huỷ</Button>
      }
    >
      <div className="px-5 py-4 space-y-4">
        {/* Input — single field, mode is implicit. */}
        <SearchOrUrlField
          query={query}
          setQuery={setQuery}
          disabled={picked !== null}
        />

        {picked ? (
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
        ) : (
          <Results
            query={query}
            sources={sources}
            onPick={setPicked}
          />
        )}
      </div>
    </Modal>
  )
}


// ── Input ────────────────────────────────────────────────────────────

function SearchOrUrlField({
  query, setQuery, disabled,
}: {
  query: string; setQuery: (s: string) => void; disabled: boolean
}) {
  const isUrl = isUrlLike(query)
  const Icon  = isUrl ? LinkIcon : Search
  return (
    <div>
      <label className={label}>
        {isUrl ? 'Đường dẫn manga' : 'Tên truyện hoặc đường dẫn'}
      </label>
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
          placeholder="https://… hoặc gõ tên truyện"
          disabled={disabled}
          className={cn(input, 'pl-9')}
        />
      </div>
    </div>
  )
}


// ── Results ──────────────────────────────────────────────────────────

function Results({
  query, sources, onPick,
}: {
  query:   string
  sources: InstalledSource[]
  onPick:  (p: Picked) => void
}) {
  // URL paste short-circuits to a single match — no search call.
  if (isUrlLike(query)) {
    return <UrlResolver url={query} sources={sources} onPick={onPick} />
  }

  return <SearchResults query={query} sources={sources} onPick={onPick} />
}


function UrlResolver({
  url, sources, onPick,
}: {
  url: string; sources: InstalledSource[]; onPick: (p: Picked) => void
}) {
  const match = useMemo(() => matchSource(url, sources), [url, sources])

  if (!match) {
    return (
      <Card>
        <CardLeft icon={AlertTriangle} tone="warning" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Chưa hỗ trợ nguồn này</p>
          <p className="text-xs text-text-subtle mt-0.5">
            Cài thêm nguồn trong Cài đặt rồi quay lại.
          </p>
        </div>
      </Card>
    )
  }

  return <UrlImportCard match={match} onPick={onPick} />
}


function UrlImportCard({
  match, onPick,
}: {
  match: ReturnType<typeof matchSource> & object
  onPick: (p: Picked) => void
}) {
  const { source, upstreamRef } = match
  const manifest = source.manifest

  // Pull the detail page so we have title/cover/description before
  // confirming. Lazy: fires only once the URL stays stable.
  const [loading, setLoading] = useState(true)
  const [err,     setErr]     = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setErr(null)
    fetchMangaDetail(manifest, upstreamRef)
      .then((d) => {
        if (cancelled) return
        onPick({
          source,
          upstreamRef,
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
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [manifest, upstreamRef, source, onPick])

  if (loading) {
    return (
      <Card>
        <CardLeft icon={Loader2} tone="neutral" spin />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Đang tải từ {manifest.name}…</p>
          <p className="text-xs text-text-subtle truncate mt-0.5">{upstreamRef}</p>
        </div>
      </Card>
    )
  }

  if (err) {
    return (
      <Card>
        <CardLeft icon={AlertTriangle} tone="error" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Không tải được</p>
          <p className="text-xs text-error-text mt-0.5 line-clamp-2">{err}</p>
        </div>
      </Card>
    )
  }
  // After onPick fires we render nothing — parent swaps to PickedDetail.
  return null
}


function SearchResults({
  query, sources, onPick,
}: {
  query:   string
  sources: InstalledSource[]
  onPick:  (p: Picked) => void
}) {
  const { hits, loading, failures, total } = useFanoutSearch(query, sources)

  if (query.trim().length < 2) {
    return (
      <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-8 text-center">
        <Search size={20} className="mx-auto text-text-subtle" />
        <p className="text-sm text-text-muted mt-2">
          Gõ tên truyện để tìm trên {total} nguồn cùng lúc
        </p>
        <p className="text-xs text-text-subtle mt-1">
          Hoặc dán đường dẫn manga để thêm trực tiếp
        </p>
      </div>
    )
  }

  if (loading && hits.length === 0) {
    return (
      <Card>
        <CardLeft icon={Loader2} tone="neutral" spin />
        <p className="text-sm text-text-muted">Đang tìm trên {total} nguồn…</p>
      </Card>
    )
  }

  if (hits.length === 0) {
    return (
      <Card>
        <CardLeft icon={Search} tone="neutral" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Không tìm thấy</p>
          <p className="text-xs text-text-subtle mt-0.5">
            Thử đổi từ khoá hoặc dán đường dẫn trực tiếp.
          </p>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-1.5">
      <p className="text-[11px] uppercase tracking-wider text-text-subtle">
        {hits.length} kết quả {loading && '· đang tìm thêm…'}
        {failures.length > 0 && ` · ${failures.length} nguồn lỗi`}
      </p>
      <ul className="rounded-md bg-surface-2 divide-y divide-border-soft overflow-hidden">
        {hits.slice(0, 30).map(({ source, manga }) => (
          <SearchResultRow
            key={`${source.manifest.id}::${manga.id}`}
            source={source}
            manga={manga}
            onPick={onPick}
          />
        ))}
      </ul>
    </div>
  )
}


function SearchResultRow({
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
        source,
        upstreamRef: manga.url,
        title:       d.title || manga.title,
        cover:       d.cover ?? manga.cover,
        description: d.description,
        author:      d.author,
        status:      d.status,
        languages:   d.availableLanguages ?? manifest.languages,
        nsfw:        !!manifest.nsfw,
      })
    } catch {
      // Fall back to summary fields when detail fetch fails — user can
      // still add the manga; refresh later picks up the rest.
      onPick({
        source,
        upstreamRef: manga.url,
        title:       manga.title,
        cover:       manga.cover,
        description: null,
        author:      null,
        status:      null,
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
          <p className="text-[11px] text-text-subtle mt-0.5 inline-flex items-center gap-1.5">
            <Globe size={10} />
            {manifest.name}
            {manifest.languages.length > 0 && (
              <span className="uppercase">
                · {manifest.languages.slice(0, 3).join('/')}
              </span>
            )}
          </p>
        </div>
        {resolving && (
          <Loader2 size={14} className="text-text-subtle animate-spin shrink-0" />
        )}
      </button>
    </li>
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
      {/* Picked-manga card */}
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
              {picked.author}
              {picked.status && ` · ${picked.status}`}
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

      {/* Form */}
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

function SearchHint({
  query, sources,
}: {
  query: string; sources: InstalledSource[]
}) {
  if (query.trim().length === 0) {
    return <>{sources.length} nguồn sẵn sàng</>
  }
  if (isUrlLike(query)) return <>Đang phân giải đường dẫn</>
  return <>Tìm trên {sources.filter((s) => s.enabled).length} nguồn</>
}


function FooterContext({ picked }: { picked: Picked }) {
  return (
    <span className="inline-flex items-center gap-2 truncate">
      <span className="truncate">{picked.title}</span>
      <Tag tone="info" size="sm">{picked.source.manifest.name}</Tag>
    </span>
  )
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
      const entry = await api.createLibraryEntry({
        material_id:    material.id,
        title:          picked.title,
        cover_url:      picked.cover ?? null,
        target_lang:    targetLang,
        auto_translate: autoTr,
        status,
      })
      return entry
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


// ── Card primitives ──────────────────────────────────────────────────

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-md bg-surface-2 border border-border-soft">
      {children}
    </div>
  )
}

function CardLeft({
  icon: Icon, tone, spin,
}: {
  icon: typeof Search
  tone: 'success' | 'warning' | 'error' | 'neutral'
  spin?: boolean
}) {
  const cls = {
    success: 'text-success-text bg-success/15',
    warning: 'text-warning-text bg-warning/15',
    error:   'text-error-text bg-error/15',
    neutral: 'text-text-muted bg-surface',
  }[tone]
  return (
    <span className={cn(
      'inline-flex items-center justify-center size-8 rounded-sm shrink-0',
      cls,
    )}>
      <Icon size={14} className={spin ? 'animate-spin' : ''} />
    </span>
  )
}
