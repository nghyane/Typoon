// ChapterList — virtualized chapter spine.
//
// Reads:
//   merged chapters    via WorkChaptersContext
//   work.target_lang   via WorkIdentityContext
//   chapterStateMap    via WorkChaptersContext (passed to rows)
//
// Local state: search query, lang filter, sort. URL-persisted later.
//
// Filter semantics (matches legacy hub):
//   null (Tự động)            → 1 row per chapter, version=best match for target
//   activeLang === target     → ditto, hide chapters with no target version
//   activeLang !== target     → 1 row per raw version of that lang

import {
  useDeferredValue, useEffect, useMemo, useRef, useState,
} from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { Languages } from 'lucide-react'

import { useWorkIdentity } from '../contexts/WorkIdentityContext'
import { useWorkChapters } from '../contexts/WorkChaptersContext'
import { pickBestVersion } from '../data/selectors/mergeChapters'
import type { MergedChapter, SourceVersion } from '../data/types'

import { ChapterToolbar, type SortBy } from '../ChapterToolbar'
import { ChapterRow } from './ChapterRow'


/** One render row — same chapter may produce multiple rows when the
 *  user filters by a non-target language. */
export interface Row {
  chapter: MergedChapter
  /** Surfaced version for this row. Null for upload-only chapters. */
  version: SourceVersion | null
  /** Stable key for the virtualizer. */
  key:     string
}


export function ChapterList() {
  const { work } = useWorkIdentity()
  const { merged, loading } = useWorkChapters()

  const tgt = work.target_lang.toLowerCase()

  // Lang counts — chapters per lang (NOT versions).
  const langCounts = useMemo(() => {
    const m = new Map<string, number>()
    for (const ch of merged) {
      const seen = new Set<string>()
      for (const v of ch.sourceVersions) {
        if (seen.has(v.lang)) continue
        seen.add(v.lang)
        m.set(v.lang, (m.get(v.lang) ?? 0) + 1)
      }
      if (ch.hasUpload && !seen.has(tgt)) {
        m.set(tgt, (m.get(tgt) ?? 0) + 1)
      }
    }
    return [...m.entries()].sort((a, b) => {
      if (a[0] === tgt) return -1
      if (b[0] === tgt) return  1
      return a[0].localeCompare(b[0])
    })
  }, [merged, tgt])

  // Lang filter — null = "Tự động" (default). Stays null on mount;
  // `pickBestVersion(targetLang)` inside the row builder already
  // surfaces the target-language version per chapter when present,
  // so auto-switching the filter would only confuse the picker UI.
  const [activeLang, setActiveLang] = useState<string | null>(null)

  const [q,      setQ]      = useState('')
  const [sortBy, setSortBy] = useState<SortBy>('newest')
  const deferredQ = useDeferredValue(q)

  // Search haystack per chapter.
  const haystacks = useMemo(() => {
    const m = new Map<string, string>()
    for (const ch of merged) {
      m.set(ch.numberNorm, `${ch.number} ${ch.label}`.toLowerCase())
    }
    return m
  }, [merged])

  const rows = useMemo<Row[]>(() => {
    const term = deferredQ.trim().toLowerCase()
    const out: Row[] = []
    const targetMode = activeLang === null || activeLang === tgt

    for (const ch of merged) {
      if (term && !(haystacks.get(ch.numberNorm) ?? '').includes(term)) continue

      if (targetMode) {
        const v = pickBestVersion(ch, tgt)
        if (activeLang !== null && !ch.hasUpload && (!v || v.lang !== tgt)) continue
        out.push({ chapter: ch, version: v, key: ch.numberNorm })
      } else {
        for (const v of ch.sourceVersions) {
          if (v.lang !== activeLang) continue
          out.push({
            chapter: ch,
            version: v,
            key:     `${ch.numberNorm}:${v.source.manifest.id}:${v.ref.id}`,
          })
        }
      }
    }

    const dir = sortBy === 'newest' ? 1 : -1
    out.sort((a, b) => (b.chapter.sortKey - a.chapter.sortKey) * dir)
    return out
  }, [merged, haystacks, deferredQ, activeLang, sortBy, tgt])

  return (
    <section className="px-4 sm:px-6">
      <ChapterToolbar
        q={q}
        setQ={setQ}
        sortBy={sortBy}
        setSortBy={setSortBy}
        langCounts={langCounts}
        activeLang={activeLang}
        setActiveLang={setActiveLang}
        targetLang={tgt}
        count={rows.length}
      />

      {loading && merged.length === 0 ? (
        <div className="py-16 flex justify-center"><Spinner size={20} /></div>
      ) : rows.length === 0 ? (
        <EmptyView
          totalChapters={merged.length}
          q={deferredQ}
          hasFilter={activeLang !== null}
        />
      ) : (
        <VirtualList rows={rows} targetLang={tgt} />
      )}
    </section>
  )
}


// ── Virtual list ────────────────────────────────────────────────


function useRowHeight() {
  const [wide, setWide] = useState(() =>
    typeof window !== 'undefined' &&
    window.matchMedia('(min-width: 640px)').matches,
  )
  useEffect(() => {
    if (typeof window === 'undefined') return
    const mq = window.matchMedia('(min-width: 640px)')
    const onChange = () => setWide(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return wide ? 44 : 52
}


function VirtualList({ rows, targetLang }: { rows: Row[]; targetLang: string }) {
  const rowHeight = useRowHeight()
  const parentRef = useRef<HTMLDivElement>(null)
  const [scrollEl,     setScrollEl]     = useState<HTMLElement | null>(null)
  const [scrollMargin, setScrollMargin] = useState(0)

  useEffect(() => {
    let el: HTMLElement | null = parentRef.current
    while (el && el !== document.body) {
      const overflowY = getComputedStyle(el).overflowY
      if (overflowY === 'auto' || overflowY === 'scroll') {
        setScrollEl(el); break
      }
      el = el.parentElement
    }
    if (!el || el === document.body) {
      setScrollEl((document.scrollingElement as HTMLElement) ?? null)
    }
  }, [])

  useEffect(() => {
    if (!scrollEl || !parentRef.current) return
    const measure = () => {
      if (!parentRef.current || !scrollEl) return
      const listTop   = parentRef.current.getBoundingClientRect().top
      const scrollTop = scrollEl.getBoundingClientRect().top
      setScrollMargin(listTop - scrollTop + (scrollEl.scrollTop ?? 0))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(scrollEl)
    if (parentRef.current.parentElement) ro.observe(parentRef.current.parentElement)
    return () => ro.disconnect()
  }, [scrollEl])

  const virt = useVirtualizer({
    count:            rows.length,
    getScrollElement: () => scrollEl,
    estimateSize:     () => rowHeight,
    overscan:         8,
    getItemKey:       (i) => rows[i]!.key,
    scrollMargin,
  })

  return (
    <div
      ref={parentRef}
      className="relative w-full"
      style={{ height: virt.getTotalSize() || undefined }}
    >
      {virt.getVirtualItems().map((vi) => {
        const r = rows[vi.index]!
        return (
          <div
            key={vi.key}
            data-index={vi.index}
            ref={virt.measureElement}
            className="absolute left-0 right-0 border-b border-border-soft/60"
            style={{ transform: `translateY(${vi.start - virt.options.scrollMargin}px)` }}
          >
            <ChapterRow row={r} targetLang={targetLang} />
          </div>
        )
      })}
    </div>
  )
}


// ── Empty view ──────────────────────────────────────────────────


function EmptyView({
  totalChapters, q, hasFilter,
}: {
  totalChapters: number
  q:             string
  hasFilter:     boolean
}) {
  if (totalChapters === 0) {
    return <EmptyState icon={Languages} title="Chưa có chương"
      hint="Liên kết một nguồn hoặc tải lên file zip/cbz." />
  }
  if (q.trim()) {
    return <EmptyState title="Không khớp tìm kiếm"
      hint={`Không có chương nào khớp "${q.trim()}".`} />
  }
  if (hasFilter) {
    return <EmptyState title="Không có chương ở ngôn ngữ này"
      hint="Chuyển sang 'Tự động' để xem mọi ngôn ngữ." />
  }
  return null
}
