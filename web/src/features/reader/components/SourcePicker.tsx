// SourcePicker — pick a source for the current chapter (3-section
// layout tailored to the typical Vietnamese-manga workflow).
//
//   ┌────────────────────────────────────────────────┐
//   │ Bản dịch của bạn                               │
//   │   ✨ Đã dịch · 2 ngày trước     [📥 offline]   │
//   │   ⏳ Đang dịch                                  │
//   │                                                │
//   │ Đọc trực tiếp (VI)                             │
//   │   VI · TruyenQQ                       2h trước │
//   │   VI · NetTruyen                      1d trước │
//   │                                                │
//   │ Bản gốc — sẵn sàng để dịch                     │
//   │   EN · Mangadex                  5h  [Dịch →]  │
//   │   JA · Niadd                     3d  [Dịch →]  │
//   └────────────────────────────────────────────────┘
//
// Three sections express the three real intents:
//   1. "I want my own AI translation"     → translated
//   2. "Read what scanlators have done"   → raw in target lang
//   3. "Translate this raw with Typoon"   → raw in other lang
//
// State surface per row:
//   - active pill / accent strip
//   - 📥 saved offline indicator
//   - job state ("Đang dịch") for the translated section
//   - inline "Dịch →" CTA for non-target-lang raws

import { useMemo, useState } from 'react'
import { Sparkles, Download, Loader2, AlertCircle, ArrowRight } from 'lucide-react'

import { BottomSheet } from '@shared/ui/BottomSheet'
import { Popover } from '@shared/ui/Popover'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import { toast } from '@shared/ui/Toaster'
import { useIsDesktop } from '@shared/lib/useMediaQuery'

import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'
import { useWorkIdentity } from '@features/work/contexts/WorkIdentityContext'
import { useWorkActions } from '@features/work/contexts/WorkActionsContext'
import { deriveChapterState } from '@features/work/data/selectors/chapterState'
import type { SourceVersion } from '@features/work/data/types'

import { useReader } from '../ReaderContext'
import { useSourcePref, useSetSourcePref } from '../hooks/useSourcePref'
import { useChapterSources } from '../data/queries/useChapterSources'
import { versionKeyOf } from '../data/selectors/resolveSource'


interface Props {
  open:    boolean
  onClose: () => void
  anchorRef: React.RefObject<HTMLButtonElement | null>
}


export function SourcePicker({ open, onClose, anchorRef }: Props) {
  const isDesktop = useIsDesktop()

  if (isDesktop) {
    return (
      <Popover
        open={open}
        onClose={onClose}
        anchorRef={anchorRef}
        align="end"
        minWidth={340}
        maxWidth={400}
      >
        <Body onClose={onClose} />
      </Popover>
    )
  }
  return (
    <BottomSheet open={open} onClose={onClose} title="Nguồn đọc">
      <Body onClose={onClose} />
    </BottomSheet>
  )
}


// ── Body ───────────────────────────────────────────────────────


function Body({ onClose }: { onClose: () => void }) {
  const { workId, chapterRef } = useReader()
  const { work }   = useWorkIdentity()
  const { merged } = useWorkChapters()
  const pref       = useSourcePref(workId)
  const setPref    = useSetSourcePref(workId)
  const sources    = useChapterSources(workId, chapterRef)
  const actions    = useWorkActions()

  const tgt = work.target_lang.toLowerCase()
  const chapter = merged.find(c => c.numberNorm === chapterRef) ?? null
  const versions = chapter?.sourceVersions ?? []

  // Bucket the raw versions by language match.
  const { targetLangVersions, otherLangVersions } = useMemo(() => {
    const target: SourceVersion[] = []
    const other:  SourceVersion[] = []
    for (const v of versions) {
      if (v.lang === tgt) target.push(v)
      else                other.push(v)
    }
    const sortByDate = (a: SourceVersion, b: SourceVersion) => {
      const ad = a.ref.date ?? ''
      const bd = b.ref.date ?? ''
      return bd.localeCompare(ad)
    }
    return {
      targetLangVersions: target.sort(sortByDate),
      otherLangVersions:  other.sort(sortByDate),
    }
  }, [versions, tgt])

  // Translated section state.
  const chapterState = useMemo(
    () => deriveChapterState(sources.saved, sources.job),
    [sources.saved, sources.job],
  )
  const hasTranslated =
    chapterState.status === 'saved-translated' ||
    chapterState.status === 'done-online' ||
    chapterState.status === 'done-expired' ||
    chapterState.status === 'running'

  // Which row reads as "active right now"?
  //   - pref.translated → translated section
  //   - pref.raw + matching version → that row
  //   - pref.auto → whatever the resolver would pick (translated if
  //     available, else first target-lang raw, else first other-lang
  //     raw)
  const activeKey: string = useMemo(() => {
    if (pref.kind === 'translated') return 'translated'
    if (pref.kind === 'raw')        return pref.versionKey
    // auto:
    if (hasTranslated) return 'translated'
    if (targetLangVersions[0]) return versionKeyOf(targetLangVersions[0])
    if (otherLangVersions[0])  return versionKeyOf(otherLangVersions[0])
    return ''
  }, [pref, hasTranslated, targetLangVersions, otherLangVersions])

  function pickTranslated() {
    setPref({ kind: 'translated' })
    onClose()
  }

  function pickRaw(versionKey: string) {
    setPref({ kind: 'raw', versionKey })
    onClose()
  }

  const showTranslatedSection = hasTranslated
  const showTargetSection     = targetLangVersions.length > 0
  const showOtherSection      = otherLangVersions.length > 0

  if (!showTranslatedSection && !showTargetSection && !showOtherSection) {
    return (
      <p className="px-4 py-6 text-sm text-text-subtle text-center">
        Chương này chưa có nguồn nào.
      </p>
    )
  }

  return (
    <div className="flex flex-col max-h-[70dvh] sm:max-h-[70vh]">
      <div className="flex-1 overflow-y-auto p-2 space-y-3">
        {showTranslatedSection && (
          <Section title="Bản dịch của bạn">
            <TranslatedRow
              state={chapterState}
              active={activeKey === 'translated'}
              onClick={pickTranslated}
            />
          </Section>
        )}

        {showTargetSection && (
          <Section
            title={`Đọc trực tiếp (${work.target_lang.toUpperCase()})`}
            hint={showTranslatedSection ? undefined : 'Bản scanlator có sẵn'}
          >
            {targetLangVersions.map(v => {
              const vk = versionKeyOf(v)
              return (
                <RawRow
                  key={vk}
                  version={v}
                  active={activeKey === vk}
                  onPick={() => pickRaw(vk)}
                />
              )
            })}
          </Section>
        )}

        {showOtherSection && (
          <Section
            title="Bản gốc — sẵn sàng để dịch"
            hint={showTargetSection ? undefined : 'Bấm “Dịch” để tạo bản tiếng Việt'}
          >
            {otherLangVersions.map(v => {
              const vk = versionKeyOf(v)
              return (
                <RawRow
                  key={vk}
                  version={v}
                  active={activeKey === vk}
                  onPick={() => pickRaw(vk)}
                  onTranslate={async () => {
                    try {
                      await actions.spawnTranslate(chapterRef, v)
                      toast.success('Đã bắt đầu dịch chương.')
                      setPref({ kind: 'translated' })
                      onClose()
                    } catch (e) {
                      toast.error((e as Error).message || 'Không tạo được job dịch.')
                    }
                  }}
                />
              )
            })}
          </Section>
        )}
      </div>
    </div>
  )
}


// ── Section ────────────────────────────────────────────────────


function Section({
  title, hint, children,
}: {
  title:    string
  hint?:    string
  children: React.ReactNode
}) {
  return (
    <section className="space-y-1">
      <div className="px-3 pt-1 flex items-baseline justify-between gap-2">
        <h3 className="text-xs uppercase tracking-wider text-text-subtle font-semibold">
          {title}
        </h3>
        {hint && (
          <span className="text-xs text-text-subtle">{hint}</span>
        )}
      </div>
      <div>{children}</div>
    </section>
  )
}


// ── Translated row ─────────────────────────────────────────────


function TranslatedRow({
  state, active, onClick,
}: {
  state:   ReturnType<typeof deriveChapterState>
  active:  boolean
  onClick: () => void
}) {
  const meta = describeTranslatedState(state)
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'relative w-full flex items-center gap-3 h-12 px-3 rounded-sm text-left',
        'transition-colors duration-150 cursor-pointer',
        active
          ? 'bg-accent-bg'
          : 'hover:bg-hover',
      )}
    >
      {active && (
        <span
          aria-hidden
          className="absolute left-0 top-2 bottom-2 w-[2px] rounded-full bg-accent"
        />
      )}
      <Sparkles size={16} className={active ? 'text-accent shrink-0' : 'text-accent-text shrink-0'} />
      <span className="flex-1 min-w-0">
        <span className={cn(
          'block text-sm truncate',
          active ? 'text-accent-text font-medium' : 'text-text',
        )}>
          Bản dịch của bạn
        </span>
        <span className="block text-xs text-text-subtle truncate">
          {meta.label}
        </span>
      </span>
      {meta.icon}
    </button>
  )
}


function describeTranslatedState(
  state: ReturnType<typeof deriveChapterState>,
): { label: string; icon: React.ReactNode } {
  switch (state.status) {
    case 'saved-translated':
      return {
        label: state.archive?.saved_at
          ? `Đã dịch · ${timeAgo(state.archive.saved_at)}`
          : 'Đã dịch · đã lưu offline',
        icon: <Download size={13} className="text-success shrink-0" />,
      }
    case 'done-online':
      return {
        label: state.job?.created_at
          ? `Đã dịch · ${timeAgo(state.job.created_at)}`
          : 'Đã dịch',
        icon: null,
      }
    case 'done-expired':
      return {
        label: 'Đã dịch · link hết hạn — sẽ làm mới',
        icon: <AlertCircle size={13} className="text-warning shrink-0" />,
      }
    case 'running':
      return {
        label: 'Đang dịch…',
        icon: <Loader2 size={13} className="text-info animate-spin shrink-0" />,
      }
    default:
      return { label: '', icon: null }
  }
}


// ── Raw row ────────────────────────────────────────────────────


function RawRow({
  version: v, active, onPick, onTranslate,
}: {
  version:     SourceVersion
  active:      boolean
  onPick:      () => void
  /** Optional inline CTA — present only for non-target-lang raws. */
  onTranslate?: () => Promise<void> | void
}) {
  const [busy, setBusy] = useState(false)

  async function handleTranslate(e: React.MouseEvent) {
    e.stopPropagation()
    if (!onTranslate || busy) return
    setBusy(true)
    try { await onTranslate() }
    finally { setBusy(false) }
  }

  return (
    <div
      className={cn(
        'relative flex items-center gap-2 h-10 pl-3 pr-2 rounded-sm',
        'transition-colors duration-150',
        active
          ? 'bg-accent-bg'
          : 'hover:bg-hover',
      )}
    >
      {active && (
        <span
          aria-hidden
          className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent"
        />
      )}
      <button
        type="button"
        onClick={onPick}
        aria-pressed={active}
        className="flex-1 min-w-0 flex items-center gap-3 text-left cursor-pointer"
      >
        <span className={cn(
          'shrink-0 inline-flex items-center justify-center min-w-[2rem]',
          'text-xs uppercase font-semibold tabular-nums',
          active ? 'text-accent' : 'text-text-subtle',
        )}>
          {v.lang.toUpperCase()}
        </span>
        <span className={cn(
          'flex-1 min-w-0 truncate text-sm',
          active ? 'text-accent-text font-medium' : 'text-text',
        )}>
          {v.source.manifest.name}
          {v.ref.scanlator && (
            <span className="text-text-subtle"> · @{v.ref.scanlator}</span>
          )}
        </span>
        {v.ref.date && (
          <span className="text-xs text-text-subtle tabular-nums shrink-0">
            {timeAgo(v.ref.date)}
          </span>
        )}
      </button>
      {onTranslate && (
        <button
          type="button"
          onClick={handleTranslate}
          disabled={busy}
          aria-label="Dịch chương này"
          className={cn(
            'shrink-0 inline-flex items-center gap-1',
            'h-7 px-2 rounded-sm text-xs font-medium',
            'text-accent-text hover:bg-accent/15',
            'transition-colors cursor-pointer',
            'disabled:opacity-60 disabled:cursor-not-allowed',
          )}
        >
          {busy ? (
            <Loader2 size={12} className="animate-spin" />
          ) : (
            <>
              Dịch
              <ArrowRight size={12} />
            </>
          )}
        </button>
      )}
    </div>
  )
}
