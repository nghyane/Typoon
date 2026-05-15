// SourcePicker — drop-down listing every version of the current
// chapter. Desktop: anchored popover from the source chip on the
// top bar. Mobile: bottom sheet.
//
// One row per version, sorted: AI done in target → raw target →
// AI in-flight → raw non-target → uninstalled/unreadable. Each
// row has a single action:
//
//   ✓ Đang đọc        — currently-rendering version (no action).
//   Đọc               — switch to this version. The pick is
//                       persisted as the work's source pref the
//                       moment the user taps, so prev/next/auto
//                       navigation automatically inherits the
//                       same (kind, lang, sourceLang) selection.
//                       No "save as default" toggle — every pick
//                       IS the commitment.
//   ✨ Dịch           — non-target raw, no AI yet for this source.
//                       Triggers the spawn pipeline. The row stays
//                       in the picker and switches to an inline
//                       progress chip ("Tải ảnh 12/40") until the
//                       AI translation row materialises.
//   Đã xong           — terminal chip shown between `phase: done`
//                       and the work payload refetch landing the
//                       new AI row, preventing double-spawn.
//   Thử lại           — retry after a spawn error.

import { useEffect, useMemo, useState } from 'react'
import { Sparkles, Check, CircleAlert, Loader2 } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { MenuShell } from '@shared/ui/MenuShell'
import { languageName } from '@shared/lib/lang'
import type { HubChapter, HubVersion } from '@features/title/mergeChapters'
import type { SpawnProgress } from '@features/title/useSpawnChapter'

import { LangChip, AiChip, RowButton } from './atoms'


interface Props {
  open:          boolean
  onClose:       () => void
  anchorRef:     React.RefObject<HTMLElement | null>
  targetLang:    string | null
  chapter:       HubChapter | null
  picked:        HubVersion | null
  /** Chapter-level spawn state — `useSpawnChapters` keys by chapter
   *  number so the same slot tracks any source-row spawn under this
   *  chapter. SourcePicker pins which version was last clicked and
   *  paints progress only on that row, keeping the rest static. */
  spawnState:    SpawnProgress | null
  onPick:        (version: HubVersion) => void
  onSpawn:       (chapter: HubChapter, raw: HubVersion) => void
}


export function SourcePicker({
  open, onClose, anchorRef,
  targetLang, chapter, picked,
  spawnState, onPick, onSpawn,
}: Props) {
  const tgt = normalizeLang(targetLang)

  // Pin which raw version the user last clicked "Dịch" on so the
  // chapter-level spawn slot can be painted on the right row only.
  // Clears when the spawn lands (`done`) or the picker resets — a
  // fresh open shouldn't keep the previous row's progress chip.
  const [spawningKey, setSpawningKey] = useState<string | null>(null)
  useEffect(() => {
    if (!spawnState) { setSpawningKey(null); return }
    if (spawnState.phase === 'done' || spawnState.phase === 'idle') {
      setSpawningKey(null)
    }
  }, [spawnState])
  useEffect(() => {
    if (!open) setSpawningKey(null)
  }, [open])

  const translatedKeys = useMemo(() => {
    if (!chapter) return new Set<string>()
    const s = new Set<string>()
    for (const v of chapter.versions) {
      if (v.kind !== 'translation') continue
      if (!v.sourceLang || v.materialId == null) continue
      if (normalizeLang(v.lang) !== tgt) continue
      s.add(`${normalizeLang(v.sourceLang)}::${v.materialId}`)
    }
    return s
  }, [chapter, tgt])

  const rows = useMemo(() => sortVersions(chapter, tgt), [chapter, tgt])

  return (
    <MenuShell
      open={open}
      onClose={onClose}
      anchorRef={anchorRef}
      title="Nguồn đọc"
      align="end"
      minWidth={320}
      maxWidth={420}
    >
      <div className="py-1">
        {rows.length === 0 && (
          <p className="px-4 py-6 text-sm text-text-subtle text-center">
            Chương này chưa có nguồn nào.
          </p>
        )}

        {rows.map((v) => {
          const isPicked = picked?.key === v.key
          const isSpawnable =
            v.kind === 'raw'
            && Boolean(v.upstreamUrl && v.sourceId)
            && normalizeLang(v.lang) !== tgt
            && v.materialId != null
            && !translatedKeys.has(`${normalizeLang(v.lang)}::${v.materialId}`)

          return (
            <SourceRow
              key={v.key}
              version={v}
              isPicked={isPicked}
              isSpawnable={isSpawnable}
              spawnState={spawningKey === v.key ? spawnState : null}
              onPick={() => {
                onPick(v)
                onClose()
              }}
              onSpawn={() => {
                if (!chapter) return
                setSpawningKey(v.key)
                onSpawn(chapter, v)
              }}
            />
          )
        })}
      </div>
    </MenuShell>
  )
}


// ── Row ────────────────────────────────────────────────────────


function SourceRow({
  version: v, isPicked, isSpawnable, spawnState, onPick, onSpawn,
}: {
  version:     HubVersion
  isPicked:    boolean
  isSpawnable: boolean
  spawnState:  SpawnProgress | null
  onPick:      () => void
  onSpawn:     () => void
}) {
  const isTranslation = v.kind === 'translation'
  const pickable =
    isTranslation
      ? v.state === 'done'
      : Boolean(v.upstreamUrl && v.sourceId)

  const phase: SpawnProgress['phase'] = spawnState?.phase ?? 'idle'
  const isSpawning =
    phase !== 'idle' && phase !== 'done' && phase !== 'error'

  const primary =
    isTranslation
      ? translationLabel(v)
      : rawLabel(v)
  const meta = isSpawning
    ? spawnProgressLabel(spawnState!)
    : phase === 'error'
      ? (spawnState?.error ?? 'Lỗi — bấm để thử lại')
      : versionStateHint(v)

  // Action region. A picked raw row that is also spawnable shows both
  // the "Đang đọc" badge AND the "Dịch" button so the user can kick
  // off a translation without first having to switch away from the
  // current raw version.
  let action: React.ReactNode = null
  if (isPicked && isSpawnable && phase === 'idle') {
    action = (
      <div className="flex items-center gap-1.5">
        <ActiveBadge />
        <RowButton variant="accent" onClick={onSpawn}>
          <Sparkles size={11} />
          Dịch
        </RowButton>
      </div>
    )
  } else if (isPicked) {
    action = <ActiveBadge />
  } else if (isSpawning) {
    action = <SpawningChip />
  } else if (phase === 'done') {
    action = <DoneChip />
  } else if (phase === 'error') {
    action = (
      <RowButton variant="soft" onClick={onSpawn} title={spawnState?.error ?? undefined}>
        <CircleAlert size={11} />
        Thử lại
      </RowButton>
    )
  } else if (isSpawnable) {
    action = (
      <RowButton variant="accent" onClick={onSpawn}>
        <Sparkles size={11} />
        Dịch
      </RowButton>
    )
  } else if (pickable) {
    action = <RowButton variant="soft" onClick={onPick}>Đọc</RowButton>
  } else {
    action = <span className="text-xs text-text-subtle">—</span>
  }

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-4 py-2.5',
        'border-l-2 transition-colors duration-150',
        isPicked
          // Active state: neutral surface lift + accent rail on the
          // left edge. Accent itself stays in the action column
          // (text "Đang đọc", chip glow) so the row doesn't read
          // as "I'm a CTA, click me".
          ? 'bg-row-active border-l-accent'
        : isSpawning || phase === 'done'
          // Info elevation for the brief window the spawn pipeline
          // owns this row. Same border-l-0 as inactive rows so the
          // active-rail stays a one-row signal.
          ? 'bg-info-bg border-l-transparent'
        : pickable || isSpawnable
          ? 'border-l-transparent hover:bg-hover'
          : 'border-l-transparent',
      )}
    >
      <div className="w-4 shrink-0 flex items-center justify-center">
        {isPicked && <Check size={14} className="text-accent" />}
      </div>

      <div className="shrink-0 flex items-center gap-1">
        <LangChip lang={v.lang} active={isPicked} />
        {isTranslation && <AiChip active={isPicked} />}
      </div>

      <div className="min-w-0 flex-1">
        <p
          className={cn(
            'text-sm truncate',
            isPicked ? 'text-text font-medium' : 'text-text',
          )}
        >
          {primary}
        </p>
        {meta && (
          <p className="text-xs text-text-subtle truncate mt-0.5">
            {meta}
          </p>
        )}
      </div>

      <div className="shrink-0">{action}</div>
    </div>
  )
}


function ActiveBadge() {
  return (
    <span
      className={cn(
        'inline-flex items-center h-7 px-2',
        'text-xs uppercase tracking-wide font-medium text-accent',
      )}
    >
      Đang đọc
    </span>
  )
}


function SpawningChip() {
  // Spinner-only chip — the verbose phase label rides in the meta
  // line below the row primary text. Keeps the action column
  // narrow so row height stays uniform.
  return (
    <span
      className={cn(
        'inline-flex items-center justify-center',
        'h-7 w-7 rounded-md',
        'bg-info-bg text-info-text',
      )}
      aria-label="Đang dịch…"
    >
      <Loader2 size={12} className="animate-[ts-spin_0.8s_linear_infinite]" />
    </span>
  )
}


function DoneChip() {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1',
        'h-7 px-2.5 rounded-md',
        'bg-success-bg text-success-text text-xs font-medium',
      )}
    >
      <Check size={11} />
      Đã xong
    </span>
  )
}


// ── Helpers ───────────────────────────────────────────────────


/** Stable row ordering. Pure so the picker doesn't accidentally
 *  re-sort on every spawn-progress tick. */
function sortVersions(ch: HubChapter | null, tgt: string): HubVersion[] {
  if (!ch) return []
  const out = [...ch.versions]
  out.sort((a, b) => sortKey(a, tgt) - sortKey(b, tgt))
  return out
}


function sortKey(v: HubVersion, tgt: string): number {
  const isTgtLang = normalizeLang(v.lang) === tgt
  if (v.kind === 'translation') {
    if (!isTgtLang) return 50
    if (v.state === 'done') return 0
    if (v.state === 'running' || v.state === 'pending') return 20
    if (v.state === 'error' || v.state === 'blocked') return 30
    return 40
  }
  const installed = Boolean(v.upstreamUrl && v.sourceId)
  if (!installed) return 90
  return isTgtLang ? 10 : 40
}


function translationLabel(v: HubVersion): string {
  const user = v.creatorName ? `@${v.creatorName}` : 'Người dùng'
  const src  = v.sourceLang ? languageName(v.sourceLang) : languageName(v.lang)
  return `${user} · từ ${src}`
}


function rawLabel(v: HubVersion): string {
  const scan = v.creatorName ? `@${v.creatorName}` : 'Scanlator'
  const src  = v.sourceName ?? 'Nguồn'
  return `${src} · ${scan}`
}


function versionStateHint(v: HubVersion): string | null {
  if (v.kind !== 'translation') return null
  if (v.state === 'done')     return 'Sẵn sàng đọc'
  if (v.state === 'running')  return 'Đang dịch…'
  if (v.state === 'pending')  return 'Đang chờ trong hàng'
  if (v.state === 'error')    return 'Lỗi — thử lại từ work hub'
  if (v.state === 'blocked')  return 'Tạm dừng — admin'
  return null
}


function spawnProgressLabel(p: SpawnProgress): string {
  switch (p.phase) {
    case 'fetching':    return 'Lấy danh sách trang…'
    case 'downloading': return `Tải ảnh ${p.current}/${p.total}`
    case 'packing':     return 'Đóng gói…'
    case 'uploading':   return `Tải lên ${p.pct}%`
    case 'spawning':    return 'Khởi tạo…'
    case 'error':       return p.error ?? 'Lỗi'
    default:            return 'Đang dịch…'
  }
}


function normalizeLang(s: string | null | undefined): string {
  if (!s) return ''
  return s.toLowerCase().split(/[-_]/)[0] ?? ''
}
