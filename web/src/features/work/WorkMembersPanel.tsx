// WorkMembersPanel — "Nguồn đang đọc" section on the Work hub.
//
// Mirrors the LinkSuggestionPanel pattern (flat section, no card
// chrome, route padding) but inverted in concept: lists materials
// CURRENTLY in this Work plus the affordances to leave it.
//
// Two affordances per row:
//
//   1. Báo nhầm nguồn → community split vote (+1). Reaches the
//      threshold = inline move to a fresh Work + toast with link.
//   2. Hoàn tác liên kết (owner-only, undo window) → instant
//      force-unlink. Visible only to the user who fired the
//      force-link AND only while the window is open. Server gates
//      both; the UI here uses `force_link_undo_expires_at` to render
//      a live countdown and hide itself when the window closes.
//
// The panel only renders when the Work has ≥2 materials — a single-
// material work has nothing to split / nothing to compare against.

import { useEffect, useRef, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import {
  ChevronDown, Layers, RotateCcw, Undo2, X,
} from 'lucide-react'

import type { ApiWorkMember } from '@shared/api/api'
import { Cover } from '@shared/ui/Cover'
import { Badge, Spinner, Tag } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { toast } from '@shared/ui/Toaster'
import { confirm } from '@shared/ui/Confirm'
import { cn } from '@shared/lib/cn'
import { useSources } from '@features/browse/sources'

import {
  isUndoWindowOpen, useCastSplitVote, useForceUnlink, useWorkMembers,
} from './linkVotes'


/** Number of rows always visible before the "Xem thêm" affordance. */
const INITIAL_VISIBLE = 3


interface Props {
  workId: number
}


export function WorkMembersPanel({ workId }: Props) {
  const q = useWorkMembers(workId)
  const members = q.data ?? []
  const [expanded, setExpanded] = useState(false)

  // Ref used only for scroll-into-view on collapse.
  const headerRef = useRef<HTMLElement>(null)

  // Single-material works have nothing meaningful to surface — skip
  // the section entirely so the hub stays tight.
  if (!q.isPending && members.length < 2) return null

  const hiddenCount = Math.max(0, members.length - INITIAL_VISIBLE)

  function handleToggle() {
    const wasExpanded = expanded
    setExpanded((v) => !v)
    if (wasExpanded) {
      requestAnimationFrame(() => {
        headerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      })
    }
  }

  return (
    <section ref={headerRef} className="px-4 sm:px-6 py-3">
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="flex items-center gap-2 mb-2">
        <h2 className="text-xs uppercase tracking-wide text-text-subtle inline-flex items-center gap-1.5">
          <Layers size={12} />
          Nguồn đang đọc
        </h2>
        {!q.isPending && (
          <Tag tone="neutral" size="sm">{members.length}</Tag>
        )}
      </header>

      {q.isPending ? (
        <div className="py-2 flex justify-center">
          <Spinner size={16} />
        </div>
      ) : expanded ? (
        // ── Expanded — full rows ──────────────────────────
        <>
          <ul className="space-y-1">
            {members.map((m) => (
              <MemberRow
                key={m.material_id}
                member={m}
                workId={workId}
              />
            ))}
          </ul>
          <button
            type="button"
            onClick={handleToggle}
            className={cn(
              'mt-2 flex items-center gap-1 text-xs cursor-pointer',
              'text-text-subtle hover:text-text transition-colors',
            )}
          >
            <ChevronDown size={12} className="rotate-180 transition-transform duration-200" />
            Thu gọn
          </button>
        </>
      ) : (
        // ── Collapsed — pill chips, ultra-compact ─────────
        <div className="flex items-center gap-1.5 flex-wrap">
          {members.slice(0, INITIAL_VISIBLE).map((m) => (
            <MemberChip key={m.material_id} member={m} workId={workId} />
          ))}
          {hiddenCount > 0 && (
            <button
              type="button"
              onClick={handleToggle}
              className={cn(
                'inline-flex items-center gap-1 h-6 px-2 rounded-full',
                'text-xs text-text-subtle bg-surface-2 hover:bg-hover',
                'border border-border-soft/40 hover:border-border-soft/70',
                'transition-colors cursor-pointer',
              )}
            >
              +{hiddenCount}
              <ChevronDown size={11} />
            </button>
          )}
        </div>
      )}
    </section>
  )
}


// ── Member chip (collapsed view) ──────────────────────────────
//
// Ultra-compact pill: source name + lang tag, tapping it expands the
// full list. No action affordances in this state — keeps the surface
// scannable at a glance.


function MemberChip({
  member, workId: _workId,
}: {
  member:  ApiWorkMember
  workId:  number
}) {
  const installed   = useSources((s) => s.sources)
  const sourceLabel = member.source
    ? (installed[member.source]?.manifest.name ?? member.source)
    : 'Tải lên'
  const langLabel = (member.languages[0] ?? '').toUpperCase()

  const hasAlert = member.viewer_split_vote === 1

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 h-6 px-2 rounded-full',
        'text-xs text-text-muted bg-surface-2',
        'border',
        hasAlert
          ? 'border-warning/40 text-warning-text'
          : 'border-border-soft/30',
      )}
    >
      {langLabel && (
        <span className="font-medium text-text-subtle">{langLabel}</span>
      )}
      <span className="truncate max-w-[8rem]">{sourceLabel}</span>
      {hasAlert && <span className="text-warning shrink-0">·</span>}
    </span>
  )
}


// ── Member row (expanded view) ─────────────────────────────────


function MemberRow({
  member, workId,
}: {
  member: ApiWorkMember
  workId: number
}) {
  const installed = useSources((s) => s.sources)
  const nav       = useNavigate()

  const sourceLabel = member.source
    ? (installed[member.source]?.manifest.name ?? member.source)
    : 'Tải lên'
  const langLabel = (member.languages[0] ?? '').toUpperCase()

  const splitVote = useCastSplitVote(workId)
  const unlink    = useForceUnlink(workId)

  // ── Owner undo window — live countdown.
  // Re-render once a second so the "(còn 8:42)" hint stays current
  // and the affordance disappears when the window closes without
  // waiting for the next refetch.
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!member.force_link_undo_expires_at) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [member.force_link_undo_expires_at])
  const canUndo = isUndoWindowOpen(member, now)

  const voted     = member.viewer_split_vote
  const score     = member.pending_split_score
  const threshold = member.pending_split_threshold
  const busy      = splitVote.isPending || unlink.isPending

  // ── Actions ────────────────────────────────────────────────

  const onSplitVote = async (vote: 1 | -1) => {
    if (vote === 1) {
      const ok = await confirm({
        title: 'Báo nhầm nguồn?',
        description: (
          <div className="space-y-2 text-sm text-text-muted">
            <p>
              Đề xuất tách <b className="text-text">{member.title}</b> khỏi
              truyện này. Nếu đủ {threshold} phiếu, nguồn sẽ chuyển sang
              một trang truyện riêng.
            </p>
          </div>
        ),
        confirmText: 'Báo nhầm nguồn',
      })
      if (!ok) return
    }
    splitVote.mutate(
      { material_id: member.material_id, vote },
      {
        onSuccess: (res) => {
          if (res.split && res.new_work_id != null) {
            toast.success(
              `Đã tách "${member.title}" sang trang riêng.`,
            )
            return
          }
          if (res.blocked_reason === 'solo_member') {
            toast.error(
              'Không thể tách — truyện cần ít nhất 2 nguồn để tách.',
            )
            return
          }
          if (vote === 1) {
            const need = Math.max(0, threshold - res.score)
            toast.success(
              need > 0
                ? `Đã báo nhầm. Còn ${need} phiếu để tách.`
                : `Đã báo nhầm.`,
            )
          } else {
            toast.success('Đã hoàn tác phiếu.')
          }
        },
        onError: (e: Error) => toast.error(e.message),
      },
    )
  }

  const onUndo = async () => {
    const ok = await confirm({
      title: `Hoàn tác liên kết "${member.title}"?`,
      description: 'Nguồn này sẽ chuyển sang trang truyện riêng ngay lập tức.',
      confirmText: 'Hoàn tác',
    })
    if (!ok) return
    unlink.mutate(
      { material_id: member.material_id },
      {
        onSuccess: (res) => {
          if (res.split && res.new_work_id != null) {
            const newId = res.new_work_id
            toast.success(
              `Đã hoàn tác. "${member.title}" có trang riêng.`,
            )
            // Auto-nav so the user lands on the new isolated work
            // and can verify. Stays a single render frame after the
            // toast so the message has a moment to render.
            setTimeout(() => nav({
              to:     '/w/$workId',
              params: { workId: String(newId) },
            }), 200)
            return
          }
          if (res.blocked_reason === 'solo_member') {
            toast.error('Không thể hoàn tác — chỉ còn 1 nguồn trong truyện.')
            return
          }
          toast.success('Đã hoàn tác.')
        },
        onError: (e: Error) => toast.error(e.message),
      },
    )
  }

  // ── Render ────────────────────────────────────────────────

  return (
    <li
      className={cn(
        'flex items-center gap-3 p-2 rounded-sm transition-colors',
        'bg-surface-2/40 hover:bg-surface-2/70',
      )}
    >
      <div className="w-10 h-14 shrink-0 rounded-sm overflow-hidden">
        <Cover
          src={member.cover_url}
          title={member.title}
          className="w-full h-full"
          fontSize="text-xs"
        />
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-sm text-text truncate">{member.title}</div>
        <div className="flex items-center gap-1.5 mt-0.5 min-w-0">
          {langLabel && (
            <Tag tone="outline" size="sm">{langLabel}</Tag>
          )}
          <span className="text-xs text-text-subtle truncate">
            {sourceLabel}
          </span>
        </div>
      </div>

      <SplitAction
        voted={voted}
        score={score}
        threshold={threshold}
        busy={busy}
        canUndoForceLink={canUndo}
        undoExpiresAt={member.force_link_undo_expires_at}
        now={now}
        onVote={onSplitVote}
        onUndoForceLink={onUndo}
      />
    </li>
  )
}


// ── Action cluster (right-aligned) ─────────────────────────────


function SplitAction({
  voted, score, threshold, busy, canUndoForceLink, undoExpiresAt, now,
  onVote, onUndoForceLink,
}: {
  voted:            number | null
  score:            number
  threshold:        number
  busy:             boolean
  canUndoForceLink: boolean
  undoExpiresAt:    string | null
  now:              number
  onVote:           (v: 1 | -1) => void
  onUndoForceLink:  () => void
}) {
  // Voted = 1 → "Đang chờ tách (X/N)" + undo. Most prominent affordance.
  if (voted === 1) {
    const need = Math.max(0, threshold - score)
    return (
      <div className="shrink-0 flex flex-col items-end gap-1">
        <Badge tone="warning">
          Đang chờ tách
          {need > 0 && (
            <span className="tabular text-text-muted font-normal">
              {score}/{threshold}
            </span>
          )}
        </Badge>
        <UndoLink
          busy={busy}
          label="Bỏ phiếu"
          onClick={() => onVote(-1)}
        />
      </div>
    )
  }

  // Voted = -1 → user already pushed back. Tiny "đã giữ" badge +
  // option to flip to +1 (re-report).
  if (voted === -1) {
    return (
      <div className="shrink-0 flex flex-col items-end gap-1">
        <Badge tone="neutral">Đã giữ</Badge>
        <UndoLink
          busy={busy}
          label="Báo nhầm"
          onClick={() => onVote(1)}
        />
      </div>
    )
  }

  // No vote yet. Owner of a recent force_link gets the instant undo
  // button INSTEAD of the report button — undoing your own mistake
  // shouldn't need a vote.
  if (canUndoForceLink) {
    return (
      <div className="shrink-0 flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          disabled={busy}
          onClick={onUndoForceLink}
        >
          <Undo2 size={14} />
          Hoàn tác
          {undoExpiresAt && (
            <span className="tabular text-text-subtle">
              ({formatCountdown(undoExpiresAt, now)})
            </span>
          )}
        </Button>
      </div>
    )
  }

  // Default: explicit "Báo nhầm nguồn" button. Wording is the
  // affordance — user reads it and knows what it does without a
  // tooltip.
  return (
    <div className="shrink-0">
      <Button
        variant="ghost"
        size="sm"
        disabled={busy}
        onClick={() => onVote(1)}
        title="Đề xuất tách nguồn này khỏi truyện"
      >
        <X size={14} />
        Báo nhầm nguồn
      </Button>
    </div>
  )
}


function UndoLink({
  busy, label, onClick,
}: {
  busy:    boolean
  label:   string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      disabled={busy}
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1 text-xs cursor-pointer',
        'text-text-subtle hover:text-text transition-colors',
        busy && 'opacity-60 cursor-wait',
      )}
    >
      <RotateCcw size={11} /> {label}
    </button>
  )
}


/** "9:42" countdown for the owner undo window. Returns "" when
 *  the timestamp is missing / already past. */
function formatCountdown(iso: string, now: number): string {
  const expires = Date.parse(iso)
  if (!Number.isFinite(expires)) return ''
  const remaining = Math.max(0, Math.floor((expires - now) / 1000))
  const min = Math.floor(remaining / 60)
  const sec = remaining % 60
  return `${min}:${sec.toString().padStart(2, '0')}`
}