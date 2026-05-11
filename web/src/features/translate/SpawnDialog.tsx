import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Sparkles, Lock, Globe2, Users } from 'lucide-react'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import { api, type DraftVisibility, type SpawnTranslateResult } from '@shared/api/api'

// =============================================================================
// SpawnDialog — confirm a translation spawn for one chapter.
//
// Caller passes (chapterId, sourceLang→targetLang label) and a callback
// fired with the resulting translation_id. Modal handles:
//   • Default visibility = guild (from the active DA instance)
//   • Power options: all_guilds / private behind a disclosure
//   • Cache hit vs miss copy (we can't predict at modal time; the
//     spawn RPC reports cache_hit in the result, and the dialog
//     hands off to the caller for "Already done → just open it")
//   • Quota meter alongside the primary action
//
// Why a single confirm step instead of a two-step "preview + spawn":
// the cache lookup is server-side and cheap. We optimistically issue
// the spawn; if it cache-hits, no quota was consumed and the caller
// transitions straight into the reader.
// =============================================================================

interface Props {
  open:        boolean
  onClose:     () => void
  chapterId:   number
  /** Display copy: "Naruto Ch 1099 · 🇨🇳 → 🇻🇳" */
  title:       string
  /** The target language we'll spawn against. The picker that selects
   *  this lives upstream (manga page header); the dialog just spawns. */
  targetLang:  string
  /** Guild id from the active DA instance, when one is in scope.
   *  When null, we fall back to "private" (no guild scope available). */
  scopeGuildId: string | null
  onSpawned:   (result: SpawnTranslateResult) => void
}

export function SpawnDialog({
  open, onClose, chapterId, title, targetLang, scopeGuildId, onSpawned,
}: Props) {
  const [visibility, setVisibility] = useState<DraftVisibility>(
    scopeGuildId ? 'guild' : 'private',
  )
  const [advanced, setAdvanced]   = useState(false)

  const { data: quota } = useQuery({
    queryKey: ['quota'],
    queryFn:  api.getQuota,
    staleTime: 10_000,
    enabled:  open,
  })

  const mutation = useMutation({
    mutationFn: () =>
      api.spawnTranslate({
        chapter_id:     chapterId,
        target_lang:    targetLang,
        force_private:  visibility === 'private',
        visibility,
        scope_guild_id: visibility === 'guild' ? scopeGuildId : null,
      }),
    onSuccess: (result) => {
      onSpawned(result)
      onClose()
    },
  })

  const exhausted = quota
    && !quota.is_admin
    && (quota.remaining_day === 0 || quota.remaining_hour === 0)

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Dịch chương"
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={mutation.isPending}>
            Huỷ
          </Button>
          <Button
            variant="primary"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !!exhausted}
          >
            <Sparkles size={13} />
            {mutation.isPending ? 'Đang gửi…' : 'Dịch'}
          </Button>
        </>
      }
    >
      <div className="text-sm text-text">
        <p className="font-medium mb-1">{title}</p>
        <p className="text-text-subtle text-xs">
          Pipeline sẽ chạy: OCR → Dịch bằng LLM → Render (~2 phút).
          Nếu có người đã dịch chương này, bạn dùng bản đó miễn phí.
        </p>
      </div>

      {/* Visibility — guild default, advanced disclosure for power
          users. Most users never expand this row. */}
      <div className="mt-5 space-y-2">
        <VisibilityRow
          value="guild"
          current={visibility}
          onPick={setVisibility}
          icon={<Users size={13} />}
          label="Chia sẻ với guild đang ở"
          hint={scopeGuildId ? 'Thành viên cùng guild dùng được' : 'Cần ở trong một guild'}
          disabled={!scopeGuildId}
        />
        {advanced && (
          <>
            <VisibilityRow
              value="all_guilds"
              current={visibility}
              onPick={setVisibility}
              icon={<Globe2 size={13} />}
              label="Chia sẻ tất cả guild của tôi"
              hint="Mọi guild bạn đang là thành viên"
            />
            <VisibilityRow
              value="private"
              current={visibility}
              onPick={setVisibility}
              icon={<Lock size={13} />}
              label="Chỉ tôi"
              hint="Không dùng cache, không chia sẻ"
            />
          </>
        )}
        {!advanced && (
          <button
            type="button"
            onClick={() => setAdvanced(true)}
            className="text-[11px] text-text-subtle hover:text-text-muted cursor-pointer"
          >
            Tuỳ chọn nâng cao…
          </button>
        )}
      </div>

      {/* Quota strip */}
      {quota && !quota.is_admin && (
        <div className="mt-5 rounded-sm bg-surface px-3 py-2 text-[11px] text-text-subtle flex items-center justify-between">
          <span>
            Còn lại: <span className="text-text">{quota.remaining_day}/{quota.limit_day}</span> hôm nay
          </span>
          <span>
            <span className="text-text">{quota.remaining_hour}/{quota.limit_hour}</span> trong giờ
          </span>
        </div>
      )}
      {exhausted && (
        <p className="mt-2 text-[11px] text-error-text">
          Đã hết quota. Thử lại sau.
        </p>
      )}

      {/* Error */}
      {mutation.isError && (
        <p className="mt-3 text-[12px] text-error-text">
          {(mutation.error as Error)?.message ?? 'Spawn thất bại'}
        </p>
      )}
    </Modal>
  )
}

interface RowProps {
  value:    DraftVisibility
  current:  DraftVisibility
  onPick:   (v: DraftVisibility) => void
  icon:     React.ReactNode
  label:    string
  hint:     string
  disabled?: boolean
}

function VisibilityRow({
  value, current, onPick, icon, label, hint, disabled,
}: RowProps) {
  const active = value === current
  return (
    <button
      type="button"
      onClick={() => !disabled && onPick(value)}
      disabled={disabled}
      className={cn(
        'w-full flex items-start gap-3 rounded-sm px-3 py-2 text-left',
        'transition-colors cursor-pointer',
        active
          ? 'bg-accent/15 ring-1 ring-inset ring-accent'
          : 'bg-surface hover:bg-surface-2',
        disabled && 'opacity-50 cursor-not-allowed',
      )}
    >
      <span className={cn(
        'mt-0.5 size-5 rounded-sm flex items-center justify-center shrink-0',
        active ? 'bg-accent text-accent-fg' : 'bg-surface-2 text-text-muted',
      )}>
        {icon}
      </span>
      <span className="flex-1 min-w-0">
        <p className="text-[13px] text-text font-medium leading-snug">{label}</p>
        <p className="text-[11px] text-text-subtle mt-0.5">{hint}</p>
      </span>
    </button>
  )
}
