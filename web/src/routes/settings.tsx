import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { Plus, Copy, Check, Trash2, Key, ShieldAlert, Zap } from 'lucide-react'
import { api, type ApiTokenInfo } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import { input as inputCls, label as labelCls, Spinner } from '@shared/ui/primitives'
import { SettingsSection, SettingsDivider } from '@shared/ui/SettingsForm'
import { DataTable, Th } from '@shared/ui/DataTable'
import { EmptyState } from '@shared/ui/EmptyState'
import { toast } from '@shared/ui/Toaster'
import { Modal } from '@shared/ui/Modal'
import { confirm } from '@shared/ui/Confirm'
import { timeAgo } from '@shared/lib/time'
import { cn } from '@shared/lib/cn'

const TOKEN_NAME_MIN = 2
const TOKEN_NAME_MAX = 40

// =============================================================================
// Account-level settings page. One section (API tokens) — but laid out so a
// future second section drops in without restructuring: page header on top,
// `SettingsSection` blocks below with `SettingsDivider` between them.
//
// Visual rules:
//   - No decorative icon-box at the page header (sidebar item is the context).
//   - Section primary action stays in the header row, visible in every state
//     so it doesn't jump between empty / list.
//   - Single bordered container; inner state (loading / empty / list) swaps
//     without changing the outer frame.
//   - Solid `bg-surface` (no alpha). Matches project-detail cards.
// =============================================================================

function SettingsPage() {
  return (
    <div className="px-6 pt-8 pb-20 max-w-3xl">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-text">Cài đặt</h1>
        <p className="text-sm text-text-subtle mt-1">Quản lý tài khoản và quyền truy cập của bạn.</p>
      </header>

      <QuotaSection />
      <SettingsDivider />
      <TokensSection />
    </div>
  )
}

// ── Quota ───────────────────────────────────────────────────────────────────
//
// Read-only summary of the per-user chapter quota that the API enforces on
// upload+start, /start, /redo. Three meters: today / this hour / concurrent.
// Hidden for admins (their quota is uncapped — showing 0/0 would mislead).

function QuotaSection() {
  const { data, isLoading } = useQuery({
    queryKey: ['quota'],
    queryFn:  api.getQuota,
    refetchInterval: 30_000,
  })

  if (isLoading) {
    return (
      <SettingsSection title="Quota" description="Đang tải…">
        <div className="h-12 flex items-center"><Spinner /></div>
      </SettingsSection>
    )
  }
  if (!data) return null

  if (data.is_admin) {
    return (
      <SettingsSection
        title="Quota"
        description="Tài khoản admin không bị giới hạn chương dịch."
      >
        <div className="flex items-center gap-2 text-[13px] text-text-muted">
          <Zap size={14} className="text-emerald-400" />
          <span>Không giới hạn</span>
        </div>
      </SettingsSection>
    )
  }

  return (
    <SettingsSection
      title="Quota"
      description="Mỗi lần dịch một chương (upload + start, start, redo) sẽ tốn 1 lượt. Quota reset theo cửa sổ trượt."
    >
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <QuotaMeter
          label="Hôm nay"
          used={data.used_day}
          limit={data.limit_day}
        />
        <QuotaMeter
          label="Trong giờ"
          used={data.used_hour}
          limit={data.limit_hour}
        />
        <QuotaMeter
          label="Đang xử lý"
          used={data.in_flight}
          limit={data.limit_concurrent}
        />
      </div>
    </SettingsSection>
  )
}

function QuotaMeter({ label, used, limit }: { label: string; used: number; limit: number }) {
  const pct  = limit > 0 ? Math.min(100, Math.round((used / limit) * 100)) : 0
  const tone =
    pct >= 90 ? 'bg-rose-500'
    : pct >= 50 ? 'bg-amber-500'
    : 'bg-emerald-500'
  return (
    <div className="rounded-md border border-border-soft bg-surface px-3 py-3">
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] text-text-subtle">{label}</span>
        <span className="text-[13px] tabular-nums">
          <span className="text-text font-medium">{used}</span>
          <span className="text-text-subtle">/{limit}</span>
        </span>
      </div>
      <div className="mt-2 h-1.5 rounded-full bg-surface-2 overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', tone)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── API tokens ──────────────────────────────────────────────────────────────

function TokensSection() {
  const qc = useQueryClient()
  const [wizardOpen, setWizardOpen] = useState(false)

  const { data: tokens = [], isLoading } = useQuery({
    queryKey: ['tokens'],
    queryFn:  api.listTokens,
  })

  const revoke = useMutation({
    mutationFn: (id: number) => api.revokeToken(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tokens'] })
      toast.success('Đã thu hồi')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const askRevoke = async (t: ApiTokenInfo) => {
    const ok = await confirm({
      title:       `Thu hồi token "${t.name}"?`,
      description: (
        <>
          Tool đang dùng token này sẽ <strong className="text-text">ngừng hoạt động</strong> ngay lập tức.
          Hành động không thể hoàn tác.
        </>
      ),
      confirmText: 'Thu hồi',
      tone:        'danger',
    })
    if (ok) revoke.mutate(t.id)
  }

  return (
    <>
      <SettingsSection
        title="API tokens"
        description="Token cho extension Typoon trên Chrome/Firefox truy cập tài khoản của bạn. Mỗi thiết bị dùng một token riêng và có thể thu hồi bất kỳ lúc nào."
        action={
          <Button variant="primary" onClick={() => setWizardOpen(true)}>
            <Plus size={13} />
            Tạo token
          </Button>
        }
      >
        <DataTable>
          <thead>
            <tr className="bg-surface-2">
              <Th>Tên</Th>
              <Th className="hidden sm:table-cell">Token</Th>
              <Th>Hoạt động</Th>
              <Th className="w-12 text-right pr-3 sr-only">Thao tác</Th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan={4}>
                  <div className="flex items-center justify-center gap-2 py-12 text-sm text-text-subtle">
                    <Spinner /> Đang tải…
                  </div>
                </td>
              </tr>
            )}

            {!isLoading && tokens.length === 0 && (
              <tr>
                <td colSpan={4}>
                  <EmptyState
                    icon={Key}
                    title="Chưa có token nào"
                    hint="Tạo token để dùng với extension Typoon. Token sẽ hiện ở đây sau khi tạo."
                    action={
                      <Button variant="primary" onClick={() => setWizardOpen(true)}>
                        <Plus size={13} />
                        Tạo token đầu tiên
                      </Button>
                    }
                  />
                </td>
              </tr>
            )}

            {!isLoading && tokens.map((t) => (
              <TokenRow
                key={t.id}
                token={t}
                pending={revoke.isPending && revoke.variables === t.id}
                onRevoke={() => askRevoke(t)}
              />
            ))}
          </tbody>
        </DataTable>
      </SettingsSection>

      <CreateTokenWizard
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        existing={tokens}
      />
    </>
  )
}

function TokenRow({
  token: t, pending, onRevoke,
}: {
  token:    ApiTokenInfo
  pending:  boolean
  onRevoke: () => void
}) {
  return (
    <tr className="border-b border-border-soft last:border-0 group hover:bg-hover transition-colors">
      <td className="px-3 py-2.5 text-[13px] font-medium text-text">
        {t.name}
      </td>
      <td className="px-3 py-2.5 text-xs hidden sm:table-cell">
        <code className="font-mono text-text-muted">typ_{t.prefix}…</code>
      </td>
      <td className="px-3 py-2.5 text-xs text-text-subtle whitespace-nowrap">
        {t.last_used ? (
          <span title={t.last_used}>Dùng {timeAgo(t.last_used)}</span>
        ) : (
          <span>Chưa dùng</span>
        )}
        {t.created_at && (
          <>
            <span aria-hidden className="text-text-subtle/50 mx-2">·</span>
            <span title={t.created_at}>Tạo {timeAgo(t.created_at)}</span>
          </>
        )}
      </td>
      <td className="px-3 py-2.5 text-right pr-3">
        <Button
          variant="ghost"
          size="sm"
          icon
          onClick={onRevoke}
          disabled={pending}
          className="text-text-subtle hover:text-error-text hover:bg-error/10"
          title="Thu hồi"
          aria-label={`Thu hồi token ${t.name}`}
        >
          {pending ? <Spinner /> : <Trash2 size={13} />}
        </Button>
      </td>
    </tr>
  )
}

// ── create wizard ──────────────────────────────────────────────────────────
//
// Single Modal with two steps so the user keeps spatial context: name → reveal.
// Step 1 validates client-side (length + duplicate name) so we don't pay a
// server round-trip for trivially bad input. Step 2 auto-selects + auto-copies
// the token (best-effort, falls back to manual Copy button), and asks for an
// explicit "Đã lưu" confirmation before allowing dismiss — avoids the user
// closing by mistake before storing the secret.

function CreateTokenWizard({
  open, onClose, existing,
}: {
  open:     boolean
  onClose:  () => void
  existing: ApiTokenInfo[]
}) {
  const qc = useQueryClient()
  const [name, setName]       = useState('')
  const [created, setCreated] = useState<{ token: string; name: string } | null>(null)
  const [confirmedSaved, setConfirmedSaved] = useState(false)

  // Reset wizard whenever it (re)opens. Keeps the next session clean even if
  // the previous one ended with a partially filled name or revealed token.
  useEffect(() => {
    if (open) {
      setName('')
      setCreated(null)
      setConfirmedSaved(false)
    }
  }, [open])

  const create = useMutation({
    mutationFn: (n: string) => api.createToken(n),
    onSuccess: (t) => {
      qc.invalidateQueries({ queryKey: ['tokens'] })
      setCreated({ token: t.token, name: t.name })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const trimmed = name.trim()
  const tooShort   = trimmed.length > 0 && trimmed.length < TOKEN_NAME_MIN
  const tooLong    = trimmed.length > TOKEN_NAME_MAX
  const duplicate  = !!trimmed && existing.some((t) => t.name.toLowerCase() === trimmed.toLowerCase())
  const validation =
    tooShort  ? `Tên cần ít nhất ${TOKEN_NAME_MIN} ký tự`
    : tooLong ? `Tên không quá ${TOKEN_NAME_MAX} ký tự`
    : duplicate ? 'Đã có token với tên này'
    : null
  const canSubmit = !!trimmed && !validation && !create.isPending

  const submit = () => { if (canSubmit) create.mutate(trimmed) }

  const tryClose = async () => {
    // After reveal: require explicit "đã lưu" confirmation before close.
    // The user isn't trapped — they can still cancel — but an accidental
    // Esc / backdrop click won't lose a non-recoverable secret.
    if (created && !confirmedSaved) {
      const ok = await confirm({
        title:       'Đóng mà chưa lưu token?',
        description: 'Token chỉ hiện 1 lần. Sau khi đóng, bạn sẽ không xem lại được — phải tạo token mới.',
        confirmText: 'Đóng',
        cancelText:  'Để tôi copy',
        tone:        'danger',
      })
      if (!ok) return
    }
    onClose()
  }

  return (
    <Modal
      open={open}
      onClose={tryClose}
      title={created ? 'Token mới' : 'Tạo API token'}
    >
      {!created ? (
        <CreateStep
          name={name}
          onName={setName}
          validation={validation}
          canSubmit={canSubmit}
          pending={create.isPending}
          onSubmit={submit}
          onCancel={tryClose}
        />
      ) : (
        <RevealStep
          token={created.token}
          name={created.name}
          confirmedSaved={confirmedSaved}
          onConfirmedSavedChange={setConfirmedSaved}
          onDone={onClose}
        />
      )}
    </Modal>
  )
}

function CreateStep({
  name, onName, validation, canSubmit, pending, onSubmit, onCancel,
}: {
  name:       string
  onName:     (v: string) => void
  validation: string | null
  canSubmit:  boolean
  pending:    boolean
  onSubmit:   () => void
  onCancel:   () => void
}) {
  const trimmed   = name.trim()
  const remaining = TOKEN_NAME_MAX - trimmed.length

  // 4 ready-made suggestions cover the common shapes so the user doesn't have
  // to invent a name. Click fills the field; a fresh field still gets focus
  // so they can edit immediately.
  const SUGGESTIONS = ['Chrome extension', 'Firefox extension', 'MacBook', 'Tampermonkey']

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); onSubmit() }}
      className="space-y-4 p-5"
    >
      <div>
        <label htmlFor="token-name" className={labelCls}>Tên gợi nhớ</label>
        <input
          id="token-name"
          value={name}
          onChange={(e) => onName(e.target.value)}
          placeholder="VD: Chrome extension"
          autoFocus
          maxLength={TOKEN_NAME_MAX + 10}
          className={inputCls}
          aria-invalid={!!validation}
          aria-describedby="token-name-hint"
        />
        <div id="token-name-hint" className="mt-1.5 flex items-start justify-between gap-3">
          <p className={`text-xs leading-relaxed ${validation ? 'text-error-text' : 'text-text-subtle'}`}>
            {validation ?? 'Đặt tên để biết token này dùng cho công cụ nào. Chỉ bạn nhìn thấy.'}
          </p>
          <span
            className={`text-[11px] font-mono shrink-0 tabular-nums ${
              remaining < 0 ? 'text-error-text' : 'text-text-subtle/60'
            }`}
          >
            {trimmed.length}/{TOKEN_NAME_MAX}
          </span>
        </div>
      </div>

      <div>
        <p className="text-[11px] uppercase tracking-wider text-text-subtle font-medium mb-2">
          Gợi ý
        </p>
        <div className="flex flex-wrap gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              type="button"
              key={s}
              onClick={() => onName(s)}
              className="px-2 h-6 rounded-full text-xs text-text-muted bg-surface-2 hover:bg-hover hover:text-text transition-colors cursor-pointer"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="flex justify-end gap-2 pt-2">
        <Button type="button" onClick={onCancel}>Huỷ</Button>
        <Button type="submit" variant="primary" disabled={!canSubmit}>
          {pending ? <Spinner /> : <Plus size={13} />}
          Tạo token
        </Button>
      </div>
    </form>
  )
}

function RevealStep({
  token, name, confirmedSaved, onConfirmedSavedChange, onDone,
}: {
  token:                  string
  name:                   string
  confirmedSaved:         boolean
  onConfirmedSavedChange: (v: boolean) => void
  onDone:                 () => void
}) {
  const [copied, setCopied] = useState(false)
  const tokenRef = useRef<HTMLInputElement>(null)

  // On reveal: best-effort auto-copy + auto-select. Auto-copy may fail in
  // some browsers without an explicit user gesture in scope; the visible
  // Copy button covers that path. Auto-select (focus + select all) always
  // works and lets the user Cmd/Ctrl+C immediately.
  useEffect(() => {
    tokenRef.current?.focus()
    tokenRef.current?.select()
    void navigator.clipboard.writeText(token).then(
      () => { setCopied(true); setTimeout(() => setCopied(false), 1800) },
      () => { /* ignore — manual button still works */ },
    )
  }, [token])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(token)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      tokenRef.current?.select()
      toast.error('Không copy được. Hãy nhấn Cmd/Ctrl+C.')
    }
  }

  return (
    <div className="space-y-4 p-5">
      <div className="flex items-start gap-2.5 p-3 rounded-sm bg-warning-bg">
        <ShieldAlert size={15} className="text-warning-text shrink-0 mt-0.5" />
        <div className="text-xs text-warning-text leading-relaxed">
          Đây là <strong>lần duy nhất</strong> token hiển thị. Copy ngay và paste vào field
          “API Token” trong extension, hoặc lưu vào trình quản lý mật khẩu.
          Mất rồi phải tạo cái mới.
        </div>
      </div>

      <div>
        <label className={labelCls}>{name}</label>
        <div className="flex items-stretch gap-2">
          <input
            ref={tokenRef}
            value={token}
            readOnly
            onFocus={(e) => e.currentTarget.select()}
            className="flex-1 px-3 rounded-sm bg-bg text-xs font-mono text-text break-all select-all focus:outline-none border border-border-soft focus:border-accent transition-colors"
          />
          <Button onClick={copy} title="Copy" className="shrink-0">
            {copied ? <Check size={13} className="text-success-text" /> : <Copy size={13} />}
            {copied ? 'Đã copy' : 'Copy'}
          </Button>
        </div>
      </div>

      <label className="flex items-start gap-2 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={confirmedSaved}
          onChange={(e) => onConfirmedSavedChange(e.target.checked)}
          className="mt-0.5 size-3.5 accent-accent cursor-pointer"
        />
        <span className="text-xs text-text-muted leading-relaxed">
          Tôi đã copy và lưu token vào nơi an toàn.
        </span>
      </label>

      <div className="flex justify-end pt-2">
        <Button variant="primary" onClick={onDone} disabled={!confirmedSaved}>
          Đã lưu
        </Button>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/settings')({
  component: SettingsPage,
})
