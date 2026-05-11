// =============================================================================
// /settings — 2-column layout with URL-bound section nav.
//
//   ┌───────────────┬──────────────────────────────────────┐
//   │  Rail nav     │  Section content                     │
//   │  • Tài khoản  │  (account / sources / tokens)        │
//   │  • Nguồn      │                                       │
//   │  • Tokens     │                                       │
//   └───────────────┴──────────────────────────────────────┘
//
// Visual rules:
//   - Solid `bg-surface` lists, `border-soft` dividers, no card chrome.
//   - Section header inline with primary action.
//   - Empty / loaded / list states swap *inside* the same list shell.
//   - No emoji; status uses `Badge` (with dot) and `Tag` (labels).
// =============================================================================

import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import {
  Plus, Copy, Check, Trash2, Key, ShieldAlert, Zap,
  UserCircle2, Compass, Power, PowerOff, ExternalLink,
} from 'lucide-react'
import { api, type ApiTokenInfo } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import {
  Spinner, Tag, input as inputCls, label as labelCls,
} from '@shared/ui/primitives'
import {
  SettingsRail, SettingsSection, SettingsList, SettingsListRow,
  useSettingsTab,
  type SettingsTab,
} from '@shared/ui/SettingsForm'
import { DataTable, Th } from '@shared/ui/DataTable'
import { EmptyState } from '@shared/ui/EmptyState'
import { toast } from '@shared/ui/Toaster'
import { Modal } from '@shared/ui/Modal'
import { confirm } from '@shared/ui/Confirm'
import { timeAgo } from '@shared/lib/time'
import { languageSummary, MULTI_LANG } from '@shared/lib/lang'
import { cn } from '@shared/lib/cn'
import {
  useSources, useEnabledSources, bundledManifests,
} from '@features/browse/sources'
import type { InstalledSource } from '@features/browse/manifest/types'

// ── tab registry ────────────────────────────────────────────────────────────

const TABS: SettingsTab[] = [
  { id: 'account', label: 'Tài khoản',   icon: UserCircle2, hint: 'Quota và profile' },
  { id: 'sources', label: 'Nguồn truyện', icon: Compass,     hint: 'Quản lý nguồn duyệt manga' },
  { id: 'tokens',  label: 'API tokens',  icon: Key,         hint: 'Cho extension Typoon' },
]

const TOKEN_NAME_MIN = 2
const TOKEN_NAME_MAX = 40

// ── page shell ──────────────────────────────────────────────────────────────

function SettingsPage() {
  const [active, setActive] = useSettingsTab('account')

  return (
    <div className="max-w-screen-xl mx-auto px-4 sm:px-6 pt-8 pb-20">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-text">Cài đặt</h1>
        <p className="text-sm text-text-subtle mt-1">
          Quản lý tài khoản, nguồn duyệt manga và quyền truy cập.
        </p>
      </header>

      <div className="flex flex-col sm:flex-row gap-6 sm:gap-10">
        <SettingsRail tabs={TABS} active={active} onChange={setActive} />

        <main className="flex-1 min-w-0 max-w-3xl">
          {active === 'account' && <AccountTab />}
          {active === 'sources' && <SourcesTab />}
          {active === 'tokens'  && <TokensTab />}
        </main>
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// 1. Account tab — quota only for now
// ════════════════════════════════════════════════════════════════════════════

function AccountTab() {
  return <QuotaSection />
}

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
          <Zap size={14} className="text-success-text" />
          <span>Không giới hạn</span>
        </div>
      </SettingsSection>
    )
  }

  return (
    <SettingsSection
      title="Quota"
      description="Mỗi lần dịch một chương (upload + start, start, redo) tốn 1 lượt. Reset theo cửa sổ trượt."
    >
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <QuotaMeter label="Hôm nay"     used={data.used_day}  limit={data.limit_day} />
        <QuotaMeter label="Trong giờ"   used={data.used_hour} limit={data.limit_hour} />
        <QuotaMeter label="Đang xử lý"  used={data.in_flight} limit={data.limit_concurrent} />
      </div>
    </SettingsSection>
  )
}

function QuotaMeter({ label, used, limit }: { label: string; used: number; limit: number }) {
  const pct  = limit > 0 ? Math.min(100, Math.round((used / limit) * 100)) : 0
  const tone =
    pct >= 90 ? 'bg-error'
    : pct >= 50 ? 'bg-warning'
    : 'bg-success'
  return (
    <div className="rounded-md bg-surface px-3 py-3">
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] text-text-subtle">{label}</span>
        <span className="text-[13px] tabular">
          <span className="text-text font-medium">{used}</span>
          <span className="text-text-subtle">/{limit}</span>
        </span>
      </div>
      <div className="mt-2 h-1 rounded-full bg-surface-2 overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', tone)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Sources tab — installed sources + (placeholder) install action
// ════════════════════════════════════════════════════════════════════════════

function SourcesTab() {
  // Hydrate bundled manifests on first mount so the list isn't empty
  // for a fresh user.
  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const sources = useEnabledSources()
  const allInstalled = bundledManifests.length

  return (
    <SettingsSection
      title="Nguồn truyện"
      description="Nguồn cung cấp manga cho mục Duyệt. Tắt nguồn không dùng đến để giảm nhiễu khi tìm kiếm. Tính năng cài thêm từ repo cộng đồng đang được phát triển."
      action={
        <Button disabled title="Sắp ra mắt — cài từ repo URL / JSON file">
          <Plus size={13} />
          Thêm nguồn
        </Button>
      }
    >
      {sources.length === 0 && allInstalled === 0 ? (
        <EmptyState
          icon={Compass}
          title="Chưa có nguồn nào"
          hint="Nguồn chính thức sẽ tự động cài khi bạn mở mục Duyệt lần đầu."
        />
      ) : (
        <SettingsList>
          {Object.values(useSources.getState().sources).map((s) => (
            <SourceRow key={s.manifest.id} source={s} />
          ))}
        </SettingsList>
      )}
    </SettingsSection>
  )
}

const ORIGIN_LABEL: Record<InstalledSource['origin'], string> = {
  bundled: 'Chính thức',
  repo:    'Từ repo',
  file:    'Tự cài',
}

function SourceRow({ source }: { source: InstalledSource }) {
  const { manifest, enabled, origin, author } = source
  const setEnabled = useSources((s) => s.setEnabled)
  const remove     = useSources((s) => s.remove)

  const langs = manifest.languages
  const isMulti = langs.length > 1 || langs.includes(MULTI_LANG)
  const langTag = isMulti ? 'MULTI' : (langs[0] ?? '').toUpperCase()

  const askRemove = async () => {
    const ok = await confirm({
      title:       `Gỡ "${manifest.name}"?`,
      description: 'Nguồn không còn xuất hiện trong Duyệt. Bạn có thể cài lại từ kho.',
      confirmText: 'Gỡ',
      tone:        'danger',
    })
    if (ok) remove(manifest.id)
  }

  return (
    <SettingsListRow className="gap-4">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className={cn(
            'text-sm font-medium truncate',
            enabled ? 'text-text' : 'text-text-subtle',
          )}>
            {manifest.name}
          </span>
          {langTag && (
            <Tag tone={isMulti ? 'info' : 'outline'} size="sm" uppercase>
              {langTag}
            </Tag>
          )}
          {manifest.nsfw && (
            <Tag tone="error" size="sm" uppercase>NSFW</Tag>
          )}
        </div>
        <div className="flex items-center gap-2 text-[11px] text-text-subtle">
          <span className="truncate">{manifest.host}</span>
          <span aria-hidden>·</span>
          <span>v{manifest.version}</span>
          <span aria-hidden>·</span>
          <Tag tone="neutral" size="sm">{ORIGIN_LABEL[origin]}</Tag>
          {author && (
            <>
              <span aria-hidden>·</span>
              <span className="truncate">{author}</span>
            </>
          )}
        </div>
      </div>

      {manifest.homepage && (
        <a
          href={manifest.homepage}
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            'inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm text-xs',
            'text-text-subtle hover:text-text hover:bg-hover transition-colors cursor-pointer',
          )}
          title="Mở trang gốc"
        >
          <ExternalLink size={12} />
          <span className="hidden sm:inline">Trang gốc</span>
        </a>
      )}

      <button
        onClick={() => setEnabled(manifest.id, !enabled)}
        title={enabled ? 'Tắt' : 'Bật'}
        aria-label={enabled ? 'Tắt nguồn' : 'Bật nguồn'}
        className={cn(
          'inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm text-xs cursor-pointer transition-colors',
          enabled
            ? 'text-accent-text bg-accent-bg hover:brightness-110'
            : 'text-text-subtle bg-surface-2 hover:bg-hover hover:text-text',
        )}
      >
        {enabled ? <Power size={12} /> : <PowerOff size={12} />}
        <span className="hidden sm:inline">{enabled ? 'Đang bật' : 'Đã tắt'}</span>
      </button>

      {origin !== 'bundled' && (
        <Button
          variant="ghost"
          size="sm"
          icon
          onClick={askRemove}
          title="Gỡ nguồn"
          className="text-text-subtle hover:text-error-text hover:bg-error/10"
        >
          <Trash2 size={13} />
        </Button>
      )}
    </SettingsListRow>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Tokens tab
// ════════════════════════════════════════════════════════════════════════════

function TokensTab() {
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
          Tool đang dùng token này sẽ <strong className="text-text">ngừng hoạt động</strong> ngay.
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
        description="Token cho extension Typoon trên Chrome / Firefox truy cập tài khoản. Mỗi thiết bị nên dùng một token riêng để dễ thu hồi khi mất."
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
                    hint="Tạo token để dùng với extension Typoon. Token chỉ hiện 1 lần sau khi tạo."
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

// ── create token wizard (carried over verbatim — UX is correct) ─────────────

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
  const tooShort  = trimmed.length > 0 && trimmed.length < TOKEN_NAME_MIN
  const tooLong   = trimmed.length > TOKEN_NAME_MAX
  const duplicate = !!trimmed && existing.some((t) => t.name.toLowerCase() === trimmed.toLowerCase())
  const validation =
    tooShort   ? `Tên cần ít nhất ${TOKEN_NAME_MIN} ký tự`
    : tooLong  ? `Tên không quá ${TOKEN_NAME_MAX} ký tự`
    : duplicate ? 'Đã có token với tên này'
    : null
  const canSubmit = !!trimmed && !validation && !create.isPending
  const submit = () => { if (canSubmit) create.mutate(trimmed) }

  const tryClose = async () => {
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
          <span className={`text-[11px] font-mono shrink-0 tabular-nums ${
            remaining < 0 ? 'text-error-text' : 'text-text-subtle/60'
          }`}>
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

  useEffect(() => {
    tokenRef.current?.focus()
    tokenRef.current?.select()
    void navigator.clipboard.writeText(token).then(
      () => { setCopied(true); setTimeout(() => setCopied(false), 1800) },
      () => { /* ignore */ },
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

  // Plain meta — `_ = name` silences unused warning while keeping
  // the prop for future caller flexibility.
  void name

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
        <label className={labelCls}>Token</label>
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

// ── route ────────────────────────────────────────────────────────────────────

interface SearchParams {
  section?: string
}

export const Route = createFileRoute('/settings')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    section: typeof s.section === 'string' ? s.section : undefined,
  }),
  component: SettingsPage,
})

// Silence unused `languageSummary` import — currently helps describe
// multi-language sources; reserved for the source-detail panel.
void languageSummary
