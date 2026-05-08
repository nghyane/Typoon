import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import {
  Settings as Cog, Key, Plus, Copy, Check, Trash2,
  AlertTriangle,
} from 'lucide-react'
import { api, type ApiTokenInfo } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import { input as inputCls, label as labelCls, Spinner } from '@shared/ui/primitives'
import { toast } from '@shared/ui/Toaster'
import { Modal } from '@shared/ui/Modal'
import { timeAgo } from '@shared/lib/time'

function SettingsPage() {
  return (
    <div className="px-6 py-10 max-w-3xl">
      <div className="flex items-center gap-3 mb-8">
        <div className="size-10 rounded-md bg-surface flex items-center justify-center">
          <Cog size={18} className="text-text-muted" />
        </div>
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text">Cài đặt</h1>
          <p className="text-sm text-text-subtle">Cấu hình tài khoản và công cụ</p>
        </div>
      </div>

      <TokensSection />
    </div>
  )
}

// ── API tokens ──────────────────────────────────────────────────────────────

function TokensSection() {
  const qc = useQueryClient()
  const [createOpen, setCreateOpen] = useState(false)
  const [revealing,  setRevealing]  = useState<{ token: string; name: string } | null>(null)

  const { data: tokens = [], isLoading } = useQuery({
    queryKey: ['tokens'],
    queryFn:  api.listTokens,
  })

  const create = useMutation({
    mutationFn: (name: string) => api.createToken(name),
    onSuccess: (t) => {
      qc.invalidateQueries({ queryKey: ['tokens'] })
      setCreateOpen(false)
      setRevealing({ token: t.token, name: t.name })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const revoke = useMutation({
    mutationFn: (id: number) => api.revokeToken(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tokens'] })
      toast.success('Đã thu hồi')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <section className="rounded-md bg-surface">
      <header className="px-5 py-4 border-b border-border-soft flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Key size={15} className="text-text-muted" />
            <h2 className="text-sm font-semibold text-text">API Tokens</h2>
          </div>
          <p className="text-xs text-text-subtle mt-1 max-w-md leading-relaxed">
            Token để công cụ ngoài (extension, script) kết nối Typoon thay cho login Discord.
            Mỗi tool tạo 1 token riêng, có thể thu hồi bất kỳ lúc nào.
          </p>
        </div>
        <Button variant="primary" onClick={() => setCreateOpen(true)} className="shrink-0">
          <Plus size={13} />
          Tạo token
        </Button>
      </header>

      <div className="divide-y divide-border-soft">
        {isLoading && (
          <div className="px-5 py-8 text-center text-sm text-text-subtle">Đang tải…</div>
        )}
        {!isLoading && tokens.length === 0 && (
          <div className="px-5 py-10 text-center">
            <p className="text-sm text-text-muted">Chưa có token nào</p>
            <p className="text-xs text-text-subtle mt-1">Tạo token đầu tiên để dùng với extension</p>
          </div>
        )}
        {tokens.map((t) => (
          <TokenRow
            key={t.id}
            token={t}
            onRevoke={() => {
              if (confirm(`Thu hồi token "${t.name}"? Tool đang dùng token này sẽ ngừng hoạt động.`))
                revoke.mutate(t.id)
            }}
          />
        ))}
      </div>

      <CreateTokenDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onSubmit={(name) => create.mutate(name)}
        pending={create.isPending}
      />

      <RevealTokenDialog
        data={revealing}
        onClose={() => setRevealing(null)}
      />
    </section>
  )
}

function TokenRow({ token: t, onRevoke }: { token: ApiTokenInfo; onRevoke: () => void }) {
  return (
    <div className="px-5 py-3 flex items-center gap-4">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-text truncate">{t.name}</p>
        <div className="flex items-center gap-3 mt-0.5 text-xs text-text-subtle">
          <code className="font-mono">typ_{t.prefix}…</code>
          {t.last_used ? (
            <span title={t.last_used}>Dùng lần cuối {timeAgo(t.last_used)}</span>
          ) : (
            <span className="text-text-subtle/60">Chưa dùng</span>
          )}
          {t.created_at && (
            <span title={t.created_at} className="text-text-subtle/60">
              · Tạo {timeAgo(t.created_at)}
            </span>
          )}
        </div>
      </div>
      <Button
        variant="ghost"
        size="sm"
        icon
        onClick={onRevoke}
        className="hover:text-error-text shrink-0"
        title="Thu hồi"
      >
        <Trash2 size={14} />
      </Button>
    </div>
  )
}

function CreateTokenDialog({
  open, onClose, onSubmit, pending,
}: {
  open: boolean
  onClose: () => void
  onSubmit: (name: string) => void
  pending: boolean
}) {
  const [name, setName] = useState('')
  return (
    <Modal open={open} onClose={onClose} title="Tạo API token">
      <div className="space-y-4 p-5">
        <div>
          <label className={labelCls}>Tên gợi nhớ</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="VD: Chrome extension, MacBook"
            autoFocus
            className={inputCls}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && name.trim()) onSubmit(name.trim())
            }}
          />
          <p className="text-xs text-text-subtle mt-1.5">
            Chỉ bạn nhìn thấy. Đặt tên để biết token này dùng cho công cụ nào.
          </p>
        </div>
        <div className="flex justify-end gap-2 pt-2">
          <Button onClick={onClose}>Huỷ</Button>
          <Button
            variant="primary"
            onClick={() => onSubmit(name.trim())}
            disabled={!name.trim() || pending}
          >
            {pending ? <Spinner /> : <Plus size={13} />}
            Tạo token
          </Button>
        </div>
      </div>
    </Modal>
  )
}

function RevealTokenDialog({
  data, onClose,
}: {
  data: { token: string; name: string } | null
  onClose: () => void
}) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    if (!data) return
    await navigator.clipboard.writeText(data.token)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <Modal open={!!data} onClose={onClose} title="Token mới">
      {data && (
        <div className="space-y-4 p-5">
          <div className="flex items-start gap-2.5 p-3 rounded-sm bg-warning-bg">
            <AlertTriangle size={15} className="text-warning-text shrink-0 mt-0.5" />
            <div className="text-xs text-warning-text leading-relaxed">
              Đây là <strong>lần duy nhất</strong> token hiển thị. Copy ngay và lưu vào nơi an toàn.
              Mất rồi phải tạo cái mới.
            </div>
          </div>

          <div>
            <label className={labelCls}>{data.name}</label>
            <div className="flex items-center gap-2">
              <code className="flex-1 px-3 py-2 rounded-sm bg-bg text-xs font-mono text-text break-all select-all">
                {data.token}
              </code>
              <Button onClick={copy} title="Copy">
                {copied ? <Check size={13} className="text-success-text" /> : <Copy size={13} />}
                {copied ? 'Đã copy' : 'Copy'}
              </Button>
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <Button variant="primary" onClick={onClose}>Đã lưu</Button>
          </div>
        </div>
      )}
    </Modal>
  )
}

export const Route = createFileRoute('/settings')({
  component: SettingsPage,
})
