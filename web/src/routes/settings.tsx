// /settings — account, tier, preferences, offline storage.

import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { LogOut, Database, HardDrive } from 'lucide-react'
import { useSession, useSignOut, useUpdatePreferredLang } from '@features/auth/session'
import { useQuota } from '@features/jobs/useQuota'
import { QuotaMeter } from '@features/jobs/QuotaMeter'
import { TierBadge } from '@features/jobs/TierBadge'
import { useArchiveStorageStats } from '@features/reader/archives'
import { useLocalSettings, useUpdateLocalSettings } from '@features/settings/local'
import { Spinner, Field, input as inputCls } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'


function SettingsPage() {
  const session  = useSession()
  const signOut  = useSignOut()
  const quota    = useQuota()
  const updPref  = useUpdatePreferredLang()
  const stats    = useArchiveStorageStats()
  const local    = useLocalSettings()
  const updLocal = useUpdateLocalSettings()

  const [tab, setTab] = useState<'account' | 'reader' | 'storage'>('account')

  if (session.status === 'loading' || !session.user) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner size={20} />
      </div>
    )
  }

  const user = session.user

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      <header>
        <h1 className="text-lg font-semibold text-text">Cài đặt</h1>
      </header>

      {/* Tabs */}
      <div className="flex flex-wrap gap-1.5" role="tablist">
        {[
          { id: 'account', label: 'Tài khoản' },
          { id: 'reader',  label: 'Trình đọc' },
          { id: 'storage', label: 'Lưu trữ'   },
        ].map(t => {
          const sel = t.id === tab
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={sel}
              onClick={() => setTab(t.id as typeof tab)}
              className={
                sel
                  ? 'h-7 px-3 rounded-full text-xs font-medium bg-accent text-accent-text'
                  : 'h-7 px-3 rounded-full text-xs font-medium bg-surface text-text-muted hover:bg-hover hover:text-text'
              }
            >
              {t.label}
            </button>
          )
        })}
      </div>

      {tab === 'account' && (
        <section className="space-y-6">
          <div className="rounded-md bg-surface p-4 space-y-3">
            <div className="flex items-center gap-3">
              {user.avatar_url && (
                <img
                  src={user.avatar_url}
                  alt={user.display_name}
                  className="size-10 rounded-full bg-surface-2"
                />
              )}
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium text-text truncate">{user.display_name}</div>
                {user.email && (
                  <div className="text-xs text-text-muted truncate">{user.email}</div>
                )}
              </div>
              <TierBadge tier={user.tier} />
            </div>
            <Button variant="ghost" size="sm" onClick={() => signOut()}>
              <LogOut size={14} /> Đăng xuất
            </Button>
          </div>

          {quota.data && (
            <div className="rounded-md bg-surface p-4">
              <QuotaMeter quota={quota.data} />
            </div>
          )}

          <Field label="Ngôn ngữ đích mặc định">
            <select
              className={inputCls}
              value={user.preferred_target_lang ?? ''}
              onChange={e => updPref(e.target.value || null)}
            >
              <option value="">— Không đặt mặc định —</option>
              <option value="vi">Tiếng Việt</option>
              <option value="en">Tiếng Anh</option>
            </select>
          </Field>
        </section>
      )}

      {tab === 'reader' && local.data && (
        <section className="space-y-4">
          <Field label="Chế độ đọc mặc định">
            <select
              className={inputCls}
              value={local.data.reader_mode}
              onChange={e => updLocal.mutate({ reader_mode: e.target.value as 'pager' | 'strip' })}
            >
              <option value="pager">Từng trang (Pager)</option>
              <option value="strip">Cuộn dọc (Strip)</option>
            </select>
          </Field>

          <Field label="Giao diện">
            <select
              className={inputCls}
              value={local.data.theme}
              onChange={e => updLocal.mutate({ theme: e.target.value as 'system' | 'light' | 'dark' })}
            >
              <option value="system">Theo hệ thống</option>
              <option value="dark">Tối</option>
              <option value="light">Sáng</option>
            </select>
          </Field>
        </section>
      )}

      {tab === 'storage' && stats && (
        <section className="space-y-3">
          <div className="rounded-md bg-surface p-4 space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <HardDrive size={14} className="text-text-subtle" />
              <span className="font-medium text-text">Bộ nhớ thiết bị</span>
            </div>
            <div className="text-xs text-text-muted space-y-1">
              <div>Tổng: {(stats.total_bytes / 1_000_000).toFixed(1)} MB · {stats.count} archive</div>
              <div>Đã dịch: {stats.translated} · Raw offline: {stats.raw}</div>
            </div>
          </div>

          <div className="rounded-md bg-surface p-4 space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Database size={14} className="text-text-subtle" />
              <span className="font-medium text-text">Server (R2)</span>
            </div>
            <p className="text-xs text-text-muted">
              File trên server tự động xoá sau 7 ngày. Lưu offline trên thiết bị
              để giữ vĩnh viễn (chiếm bộ nhớ trình duyệt).
            </p>
          </div>
        </section>
      )}
    </div>
  )
}

export const Route = createFileRoute('/settings')({
  component: SettingsPage,
  staticData: { auth: 'required' },
})
