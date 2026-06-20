// /settings — account, reader preferences, offline storage.

import { useEffect, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { Globe2, LogOut, Database, HardDrive, Sparkles } from 'lucide-react'
import { useSession, useSignOut, useUpdatePreferredLang } from '@features/auth/session'
import { useArchiveStorageStats } from '@features/reader/archives'
import { useLocalSettings, useUpdateLocalSettings } from '@features/settings/local'
import { useAllSources, useSources } from '@features/browse/sources'
import { languageSummary } from '@shared/lib/lang'
import { Spinner, Field, input as inputCls } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'


function SettingsPage() {
  const session  = useSession()
  const signOut  = useSignOut()
  const updPref  = useUpdatePreferredLang()
  const stats    = useArchiveStorageStats()
  const local    = useLocalSettings()
  const updLocal = useUpdateLocalSettings()
  const sources  = useAllSources()
  const ensureSources = useSources(s => s.ensureBundled)
  const setSourceEnabled = useSources(s => s.setEnabled)

  const [tab, setTab] = useState<'account' | 'sources' | 'reader' | 'translation' | 'storage'>('account')

  useEffect(() => {
    ensureSources()
  }, [ensureSources])

  if (session.status === 'loading' || !session.user) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner size={20} />
      </div>
    )
  }

  const user = session.user
  const targetLang = user.preferred_target_lang ?? local.data?.default_target_lang ?? 'vi'
  const updateTargetLang = (lang: string) => {
    updPref(lang)
    updLocal.mutate({ default_target_lang: lang })
  }
  const enabledSourceCount = sources.filter(s => s.enabled).length

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      <header>
        <h1 className="text-lg font-semibold text-text">Cài đặt</h1>
      </header>

      {/* Tabs */}
      <div className="flex flex-wrap gap-1.5" role="tablist">
        {[
          { id: 'account', label: 'Tài khoản' },
          { id: 'sources', label: `Nguồn (${enabledSourceCount}/${sources.length})` },
          { id: 'reader',  label: 'Trình đọc' },
          { id: 'translation', label: 'Dịch' },
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
                  ? 'h-7 px-3 rounded-full text-xs font-medium bg-accent-bg text-accent-text'
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
              {user.tier && (
                <span className="rounded-full bg-surface-2 px-2 py-1 text-xs text-text-subtle">
                  {user.tier.name}
                </span>
              )}
            </div>
            <Button variant="ghost" size="sm" onClick={() => signOut()}>
              <LogOut size={14} /> Đăng xuất
            </Button>
          </div>

          <Field label="Ngôn ngữ đích mặc định">
            <select
              className={inputCls}
              value={targetLang}
              onChange={e => updateTargetLang(e.target.value)}
            >
              <option value="vi">Tiếng Việt</option>
              <option value="en">Tiếng Anh</option>
            </select>
          </Field>
        </section>
      )}

      {tab === 'sources' && (
        <section className="space-y-3">
          <div className="rounded-md bg-surface p-4 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-sm">
                <Globe2 size={14} className="text-accent" />
                <span className="font-medium text-text">Nguồn truyện</span>
              </div>
              <Button variant="secondary" size="sm" onClick={() => ensureSources()}>
                Cài nguồn mặc định
              </Button>
            </div>
            <p className="text-xs text-text-muted">
              Bật ít nhất một nguồn để tìm và thêm truyện. Nguồn mặc định đi kèm ứng dụng, không cần cài ngoài.
            </p>

            {sources.length === 0 ? (
              <div className="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
                <p className="text-sm text-text-muted">Chưa cài nguồn nào</p>
                <p className="text-xs text-text-subtle mt-1">Bấm “Cài nguồn mặc định” để khôi phục danh sách nguồn.</p>
              </div>
            ) : (
              <ul className="space-y-2">
                {sources.map(source => (
                  <li
                    key={source.manifest.id}
                    className="flex items-center justify-between gap-3 rounded-md bg-surface-2 px-3 py-2"
                  >
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-text truncate">{source.manifest.name}</div>
                      <div className="text-xs text-text-muted truncate">
                        {source.manifest.host} · {languageSummary(source.manifest.languages)} · {source.origin}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setSourceEnabled(source.manifest.id, !source.enabled)}
                      className={source.enabled
                        ? 'h-7 rounded-full bg-accent-bg px-3 text-xs font-medium text-accent-text'
                        : 'h-7 rounded-full bg-surface px-3 text-xs font-medium text-text-muted hover:bg-hover hover:text-text'}
                    >
                      {source.enabled ? 'Đang bật' : 'Đã tắt'}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      )}

      {tab === 'reader' && local.data && (
        <section className="space-y-4">
          <Field label="Kiểu đọc mặc định">
            <select
              className={inputCls}
              value={local.data.reader_mode}
              onChange={e => updLocal.mutate({ reader_mode: e.target.value as 'standard' | 'rtl' | 'vertical' | 'webtoon' })}
            >
              <option value="webtoon">Cuộn dọc</option>
              <option value="rtl">Manga phải sang trái</option>
              <option value="standard">Từng trang trái sang phải</option>
              <option value="vertical">Từng trang dọc</option>
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

      {tab === 'translation' && (
        <section className="space-y-3">
          <div className="rounded-md bg-surface p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <Sparkles size={14} className="text-accent" />
              <span className="font-medium text-text">Dịch trong reader</span>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <Field label="Ngôn ngữ đích">
                <select
                  className={inputCls}
                  value={targetLang}
                  onChange={e => updateTargetLang(e.target.value)}
                >
                  <option value="vi">Tiếng Việt</option>
                  <option value="en">Tiếng Anh</option>
                </select>
              </Field>
              <div className="rounded-md bg-surface-2 px-3 py-2">
                <div className="text-xs text-text-subtle">Trạng thái</div>
                <div className="mt-1 inline-flex items-center gap-1.5 text-sm font-medium text-accent-text">
                  <Sparkles size={13} /> Realtime
                </div>
              </div>
            </div>
          </div>
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
              <div>Tổng: {(stats.total_bytes / 1_000_000).toFixed(1)} MB · {stats.count} gói offline</div>
              <div>Raw offline: {stats.raw}</div>
            </div>
          </div>

          <div className="rounded-md bg-surface p-4 space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Database size={14} className="text-text-subtle" />
              <span className="font-medium text-text">Dữ liệu dịch realtime</span>
            </div>
            <p className="text-xs text-text-muted">Cache bản dịch trong reader.</p>
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
