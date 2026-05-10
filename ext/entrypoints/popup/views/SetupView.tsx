// First-run setup. With the API URL baked in, the only thing we
// collect is the token. Verify against /api/me/projects and write
// the config — the host permission was granted at install time.

import { useState } from 'react'
import { Button } from '@shared/ui/Button'
import { Field, input } from '@shared/ui/Field'
import { TypoonClient } from '@core/typoon'
import { API_URL, PROFILE_KEY } from '@core/config'
import { chromeStorage } from '@shell/adapters/chrome-storage'
import { useConfig } from '@shell/hooks/useConfig'

type Status =
  | { kind: 'idle' }
  | { kind: 'busy' }
  | { kind: 'error'; message: string }

export function SetupView() {
  const { save } = useConfig()

  const [token,  setToken]  = useState('')
  const [status, setStatus] = useState<Status>({ kind: 'idle' })

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    // Strip every whitespace character — clipboard paste from the SPA
    // sometimes brings along a trailing newline or zero-width space
    // that breaks bcrypt verify on the server side.
    const t = token.replace(/\s+/g, '')
    if (!t) {
      setStatus({ kind: 'error', message: 'Cần nhập API token.' })
      return
    }
    if (!t.startsWith('typ_')) {
      setStatus({ kind: 'error', message: 'Token phải bắt đầu bằng "typ_".' })
      return
    }

    setStatus({ kind: 'busy' })
    console.info('[typoon] verifying', { prefix: t.slice(0, 12) + '…', apiUrl: API_URL })
    const client = new TypoonClient({ apiUrl: API_URL, token: t })
    const result = await client.verify()
    if (!result.ok) {
      setStatus({
        kind: 'error',
        message: result.reason === 'unauthorized'
          ? 'Token không hợp lệ. Hãy tạo lại trên trang Cài đặt → API tokens.'
          : `Không kết nối được tới máy chủ${result.detail ? `: ${result.detail}` : ''}`,
      })
      return
    }

    await save({ token: t })
    // Cache profile so popup header + picker overlay can show "logged
    // in as X" without an extra round-trip.
    await chromeStorage.set(PROFILE_KEY, {
      display_name: result.profile.display_name,
      avatar_url:   result.profile.avatar_url,
    })
    setStatus({ kind: 'idle' })
  }

  const busy = status.kind === 'busy'

  return (
    <form onSubmit={onSubmit} className="w-[360px] p-4 space-y-4">
      <header>
        <h1 className="text-sm font-semibold text-text">Kết nối Hội Mê Truyện</h1>
        <p className="text-xs text-text-subtle mt-1">
          Dán API token để extension có thể upload chương vào dự án của bạn.
        </p>
      </header>

      <Field
        label="API token"
        hint={
          <>
            Tạo tại Hội Mê Truyện → Cài đặt → API tokens. Token chỉ
            hiện 1 lần khi tạo, sao chép rồi dán vào đây.
          </>
        }
      >
        <input
          className={input}
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder="typ_…"
          value={token}
          onChange={e => setToken(e.target.value)}
          disabled={busy}
        />
      </Field>

      {status.kind === 'error' && (
        <p className="text-xs text-error-text" role="alert">{status.message}</p>
      )}

      <div className="flex justify-end">
        <Button type="submit" variant="primary" disabled={busy}>
          {busy ? 'Đang xác thực…' : 'Lưu'}
        </Button>
      </div>
    </form>
  )
}
