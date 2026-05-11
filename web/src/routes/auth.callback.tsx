import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'
import { exchangeCode, setLoginError, setToken, verifyState } from '@features/auth/auth'
import { Spinner } from '@shared/ui/primitives'

function CallbackPage() {
  const nav = useNavigate()
  const [status,  setStatus]  = useState<'working' | 'error'>('working')
  const [message, setMessage] = useState('Đang đăng nhập…')

  // React StrictMode runs effects twice in dev. The first run consumes
  // the OAuth `code` (single-use!) and the CSRF state token (also
  // single-use — we delete it from sessionStorage on read). The second
  // run would always fail. Guard with a ref so only the first execution
  // touches the network and storage.
  const ranRef = useRef(false)

  useEffect(() => {
    if (ranRef.current) return
    ranRef.current = true

    const params  = new URLSearchParams(window.location.search)
    const code    = params.get('code')
    const state   = params.get('state')
    const error   = params.get('error')
    const errDesc = params.get('error_description')

    const fail = (msg: string) => {
      setLoginError(msg)
      setStatus('error')
      setMessage(msg)
      setTimeout(() => nav({ to: '/login' }), 800)
    }

    if (error) {
      fail(`Discord: ${errDesc || error}`)
      return
    }
    if (!code) {
      fail('Thiếu mã OAuth (code) từ Discord.')
      return
    }
    if (!verifyState(state)) {
      fail('Phiên đăng nhập đã hết hạn — vui lòng thử lại.')
      return
    }

    exchangeCode(code)
      .then((token) => {
        setToken(token)
        nav({ to: '/projects' })
      })
      .catch((e: Error) => fail(e.message))
  }, [nav])

  return (
    <div className="min-h-screen flex items-center justify-center bg-bg">
      <div className="text-center">
        <div className="flex justify-center mb-3 text-text-subtle">
          <Spinner size={20} />
        </div>
        <p className={`text-sm ${status === 'error' ? 'text-error-text' : 'text-text-muted'}`}>
          {message}
        </p>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/auth/callback')({
  component: CallbackPage,
  staticData: { chrome: 'bare', auth: 'public' },
})
