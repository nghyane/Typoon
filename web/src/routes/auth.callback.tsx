import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { exchangeCode, setLoginError, setToken, verifyState } from '../lib/auth'

function CallbackPage() {
  const nav = useNavigate()
  const [status, setStatus] = useState<'working' | 'error'>('working')
  const [message, setMessage] = useState('Đang đăng nhập…')

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const code   = params.get('code')
    const state  = params.get('state')
    const error  = params.get('error')
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
      fail('CSRF state không khớp — phiên đăng nhập đã hết hạn, thử lại.')
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
    <div className="min-h-screen flex items-center justify-center bg-zinc-50">
      <div className="text-center">
        <svg width="20" height="20" viewBox="0 0 24 24" className="mx-auto animate-spin text-zinc-400 mb-3">
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" fill="none" opacity="0.2" />
          <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" fill="none" />
        </svg>
        <p className={`text-sm ${status === 'error' ? 'text-red-600' : 'text-zinc-500'}`}>
          {message}
        </p>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/auth/callback')({
  component: CallbackPage,
})
