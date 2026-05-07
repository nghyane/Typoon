import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { AlertCircle, ExternalLink } from 'lucide-react'
import {
  buildAuthorizeUrl, fetchAuthConfig, getToken, takeLoginError,
  type AuthConfig,
} from '../lib/auth'

function LoginPage() {
  const nav = useNavigate()
  const [error,  setError]  = useState<string | null>(null)
  const [cfg,    setCfg]    = useState<AuthConfig | null>(null)
  const [busy,   setBusy]   = useState(false)

  useEffect(() => {
    if (getToken()) {
      nav({ to: '/projects' })
      return
    }
    setError(takeLoginError())

    fetchAuthConfig()
      .then(setCfg)
      .catch((e: Error) => setError(`Config error: ${e.message}`))
  }, [nav])

  const onLogin = () => {
    if (!cfg?.discord_client_id) return
    setBusy(true)
    window.location.href = buildAuthorizeUrl(cfg.discord_client_id)
  }

  // The error message from the engine looks like
  // "Bạn cần tham gia Discord 'Name': https://discord.gg/xxx".
  // Pull the URL out so we can render a button instead of inline link.
  const inviteFromError = error ? extractFirstUrl(error) : null
  const errorText       = inviteFromError && error
    ? error.replace(inviteFromError, '').replace(/[:\s]+$/, '').trim()
    : error

  // Always-visible invite (config-supplied) for users who never logged in
  // before but aren't in the guild yet.
  const standingInvite = cfg?.discord_invite_url ?? null

  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-50 p-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-6">
          <div className="size-12 mx-auto rounded-2xl bg-zinc-900 flex items-center justify-center mb-3">
            <svg width="20" height="20" viewBox="0 0 13 13" fill="none">
              <path d="M2 3h9M2 6.5h5.5M2 10h7" stroke="white" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </div>
          <h1 className="text-xl font-bold tracking-tight text-zinc-900">
            {cfg?.guild_name || 'Typoon'}
          </h1>
          <p className="text-sm text-zinc-500 mt-1">Cộng đồng dịch manga</p>
        </div>

        <div className="bg-white border border-zinc-200 rounded-xl shadow-sm p-6">
          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-100 text-sm text-red-700 space-y-2.5">
              <div className="flex items-start gap-2">
                <AlertCircle size={14} className="shrink-0 mt-0.5" />
                <span className="break-words">{errorText}</span>
              </div>
              {inviteFromError && (
                <a
                  href={inviteFromError}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 h-8 px-3 rounded-lg bg-[#5865F2] text-white text-xs font-medium hover:bg-[#4752C4] cursor-pointer"
                >
                  Tham gia Discord
                  <ExternalLink size={11} />
                </a>
              )}
            </div>
          )}

          <p className="text-sm text-zinc-500 mb-4">
            Đăng nhập bằng Discord để tiếp tục.
          </p>

          <button
            onClick={onLogin}
            disabled={!cfg?.discord_client_id || busy}
            className="w-full inline-flex items-center justify-center gap-2 h-10 px-4 rounded-lg bg-[#5865F2] text-white text-sm font-medium hover:bg-[#4752C4] active:scale-[0.98] disabled:opacity-60 disabled:cursor-not-allowed transition-all cursor-pointer"
          >
            <DiscordIcon />
            {busy ? 'Đang chuyển hướng…' : 'Đăng nhập với Discord'}
          </button>

          {cfg?.guild_gated && (
            <p className="text-xs text-zinc-400 mt-4 text-center leading-relaxed">
              {cfg.guild_name
                ? `Yêu cầu là thành viên Discord ${cfg.guild_name}.`
                : 'Yêu cầu là thành viên Discord guild.'}
              {standingInvite && (
                <>
                  {' '}
                  <a
                    href={standingInvite}
                    target="_blank"
                    rel="noreferrer"
                    className="text-zinc-600 underline hover:text-zinc-900"
                  >
                    Tham gia tại đây
                  </a>.
                </>
              )}
            </p>
          )}
        </div>

        <p className="text-xs text-zinc-400 text-center mt-6">
          Self-hosted · Phase 1
        </p>
      </div>
    </div>
  )
}

function extractFirstUrl(text: string): string | null {
  const m = text.match(/https?:\/\/\S+/)
  return m ? m[0] : null
}

function DiscordIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
    </svg>
  )
}

export const Route = createFileRoute('/login')({
  component: LoginPage,
})
