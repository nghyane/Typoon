import { useEffect, type ReactNode } from 'react'
import { useNavigate, useMatches } from '@tanstack/react-router'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { BottomNav } from './BottomNav'
import { Toaster } from '@shared/ui/Toaster'
import { ConfirmHost } from '@shared/ui/Confirm'
import { useCurrentUser } from '@features/auth/auth'
import { cn } from '@shared/lib/cn'

// =============================================================================
// AppLayout — single source of truth for the page shell.
//
// Each route declares its needs through `staticData`:
//   - `chrome: 'app' | 'bare'`   — full app chrome (sidebar+header+bottomnav)
//                                  vs. bare shell (route owns its own chrome,
//                                  e.g. the chapter reader).
//   - `auth:   'public' | 'required'` — whether visiting requires a session.
//                                       Public routes render children
//                                       directly with no guard or shell.
//
// Defaults: `chrome='app'`, `auth='required'`.
// AppLayout never inspects pathnames. New routes opt in by setting
// staticData; AppLayout stays unchanged.
// =============================================================================

export function AppLayout({ children }: { children: ReactNode }) {
  const matches = useMatches()
  const leaf    = matches[matches.length - 1]
  const chrome  = leaf?.staticData?.chrome ?? 'app'
  const auth    = leaf?.staticData?.auth   ?? 'required'

  const nav = useNavigate()
  const { user, loading } = useCurrentUser()

  // Auth guard — applies only to routes that require a session.
  useEffect(() => {
    if (auth !== 'required') return
    if (!loading && !user) nav({ to: '/login' })
  }, [auth, loading, user, nav])

  // Global 401 → bounce to login.
  useEffect(() => {
    const onUnauth = () => nav({ to: '/login' })
    window.addEventListener('typoon:unauthorized', onUnauth)
    return () => window.removeEventListener('typoon:unauthorized', onUnauth)
  }, [nav])

  // Document title follows the brand. Only set when we have a real name —
  // never fall back to a hardcoded string. The HTML <title> in index.html
  // is a one-time placeholder shown before the SPA boots.
  useEffect(() => {
    if (user?.guild_name) document.title = user.guild_name
  }, [user])

  // Public route (login, oauth callback): render children verbatim, no
  // chrome, no guard. The page owns its full viewport.
  if (auth === 'public') return <>{children}</>

  // Auth required but session not ready: avoid flashing app chrome around
  // an empty body before the redirect effect fires.
  if (!loading && !user) return null

  // Bare shell: no sidebar/header/bottomnav. The page provides its own
  // toolbar (e.g. ReaderToolbar) and scrolls the document — `min-h-dvh`
  // (not `h-dvh overflow-hidden`) lets `window.scrollY` work, which the
  // reader's auto-hide toolbar relies on.
  if (chrome === 'bare') {
    return (
      <div className="min-h-dvh bg-bg text-text text-sm">
        {user && (
          <>
            {children}
            <Toaster />
            <ConfirmHost />
          </>
        )}
      </div>
    )
  }

  // Default: full app chrome.
  //
  // Safe-area: top + sides apply on the shell so Sidebar / Header /
  // Main stay inside the trimmed viewport. Bottom is intentionally
  // NOT padded here — BottomNav handles its own saib so its surface
  // color extends all the way down to the home-indicator instead of
  // leaving a `bg-bg` strip beneath a `bg-surface` bar. Reader
  // (chrome=bare) and floating overlays own their own insets.
  return (
    <div
      className={cn(
        'flex h-dvh overflow-hidden bg-bg text-text text-sm',
        'pt-[var(--sait)]',
        'pl-[var(--sail)] pr-[var(--sair)]',
      )}
    >
      {user && (
        <>
          <Sidebar
            brandName={user.guild_name}
            brandIcon={user.guild_icon_url}
          />
          <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
            <Header user={user} />
            {/* Bottom safe-area: on mobile BottomNav owns the inset
                (its bg-surface fills the home-indicator strip). On
                desktop there's no BottomNav, so Main carries it. */}
            <main className="flex-1 overflow-auto sm:pb-[var(--saib)]">{children}</main>
            <BottomNav />
          </div>
          <Toaster />
          <ConfirmHost />
        </>
      )}
    </div>
  )
}
