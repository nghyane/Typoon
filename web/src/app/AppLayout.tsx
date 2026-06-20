import { useEffect, type ReactNode } from 'react'
import { useNavigate, useMatches, useRouterState } from '@tanstack/react-router'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { BottomNav } from './BottomNav'
import { AdminTopBar } from './AdminTopBar'
import { Toaster } from '@shared/ui/Toaster'
import { ConfirmHost } from '@shared/ui/Confirm'
import { safeReturnTo, useRefreshSession, useSession } from '@features/auth/session'
import { cn } from '@shared/lib/cn'

// =============================================================================
// AppLayout — single source of truth for the page shell.
//
// Each route declares its needs through `staticData`:
//   - `chrome: 'app' | 'admin' | 'bare'`
//        • app   — full app chrome (sidebar+header+bottomnav).
//        • admin — ops workspace: tight header with back-to-app + the
//                  page title, no sidebar / no search / no bottomnav.
//                  The admin dashboard owns its own internal layout.
//        • bare  — route owns its chrome (chapter reader).
//   - `auth:   'public' | 'required'` — whether visiting requires a session.
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

  // Public route (login, oauth callback): render children verbatim, no
  // chrome, no guard. The page owns its full viewport.
  if (auth === 'public') return <>{children}</>

  return <AuthenticatedLayout chrome={chrome}>{children}</AuthenticatedLayout>
}

function AuthenticatedLayout({
  children,
  chrome,
}: {
  children: ReactNode
  chrome: 'app' | 'admin' | 'bare'
}) {
  const nav     = useNavigate()
  const session = useSession()
  const currentHref = useRouterState({ select: s => s.location.href })
  const refreshSession = useRefreshSession()

  // Auth guard — applies only to routes that require a session.
  // We branch on the discriminated status (not on `user`/`loading`
  // separately) so an `error` state doesn't get mistaken for
  // `unauthenticated` and silently bounce the user to /login.
  useEffect(() => {
    if (session.status === 'unauthenticated') {
      nav({ to: '/login', search: { redirect: safeReturnTo(currentHref) } })
    }
  }, [currentHref, session.status, nav])

  // Global 401 → bounce to login. The session module already cleared
  // the token + cache when the unauthorized 401 fired; we just route.
  useEffect(() => {
    const onUnauth = () => {
      nav({ to: '/login', search: { redirect: safeReturnTo(currentHref) } })
    }
    window.addEventListener('typoon:unauthorized', onUnauth)
    return () => window.removeEventListener('typoon:unauthorized', onUnauth)
  }, [currentHref, nav])

  // Auth required but session not authenticated: avoid flashing app
  // chrome around an empty body before the redirect effect fires.
  // While `loading`, render nothing rather than the shell — the
  // session query resolves on every nav, the flicker would be
  // visible on every route switch otherwise.
  if (session.status === 'error') {
    return <AuthErrorScreen message={session.error.message} onRetry={refreshSession} />
  }
  if (session.status !== 'authenticated') return null
  const user = session.user

  // Bare shell: no sidebar/header/bottomnav. The page provides its own
  // toolbar (e.g. ReaderToolbar) and scrolls the document — `min-h-dvh`
  // (not `h-dvh overflow-hidden`) lets `window.scrollY` work, which the
  // reader's auto-hide toolbar relies on.
  if (chrome === 'bare') {
    return (
      <div className="min-h-dvh bg-bg text-text text-sm">
        {children}
        <Toaster />
        <ConfirmHost />
      </div>
    )
  }

  // Admin workspace: no sidebar (manga nav is irrelevant), no global
  // search, no bottomnav. A tight top bar with "← Back to app" + a
  // workspace label keeps the operator anchored. The admin dashboard
  // page owns everything below the bar.
  if (chrome === 'admin') {
    return (
      <div
        className={cn(
          'flex flex-col h-dvh overflow-hidden bg-bg text-text text-sm',
          'pt-[var(--sait)]',
          'pl-[var(--sail)] pr-[var(--sair)]',
        )}
      >
        <AdminTopBar user={user} />
        <main className="flex-1 overflow-auto pb-[var(--saib)]">{children}</main>
        <Toaster />
        <ConfirmHost />
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
      <Sidebar />
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
    </div>
  )
}

function AuthErrorScreen({
  message,
  onRetry,
}: {
  message: string
  onRetry: () => unknown
}) {
  return (
    <div className="min-h-dvh flex items-center justify-center bg-bg text-text p-4">
      <div className="w-full max-w-sm bg-surface border border-border-soft rounded-md p-6 text-center space-y-4">
        <div className="text-sm text-error-text">Không thể kiểm tra đăng nhập: {message}</div>
        <button
          type="button"
          onClick={() => { void onRetry() }}
          className="h-9 px-4 rounded-sm bg-accent text-white text-sm font-medium cursor-pointer"
        >
          Thử lại
        </button>
      </div>
    </div>
  )
}
