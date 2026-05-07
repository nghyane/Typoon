import { useEffect, type ReactNode } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { Toaster } from './Toaster'
import { useServerEvents } from '../lib/events'
import { useCurrentUser } from '../lib/auth'

export function AppLayout({ children }: { children: ReactNode }) {
  const nav = useNavigate()
  const { user, loading } = useCurrentUser()

  // Redirect to /login when no token / 401. Listen for runtime 401s from
  // any API call (api.ts dispatches 'typoon:unauthorized').
  useEffect(() => {
    if (!loading && !user) {
      nav({ to: '/login' })
    }
  }, [loading, user, nav])

  useEffect(() => {
    const onUnauth = () => nav({ to: '/login' })
    window.addEventListener('typoon:unauthorized', onUnauth)
    return () => window.removeEventListener('typoon:unauthorized', onUnauth)
  }, [nav])

  // Activate SSE only after we know we're authenticated — events.ts
  // bails out internally if no token, but skipping the call avoids a
  // useless mount cycle.
  useServerEvents()

  if (loading) {
    return <FullScreenSpinner />
  }
  if (!user) {
    // Redirect already kicked; render nothing instead of flashing UI.
    return null
  }

  return (
    <div className="flex h-screen overflow-hidden bg-white text-zinc-950 text-sm">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Header user={user} />
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
      <Toaster />
    </div>
  )
}

function FullScreenSpinner() {
  return (
    <div className="h-screen flex items-center justify-center bg-zinc-50">
      <svg width="20" height="20" viewBox="0 0 24 24" className="animate-spin text-zinc-400">
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" fill="none" opacity="0.2" />
        <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" fill="none" />
      </svg>
    </div>
  )
}
