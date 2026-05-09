import { useEffect, type ReactNode } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { Toaster } from '@shared/ui/Toaster'
import { useServerEvents } from '@shared/lib/events'
import { useCurrentUser } from '@features/auth/auth'
import { useProjectInterestList } from '../store/interest'

export function AppLayout({ children }: { children: ReactNode }) {
  const nav = useNavigate()
  const { user, loading } = useCurrentUser()
  const projectIds = useProjectInterestList()

  useEffect(() => {
    if (!loading && !user) nav({ to: '/login' })
  }, [loading, user, nav])

  useEffect(() => {
    const onUnauth = () => nav({ to: '/login' })
    window.addEventListener('typoon:unauthorized', onUnauth)
    return () => window.removeEventListener('typoon:unauthorized', onUnauth)
  }, [nav])

  useServerEvents(projectIds)

  // Document title follows the brand. Only set when we have a real name —
  // never fall back to a hardcoded string. The HTML <title> in index.html
  // is a one-time placeholder shown before the SPA boots.
  useEffect(() => {
    if (user?.guild_name) document.title = user.guild_name
  }, [user])

  // No FullScreenSpinner: it caused a hard transition (centered spinner →
  // full chrome) on every reload. Render the layout shell immediately and
  // let `useDelayedFlag` skeletons in routes signal in-flight state.
  // While `loading && !user`, children rely on react-query placeholders
  // and the shell renders with empty user slots.
  if (!loading && !user) return null

  return (
    <div className="flex h-screen overflow-hidden bg-bg text-text text-sm">
      {user && (
        <>
          <Sidebar
            brandName={user.guild_name}
            brandIcon={user.guild_icon_url}
          />
          <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
            <Header user={user} />
            <main className="flex-1 overflow-auto">{children}</main>
          </div>
          <Toaster />
        </>
      )}
    </div>
  )
}