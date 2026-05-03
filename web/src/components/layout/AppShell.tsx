import { Outlet } from '@tanstack/react-router'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

export function AppShell() {
  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <TopBar />
        <main className="flex-1 overflow-y-auto" style={{ background: 'var(--color-bg)' }}>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
