import { Outlet } from '@tanstack/react-router'
import { Sidebar } from './Sidebar'

export function AppShell() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <main className="flex-1 overflow-y-auto" style={{ background: 'var(--color-bg)' }}>
        <Outlet />
      </main>
    </div>
  )
}
