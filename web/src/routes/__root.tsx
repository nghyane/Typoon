import { createRootRoute, Outlet } from '@tanstack/react-router'
import { useInvalidateOn } from '../hooks/useInvalidateOn'
import { useToastStore } from '../stores/toast'
import { cn } from '../lib/cn'

function RootLayout() {
  useInvalidateOn()
  const { toasts, remove } = useToastStore()

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <Outlet />

      {/* Toast container */}
      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50">
        {toasts.map((t) => (
          <div
            key={t.id}
            onClick={() => remove(t.id)}
            className={cn(
              'px-4 py-2 rounded-lg text-sm cursor-pointer shadow-lg',
              t.type === 'success' && 'bg-green-600 text-white',
              t.type === 'error'   && 'bg-red-600 text-white',
              t.type === 'info'    && 'bg-blue-600 text-white',
            )}
          >
            {t.message}
          </div>
        ))}
      </div>
    </div>
  )
}

export const Route = createRootRoute({ component: RootLayout })
