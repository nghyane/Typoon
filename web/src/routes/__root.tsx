import { createRootRoute } from '@tanstack/react-router'
import { useToastStore } from '../stores/toast'
import { useInvalidateOn } from '../hooks/useInvalidateOn'
import { AppShell } from '../components/layout/AppShell'
import { cn } from '../lib/cn'

function RootLayout() {
  useInvalidateOn()
  const { toasts, remove } = useToastStore()

  return (
    <>
      <AppShell />
      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50">
        {toasts.map((t) => (
          <div
            key={t.id}
            onClick={() => remove(t.id)}
            className={cn(
              'px-4 py-2.5 rounded-xl text-sm font-medium cursor-pointer shadow-sm border',
              t.type === 'success' && 'bg-white border-green-100 text-green-700',
              t.type === 'error'   && 'bg-white border-red-100 text-red-600',
              t.type === 'info'    && 'bg-white border-blue-100 text-blue-600',
            )}
          >
            {t.message}
          </div>
        ))}
      </div>
    </>
  )
}

export const Route = createRootRoute({ component: RootLayout })
