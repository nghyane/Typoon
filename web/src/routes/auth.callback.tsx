import { Link, createFileRoute, redirect } from '@tanstack/react-router'
import { Spinner } from '@shared/ui/primitives'
import { safeReturnTo } from '@features/auth/session'
import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'

type CallbackSearch = {
  code?: string
  state?: string
}

function parseCallbackSearch(search: Record<string, unknown>): CallbackSearch {
  return {
    code:  typeof search.code === 'string' ? search.code : undefined,
    state: typeof search.state === 'string' ? search.state : undefined,
  }
}

async function exchangeDiscordCallback(code: string, state: string): Promise<string> {
  const res = await fetch('/api/auth/discord/callback', {
    method:      'POST',
    credentials: 'include',
    headers:     { 'Content-Type': 'application/json' },
    body:        JSON.stringify({ code, state }),
  })

  if (!res.ok) {
    let msg = `Lỗi ${res.status}`
    try {
      const data = await res.json()
      msg = data.error?.message ?? data.error ?? msg
    } catch { /* not JSON */ }
    throw new Error(msg)
  }

  const data = await res.json() as { returnTo?: string }
  return safeReturnTo(data.returnTo)
}

type CallbackData = { status: 'error'; message: string }

function CallbackPage() {
  const data = Route.useLoaderData()

  if (data.status === 'error') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg p-4">
        <div className="bg-surface rounded-md p-6 border border-border-soft text-center space-y-4 w-full max-w-sm">
          <div className="text-sm text-error-text">Lỗi đăng nhập: {data.message}</div>
          <Link
            to="/login"
            search={{ error: data.message }}
            replace
            className="px-4 py-2 rounded-sm bg-[#5865F2] text-white text-sm hover:bg-[#4752C4]"
          >
            Quay lại trang đăng nhập
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-bg">
      <Spinner size={24} />
    </div>
  )
}

export const Route = createFileRoute('/auth/callback')({
  validateSearch: parseCallbackSearch,
  loaderDeps: ({ search }) => search,
  loader: async ({ context, deps }): Promise<CallbackData> => {
    if (!deps.code || !deps.state) {
      return { status: 'error', message: 'Thiếu code hoặc state trong callback.' }
    }

    let returnTo: string
    try {
      returnTo = await exchangeDiscordCallback(deps.code, deps.state)
      const user = await api.getSession()
      context.queryClient.clear()
      context.queryClient.setQueryData(qk.session.self(), user)
    } catch (err) {
      return {
        status:  'error',
        message: err instanceof Error ? err.message : 'Không thể hoàn tất đăng nhập.',
      }
    }

    throw redirect({ to: returnTo, replace: true })
  },
  component: CallbackPage,
  staticData: { chrome: 'bare', auth: 'public' },
})
