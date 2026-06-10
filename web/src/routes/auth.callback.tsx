import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useEffect } from 'react'
import { Spinner } from '@shared/ui/primitives'

function CallbackPage() {
  const nav = useNavigate()

  useEffect(() => {
    // Server handles the OAuth callback. Land at / and session query
    // will pick up the cookie.
    nav({ to: '/' })
  }, [nav])

  return (
    <div className="min-h-screen flex items-center justify-center bg-bg">
      <Spinner size={24} />
    </div>
  )
}

export const Route = createFileRoute('/auth/callback')({
  component: CallbackPage,
  staticData: { chrome: 'bare', auth: 'public' },
})
