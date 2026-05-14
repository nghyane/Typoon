import { Link } from '@tanstack/react-router'
import { ArrowLeft, Shield } from 'lucide-react'

import { Monogram } from '@shared/ui/primitives'
import type { SessionUser } from '@features/auth/session'
import { cn } from '@shared/lib/cn'

// =============================================================================
// AdminTopBar — top bar for the /admin/* workspace.
//
// Deliberately NOT the regular Header: admin work has no business
// sharing a search box, sidebar nav, or pipeline pill with the manga
// app. This bar is just three slots:
//
//   [ ← Quay lại app ]  [ Quản trị pipeline ]  [ avatar ]
//
// The "back to app" link is a verbatim affordance, not a hidden
// gesture — admin always knows they're in a different workspace and
// has one click out. Avatar is read-only (no dropdown); user
// settings live in the main app.
// =============================================================================

export function AdminTopBar({ user }: { user: SessionUser }) {
  return (
    <header className="flex items-center gap-3 px-3 sm:px-5 h-bar bg-bg shrink-0 border-b border-border-soft">
      <Link
        to="/library"
        className="inline-flex items-center gap-2 text-sm text-text-subtle hover:text-text transition-colors"
      >
        <ArrowLeft size={14} />
        <span className="hidden sm:inline">Quay lại app</span>
      </Link>

      <div className="flex-1 min-w-0 flex items-center gap-2 justify-center sm:justify-start sm:pl-4">
        <Shield size={14} className="text-accent-text shrink-0" />
        <span className="text-sm font-semibold text-text tracking-tight truncate">
          Quản trị pipeline
        </span>
      </div>

      <Link
        to="/settings"
        search={{ section: 'account' }}
        title={user.display_name}
        className={cn(
          'inline-flex items-center justify-center size-8 rounded-sm',
          'hover:bg-hover transition-colors cursor-pointer',
        )}
      >
        {user.avatar_url ? (
          <img
            src={user.avatar_url}
            alt=""
            className="size-7 rounded-sm object-cover"
          />
        ) : (
          <Monogram label={user.display_name} />
        )}
      </Link>
    </header>
  )
}
