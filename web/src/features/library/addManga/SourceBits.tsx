import { Search, Link as LinkIcon } from 'lucide-react'
import { cn } from '@shared/lib/cn'

// Source favicon via Google's S2 endpoint (same resolver Chrome's URL
// bar uses). onError falls back to a single-letter tile so a missing
// favicon never shows a broken-image glyph.
const FAVICON = (host: string) =>
  `https://www.google.com/s2/favicons?domain=${encodeURIComponent(host)}&sz=64`

export function Favicon({
  host, size = 16,
}: {
  host: string
  size?: number
}) {
  return (
    <span
      className="rounded-xs bg-surface-2 overflow-hidden flex items-center justify-center shrink-0"
      style={{ width: size, height: size }}
    >
      <img
        src={FAVICON(host)}
        alt=""
        width={size}
        height={size}
        loading="lazy"
        onError={(e) => {
          const el = e.currentTarget
          el.style.display = 'none'
          if (el.parentElement) {
            el.parentElement.classList.add(
              'text-[9px]', 'font-bold', 'text-text-muted',
            )
            el.parentElement.textContent = host[0]?.toUpperCase() ?? '?'
          }
        }}
        className="w-full h-full object-contain"
      />
    </span>
  )
}


// Capability indicator pills — used in sidebar rows and dropdown
// options. Active pill: tinted background. Inactive pill: rendered
// at low opacity so the user can scan vertically and tell which
// sources support which path without a tooltip.
export function CapabilityPills({
  searchable,
  size = 'sm',
}: {
  searchable: boolean
  size?: 'sm' | 'md'
}) {
  const dim = size === 'sm' ? 'size-4' : 'size-5'
  const icon = size === 'sm' ? 9 : 10
  return (
    <span className="inline-flex items-center gap-0.5 shrink-0">
      <Pill
        active={searchable}
        title={searchable ? 'Hỗ trợ tìm theo tên' : 'Chưa hỗ trợ tìm'}
        dim={dim}
      >
        <Search size={icon} />
      </Pill>
      <Pill active={true} title="Hỗ trợ dán đường dẫn" dim={dim}>
        <LinkIcon size={icon} />
      </Pill>
    </span>
  )
}

function Pill({
  active, title, dim, children,
}: {
  active:   boolean
  title:    string
  dim:      string
  children: React.ReactNode
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center justify-center rounded-xs',
        dim,
        active
          ? 'bg-bg/40 text-text-muted'
          : 'text-text-subtle/30',
      )}
    >
      {children}
    </span>
  )
}
