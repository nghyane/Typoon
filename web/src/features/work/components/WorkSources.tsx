// WorkSources — attached sources strip below the hero.
//
// Reads sources + primaryIdx from context. Detach action via
// useWorkActions(). Only the `onAttach` callback comes from the
// parent (it owns the modal state).

import { Plus } from 'lucide-react'

import { Cover } from '@shared/ui/Cover'
import { Spinner } from '@shared/ui/primitives'
import { Menu, type MenuItem } from '@shared/ui/Menu'
import { confirm } from '@shared/ui/Confirm'
import { toast } from '@shared/ui/Toaster'
import { cn } from '@shared/lib/cn'
import { useEnabledSources } from '@features/browse/sources'

import { useWorkIdentity } from '../contexts/WorkIdentityContext'
import { useWorkChapters } from '../contexts/WorkChaptersContext'
import { useWorkActions } from '../contexts/WorkActionsContext'
import type { WorkSource } from '../data/types'


interface Props {
  onAttach: () => void
}


export function WorkSources({ onAttach }: Props) {
  const { work, primaryIdx } = useWorkIdentity()
  const { loading } = useWorkChapters()

  const ordered = work.sources.length > 1 && primaryIdx > 0
    ? [work.sources[primaryIdx]!, ...work.sources.filter((_, i) => i !== primaryIdx)]
    : work.sources

  return (
    <section className="pt-1 pb-3">
      <div className="flex items-center gap-2 mb-2">
        <h2 className="text-xs uppercase tracking-wider text-text-subtle font-medium">
          Nguồn
        </h2>
        <span className="text-xs text-text-subtle tabular-nums">{work.sources.length}</span>
        {loading && <Spinner size={12} />}
      </div>

      <div className="flex flex-wrap gap-2">
        {ordered.map((s, i) => (
          <SourceCard
            key={`${s.source}:${s.upstream_ref}`}
            source={s}
            isPrimary={i === 0 && work.sources.length > 1 && primaryIdx >= 0}
          />
        ))}
        <AttachCard onAttach={onAttach} />
      </div>
    </section>
  )
}


const CARD = 'flex-1 basis-[200px] min-w-[180px] max-w-[320px]'


function SourceCard({ source, isPrimary }: {
  source:     WorkSource
  isPrimary?: boolean
}) {
  const installed   = useEnabledSources()
  const { detachSource } = useWorkActions()
  const manifest    = installed.find(s => s.manifest.id === source.source)?.manifest
  const sourceLabel = manifest?.name ?? source.source
  const langLabel   = (source.languages[0] ?? '').toUpperCase()

  async function handleDetach() {
    const ok = await confirm({
      title:       `Gỡ "${sourceLabel}"?`,
      description: 'Có thể thêm lại sau.',
      confirmText: 'Gỡ',
      tone:        'danger',
    })
    if (!ok) return
    try {
      detachSource(source.source, source.upstream_ref)
      toast.success(`Đã gỡ "${sourceLabel}".`)
    } catch (e) { toast.error((e as Error).message) }
  }

  const menuItems: MenuItem[] = [
    { key: 'detach', label: 'Gỡ nguồn', danger: true, onSelect: handleDetach },
  ]

  return (
    <div className={cn(
      CARD,
      'flex items-center gap-2 h-11 pl-2 pr-1 rounded-sm bg-surface-2',
      isPrimary && 'border-l-2 border-accent',
    )}>
      <div className="w-6 h-8 shrink-0 rounded-xs overflow-hidden">
        <Cover src={source.cover_url} title={source.title} className="w-full h-full" fontSize="text-[10px]" />
      </div>
      <div className="flex-1 min-w-0 text-sm truncate">
        {langLabel && <span className="text-xs text-text-subtle font-medium mr-1.5">{langLabel}</span>}
        <span className="text-text">{sourceLabel}</span>
        {isPrimary && <span className="ml-1.5 text-xs text-accent">Chính</span>}
      </div>
      <Menu
        trigger={<span className="text-sm tracking-[2px] leading-none">···</span>}
        align="end"
        items={menuItems}
      />
    </div>
  )
}


function AttachCard({ onAttach }: { onAttach: () => void }) {
  return (
    <button
      type="button"
      onClick={onAttach}
      className={cn(
        CARD,
        'flex items-center gap-2 h-11 px-2 rounded-sm',
        'text-sm text-text-muted bg-transparent hover:bg-surface-2 hover:text-text',
        'border border-dashed border-border-soft transition-colors cursor-pointer text-left',
      )}
    >
      <span className="w-6 h-8 shrink-0 rounded-xs flex items-center justify-center bg-surface-2">
        <Plus size={12} className="text-text-subtle" />
      </span>
      Thêm nguồn
    </button>
  )
}
