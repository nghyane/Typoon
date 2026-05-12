import { useEffect } from 'react'
import { Link } from '@tanstack/react-router'
import { ArrowLeft, AlertTriangle } from 'lucide-react'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { useHeaderStore } from '../../store/header'
import { HubHero } from './HubHero'
import { HubChapterList } from './HubChapterList'
import { useHubData } from './useHubData'

// =============================================================================
// TitleHub — `/title/$entryId` page.
//
// Three vertical sections:
//   ① Hero: cover + title + meta + follow button + activity chips.
//   ② Description: collapsible <details> to keep chapter list near top.
//   ③ Chapter list: per-material primary chapters with translation
//      overlay. Cross-material merge lands in a later slice.
//
// Loading state shows a centred spinner; missing entry returns to /library.
// =============================================================================

interface Props {
  entryId: number
}

export function TitleHub({ entryId }: Props) {
  const { entry, material, loading, error } = useHubData(entryId)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    if (material) {
      setHeader(material.material.title, [{ label: 'Thư viện', to: '/library' }])
    } else {
      setHeader('', [{ label: 'Thư viện', to: '/library' }])
    }
    return () => clearHeader()
  }, [material, setHeader, clearHeader])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }

  if (error || !entry) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được"
          hint={(error as Error)?.message ?? 'Entry không tồn tại hoặc đã bị xoá.'}
        />
      </div>
    )
  }

  if (!material) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Entry chưa có material chính"
          hint="Mở Cài đặt để link một material làm chính cho entry này."
        />
      </div>
    )
  }

  return (
    <div className="pb-16">
      <div className="sm:hidden px-4 pt-4">
        <Link
          to="/library"
          className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text"
        >
          <ArrowLeft size={14} />
          Thư viện
        </Link>
      </div>

      <HubHero entry={entry} material={material.material} />

      {material.material.description && (
        <details className="px-4 sm:px-6 pb-4 group">
          <summary className="text-xs text-text-subtle cursor-pointer hover:text-text-muted list-none flex items-center gap-1.5">
            <span>Mô tả</span>
            <span className="group-open:rotate-180 transition-transform">▾</span>
          </summary>
          <p className="text-sm text-text-muted leading-relaxed whitespace-pre-line mt-2 max-w-2xl">
            {material.material.description}
          </p>
        </details>
      )}

      <div className="px-4 sm:px-6">
        <HubChapterList
          chapters={material.chapters}
          targetLang={entry.target_lang}
        />
      </div>
    </div>
  )
}
