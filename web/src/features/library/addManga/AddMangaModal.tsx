import { useEffect, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { BookmarkPlus } from 'lucide-react'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { Tag } from '@shared/ui/primitives'
import { toast } from '@shared/ui/Toaster'
import { api, type LibraryStatus } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { useEnabledSources } from '@features/browse/sources'
import { ManualCreateForm } from './ManualCreateForm'
import { SearchPane } from './SearchPane'
import { PickedDetail } from './PickedDetail'
import type { Picked, Mode } from './types'

// =============================================================================
// AddMangaModal — Library entry point.
//
// Pro design (Option B): single column, fixed modal size 'md'. No
// sidebar; no size morphing between modes. The body content changes
// per mode but the column width stays constant — modal never "jumps"
// when the user picks a result or escapes into manual create.
//
// Three modes:
//   ① search  default; SearchPane handles input + scope + results.
//   ② picked  read-only manga card + confirm form.
//   ③ manual  freeform manga create when no source matches.
// =============================================================================

interface Props {
  open:    boolean
  onClose: () => void
}

export function AddMangaModal({ open, onClose }: Props) {
  const allSources = useEnabledSources()

  const [picked,           setPicked]           = useState<Picked | null>(null)
  const [manualSeed,       setManualSeed]       = useState<string | null>(null)
  const [query,            setQuery]            = useState('')

  const [targetLang, setTargetLang] = useState('vi')
  const [status,     setStatus]     = useState<LibraryStatus>('reading')

  useEffect(() => {
    if (!open) return
    setPicked(null)
    setManualSeed(null)
    setQuery('')
    setTargetLang('vi')
    setStatus('reading')
  }, [open])

  const mode: Mode = manualSeed !== null ? 'manual'
                   : picked !== null     ? 'picked'
                                          : 'search'

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Thêm manga vào thư viện"
      size="md"
      footerLeft={<FooterLeft mode={mode} picked={picked} totalSources={allSources.length} />}
      footer={mode === 'picked' && picked ? (
        <ConfirmActions
          picked={picked}
          targetLang={targetLang}
          status={status}
          onCancel={() => setPicked(null)}
          onDone={onClose}
        />
      ) : (
        <Button variant="ghost" onClick={onClose}>Huỷ</Button>
      )}
    >
      <div className="px-5 py-4">
        {mode === 'manual' ? (
          <ManualCreateForm
            initialTitle={manualSeed ?? ''}
            onCancel={() => setManualSeed(null)}
            onCreated={onClose}
          />
        ) : mode === 'picked' && picked ? (
          <PickedDetail
            picked={picked}
            targetLang={targetLang}
            setTargetLang={setTargetLang}
            status={status}
            setStatus={setStatus}
            onChangePick={() => setPicked(null)}
          />
        ) : (
          <SearchPane
            query={query}
            setQuery={setQuery}
            sources={allSources}
            onPick={setPicked}
            onManualCreate={(seed) => setManualSeed(seed)}
          />
        )}
      </div>
    </Modal>
  )
}


function FooterLeft({
  mode, picked, totalSources,
}: {
  mode:         Mode
  picked:      Picked | null
  totalSources: number
}) {
  if (mode === 'manual') return <span>Tạo manga không thuộc nguồn nào</span>
  if (mode === 'picked' && picked) {
    return (
      <span className="inline-flex items-center gap-2 truncate">
        <span className="truncate">{picked.title}</span>
        <Tag tone="info" size="sm">{picked.source.manifest.name}</Tag>
      </span>
    )
  }
  return <span>{totalSources} nguồn đã cài</span>
}


function ConfirmActions({
  picked, targetLang, status, onCancel, onDone,
}: {
  picked:     Picked
  targetLang: string
  status:     LibraryStatus
  onCancel:   () => void
  onDone:     () => void
}) {
  const qc = useQueryClient()
  const m = useMutation({
    mutationFn: async () => {
      const material = await api.importMaterial({
        source:       picked.source.manifest.id,
        upstream_ref: picked.upstreamRef,
        title:        picked.title,
        cover_url:    picked.cover,
        description:  picked.description,
        author:       picked.author,
        status:       picked.status,
        languages:    picked.languages,
        nsfw:         picked.nsfw,
      })
      return await api.createLibraryEntry({
        material_id: material.id,
        target_lang: targetLang,
        title:       picked.title,
        cover_url:   picked.cover ?? null,
        status,
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.library.all() })
      toast.success(`Đã thêm "${picked.title}" vào thư viện`)
      onDone()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <>
      <Button variant="ghost" onClick={onCancel} disabled={m.isPending}>
        Quay lại
      </Button>
      <Button
        variant="primary"
        onClick={() => m.mutate()}
        disabled={m.isPending}
      >
        <BookmarkPlus size={14} />
        Thêm vào thư viện
      </Button>
    </>
  )
}
