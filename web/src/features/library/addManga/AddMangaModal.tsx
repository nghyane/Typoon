import { useEffect, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'

import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { useEnabledSources } from '@features/browse/sources'

import { SearchPane } from './SearchPane'
import { useImportToLibrary } from './useImportToLibrary'

// Single-screen library entry point: search input → results (or URL
// paste, or blank-create fallback). Clicking a result imports + closes
// immediately; the Work hub owns target_lang / status / cover / chapter
// upload afterwards. The blank-create fallback creates an empty Work
// (no material) and navigates the user to it so they can upload the
// first chapter from the hub's `+ Tải lên chương` action.

export function AddMangaModal({
  open, onClose,
}: {
  open:    boolean
  onClose: () => void
}) {
  const allSources = useEnabledSources()
  const nav = useNavigate()
  const [query, setQuery] = useState('')

  useEffect(() => { if (open) setQuery('') }, [open])

  const importer = useImportToLibrary({
    onSuccess: onClose,
    onBlankCreated: (work) => {
      nav({ to: '/w/$workId', params: { workId: String(work.work.id) } })
    },
  })

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Thêm manga vào thư viện"
      size="md"
      footerLeft={
        <span className="text-text-subtle">
          {allSources.length} nguồn đã cài
        </span>
      }
      footer={
        <Button variant="ghost" onClick={onClose} disabled={importer.isPending}>
          Huỷ
        </Button>
      }
    >
      <div className="px-5 py-4">
        <SearchPane
          query={query}
          setQuery={setQuery}
          sources={allSources}
          importer={importer}
        />
      </div>
    </Modal>
  )
}
