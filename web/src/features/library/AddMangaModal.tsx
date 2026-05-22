// AddManga modal — single-screen smart input.
//
// One input box, three behaviors:
//   • URL paste     → host-match badge inline, auto-fetch detail, 1-click add
//   • Text query    → fanout search across enabled sources, scoped tabs
//   • Empty / short → source roster as toggle chips
//
// Clicking a result imports + closes immediately. The blank-create
// fallback (BlankCreateRow + UrlPasteCard "Tạo trống thay") creates an
// empty Work and navigates the user to its hub so they can upload the
// first chapter from there.

import { useEffect, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'

import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { useEnabledSources } from '@features/browse/sources'

import { SearchPane } from './addManga/SearchPane'
import { useImportToLibrary } from './addManga/useImportToLibrary'


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
      nav({ to: '/w/$workId', params: { workId: work.id }, search: { tab: undefined } })
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
