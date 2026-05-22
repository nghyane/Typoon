// LinkSearchModal — manual cross-source attach for a Work.
//
// Multi-pick lifecycle: each picked candidate fetches detail, attaches
// to the Work, marks the row as "đã liên kết", and stays in the modal
// so the user can chain multiple attachments in one session. Closing
// the modal is the only exit.
//
// Failures surface as toast errors per row. Duplicate (source,
// upstream_ref) pairs are no-ops in the store, so re-pick is safe.

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'

import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { toast } from '@shared/ui/Toaster'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { hitKey } from '@features/library/addManga/hitKey'
import type { SearchHit } from '@features/library/addManga/fanoutSearch'
import {
  useAttachSource, type WorkSource,
} from '@features/works/queries'

import { LinkSearchPane } from './LinkSearchPane'


interface Props {
  open:    boolean
  onClose: () => void
  workId:    string
  workTitle: string
  /** Sources already on this Work — feeds the self-link filter so the
   *  user never sees a row for a source they've already attached. */
  ownSources: WorkSource[]
}


export function LinkSearchModal({
  open, onClose, workId, workTitle, ownSources,
}: Props) {
  const attach = useAttachSource()
  const [pendingKey, setPendingKey] = useState<string | null>(null)
  const [pickedKeys, setPickedKeys] = useState<Set<string>>(() => new Set())

  // Reset per-session state on every fresh open.
  const handleClose = () => {
    setPendingKey(null)
    setPickedKeys(new Set())
    onClose()
  }

  const link = useMutation({
    mutationFn: async (hit: SearchHit) => {
      // Resolve canonical detail before attaching; falls back to the
      // hit snapshot on fetch error so a flaky source doesn't block
      // the link.
      const detail = await fetchMangaDetail(hit.source.manifest, hit.manga.url)
        .catch(() => null)
      const manifest = hit.source.manifest
      const source: WorkSource = {
        source:       manifest.id,
        upstream_ref: hit.manga.url,
        title:        detail?.title  ?? hit.manga.title,
        cover_url:    detail?.cover  ?? hit.manga.cover ?? null,
        languages:    detail?.availableLanguages
                  ?? manifest.languages
                  ?? [],
        added_at:     new Date().toISOString(),
      }
      await attach.mutateAsync({ work_id: workId, source })
      return { hit, title: source.title }
    },
    onSuccess: ({ hit, title }) => {
      setPickedKeys((prev) => {
        const next = new Set(prev)
        next.add(hitKey(hit))
        return next
      })
      toast.success(`Đã liên kết "${title}"`)
    },
    onError:   (e: Error) => toast.error(`Liên kết thất bại: ${e.message}`),
    onSettled: () => setPendingKey(null),
  })

  const handlePick = (hit: SearchHit) => {
    if (link.isPending) return
    setPendingKey(hitKey(hit))
    link.mutate(hit)
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Liên kết nguồn"
      size="md"
      footerLeft={
        <span className="text-text-subtle truncate">
          Liên kết vào: {workTitle}
          {pickedKeys.size > 0 && ` · đã thêm ${pickedKeys.size}`}
        </span>
      }
      footer={
        <Button variant="ghost" onClick={handleClose} disabled={link.isPending}>
          Đóng
        </Button>
      }
    >
      <div className="px-5 py-4">
        <LinkSearchPane
          ownSources={ownSources}
          initialQuery={workTitle}
          onPick={handlePick}
          pickedKeys={pickedKeys}
          busy={link.isPending}
          pendingKey={pendingKey}
        />
      </div>
    </Modal>
  )
}
