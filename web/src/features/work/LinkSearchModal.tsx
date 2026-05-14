// LinkSearchModal — manual cross-source link search, in a modal.
//
// Owns the multi-pick mutation lifecycle: each picked candidate
// imports + force-links, marks the row as "đã liên kết", invalidates
// the Work queries. The modal stays open so the user can chain
// multiple links in one session — closing it (or a merge that
// dissolves the current Work id) is the only exit. cross_refs
// conflicts surface as toast errors per row; cross-language sibling
// merges navigate to the canonical Work and let the modal unmount
// naturally.

import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'

import { api, type ApiMaterial } from '@shared/api/api'
import { Modal } from '@shared/ui/Modal'
import { Button } from '@shared/ui/Button'
import { qk } from '@shared/api/keys'
import { toast } from '@shared/ui/Toaster'
import { hitKey } from '@features/library/addManga/hitKey'
import type { SearchHit } from '@features/library/addManga/fanoutSearch'
import { importMaterialFromHit } from '@features/material/import'

import { LinkSearchPane } from './LinkSearchPane'


interface Props {
  open:    boolean
  onClose: () => void
  /** Work to link into. */
  workId:    number
  workTitle: string
  /** Materials on this Work — feeds the self-link filter, anchors
   *  the vote pair. */
  ownMaterials: ApiMaterial[]
}


export function LinkSearchModal({
  open, onClose, workId, workTitle, ownMaterials,
}: Props) {
  const qc  = useQueryClient()
  const nav = useNavigate()

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
      // Funnel through the shared importer so manual cross-source
      // linking writes the SAME ImportBody shape as "thêm vào thư
      // viện" — including `languages` from the manifest fallback.
      // Bypass-by-hand caused a real bug where Otruyen materials
      // landed with `languages={}` and the title resolver fell
      // through to a Romaji sibling.
      const m = await importMaterialFromHit(hit)
      const own = ownMaterials[0]
      if (!own) throw new Error('Work has no own material to vote with.')
      const res = await api.forceWorkLink(workId, {
        target_material_id: m.id,
        own_material_id:    own.id,
      })
      return { res, hit, importedTitle: m.title }
    },
    onSuccess: ({ res, hit, importedTitle }) => {
      qc.invalidateQueries({ queryKey: qk.work.byId(workId) })
      qc.invalidateQueries({ queryKey: qk.work.linkSuggest(workId) })
      qc.invalidateQueries({ queryKey: qk.work.members(workId) })
      qc.invalidateQueries({ queryKey: qk.library.all() })

      // Cross-Work merge → the current Work dissolved into a sibling.
      // Navigate there; the modal unmounts with the route.
      if (res.merged && res.canonical_work_id != null
          && res.canonical_work_id !== workId) {
        toast.success(`Đã gộp với "${importedTitle}"`)
        nav({
          to:     '/w/$workId',
          params: { workId: String(res.canonical_work_id) },
        })
        return
      }

      setPickedKeys((prev) => {
        const next = new Set(prev)
        next.add(hitKey(hit))
        return next
      })

      if (res.merged) {
        toast.success(`Đã liên kết "${importedTitle}"`)
        return
      }
      if (res.blocked_reason === 'cross_refs_conflict') {
        toast.error(
          `Không thể gộp "${importedTitle}" — hai nguồn khai báo định danh khác nhau.`,
        )
        return
      }
      toast.success(`Đã ghi nhận liên kết "${importedTitle}"`)
    },
    onError: (e: Error) => toast.error(`Liên kết thất bại: ${e.message}`),
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
      title="Liên kết manga"
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
          ownMaterials={ownMaterials}
          onPick={handlePick}
          pickedKeys={pickedKeys}
          busy={link.isPending}
          pendingKey={pendingKey}
        />
      </div>
    </Modal>
  )
}
