// WorkPageContent — composition root for the work detail page.
//
// Owns ephemeral UI state (modal open flags) so the route stays a
// pure auth/loading gate. Page sections read data via context hooks
// and call mutations via `useWorkActions()` — no prop drilling.

import { useState } from 'react'

import { WorkHero } from './WorkHero'
import { WorkSources } from './WorkSources'
import { WorkDescription } from './WorkDescription'
import { ChapterList } from './ChapterList'
import { LinkSearchModal } from '../LinkSearchModal'
import { UploadChapterDialog } from '../UploadChapterDialog'
import { useWorkIdentity } from '../contexts/WorkIdentityContext'
import { useWorkChapters } from '../contexts/WorkChaptersContext'


export function WorkPageContent() {
  const { work } = useWorkIdentity()
  const { merged } = useWorkChapters()

  const [attachOpen, setAttachOpen] = useState(false)
  const [uploadOpen, setUploadOpen] = useState(false)

  // Set of existing chapter refs — drives the upload dialog's
  // conflict warning.
  const existingRefs = new Set(merged.map(c => c.numberNorm))

  return (
    <div className="pb-16">
      <WorkHero onUpload={() => setUploadOpen(true)} />
      <WorkSources onAttach={() => setAttachOpen(true)} />
      <WorkDescription />
      <ChapterList />

      <LinkSearchModal
        open={attachOpen}
        onClose={() => setAttachOpen(false)}
        workId={work.id}
        workTitle={work.title}
        ownSources={work.sources}
      />
      <UploadChapterDialog
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        workId={work.id}
        workTitle={work.title}
        sourceLang={work.source_lang}
        existing={existingRefs}
      />
    </div>
  )
}
