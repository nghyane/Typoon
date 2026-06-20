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
import { useWorkIdentity } from '../contexts/WorkIdentityContext'


export function WorkPageContent() {
  const { work } = useWorkIdentity()

  const [attachOpen, setAttachOpen] = useState(false)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 pb-16">
      <WorkHero />
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
    </div>
  )
}
