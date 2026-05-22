// WorkPageProvider — composes the three work-page contexts.
//
// Order matters: Actions has no inter-dep, Chapters needs Identity
// (for primary source + manifests), Identity is self-contained.
//
// Route gates loading/error before mounting this provider so children
// can safely assume `work` exists.

import type { ReactNode } from 'react'

import type { Work } from '../data/types'
import { WorkActionsProvider } from './WorkActionsContext'
import { WorkIdentityProvider } from './WorkIdentityContext'
import { WorkChaptersProvider } from './WorkChaptersContext'


interface Props {
  work:     Work
  workId:   string
  children: ReactNode
}


export function WorkPageProvider({ work, workId, children }: Props) {
  return (
    <WorkActionsProvider workId={workId}>
      <WorkIdentityProvider work={work} workId={workId}>
        <WorkChaptersProvider>
          {children}
        </WorkChaptersProvider>
      </WorkIdentityProvider>
    </WorkActionsProvider>
  )
}
