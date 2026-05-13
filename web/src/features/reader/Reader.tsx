// Reader — shell that renders the toolbar + the page list. Source
// agnostic: takes a `ReaderSource` shape (built by `useReader`) and
// dispatches to the right body component.

import type { ReaderSource } from './types'
import { ContinuousView } from './ContinuousView'
import { SinglePageView } from './SinglePageView'

export type ViewMode = 'continuous' | 'single'


interface Props {
  source:   ReaderSource
  mode:     ViewMode
  page:     number
  onChange: (page: number) => void
}


export function ReaderBody({ source, mode, page, onChange }: Props) {
  if (mode === 'single') {
    return (
      <SinglePageView
        pages={source.pages}
        urls={source.urls}
        page={page}
        onChange={onChange}
      />
    )
  }
  return <ContinuousView pages={source.pages} urls={source.urls} />
}
