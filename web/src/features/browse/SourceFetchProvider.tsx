import { createContext, useContext, useMemo, type ReactNode } from 'react'
import { createSourceFetch, type SourceFetch } from './proxy'
import { usePublicSettings } from '@features/settings/runtime'
import { Spinner } from '@shared/ui/primitives'

const Ctx = createContext<SourceFetch | null>(null)

export function SourceFetchProvider({ children }: { children: ReactNode }) {
  const { data, isLoading } = usePublicSettings()
  const originsKey = (data?.sourceFetch.origins ?? []).join('\n')
  const sourceFetch = useMemo(
    () => createSourceFetch(data?.sourceFetch.origins ?? []),
    [originsKey],
  )

  // Block children until settings resolve so SourceFetch is stable
  // from first paint (no fallback → real switch → double-open reader).
  if (isLoading && !data) {
    return (
      <div className="flex items-center justify-center h-screen bg-bg">
        <Spinner size={20} />
      </div>
    )
  }

  return <Ctx.Provider value={sourceFetch}>{children}</Ctx.Provider>
}

export function useSourceFetch(): SourceFetch {
  return useContext(Ctx) ?? createSourceFetch([])
}
