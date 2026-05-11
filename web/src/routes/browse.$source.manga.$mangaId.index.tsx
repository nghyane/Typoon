import { createFileRoute } from '@tanstack/react-router'
import { MangaPage } from '@features/browse/views/MangaPage'
import { useResolvedSource } from '@features/browse/useResolvedSource'

function MangaDetailPage() {
  const { mangaId } = Route.useParams()
  const source = useResolvedSource()
  if (!source) return null
  let mangaUrl = ''
  try { mangaUrl = decodeURIComponent(mangaId) } catch { mangaUrl = mangaId }
  return <MangaPage source={source} mangaUrl={mangaUrl} />
}

export const Route = createFileRoute('/browse/$source/manga/$mangaId/')({
  component: MangaDetailPage,
})
