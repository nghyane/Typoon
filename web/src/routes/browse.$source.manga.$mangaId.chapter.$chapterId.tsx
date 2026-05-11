import { createFileRoute } from '@tanstack/react-router'
import { BrowseReader } from '@features/browse/views/BrowseReader'
import { useResolvedSource } from '@features/browse/useResolvedSource'

function ChapterReaderPage() {
  const { mangaId, chapterId } = Route.useParams()
  const source = useResolvedSource()
  if (!source) return null
  let mangaUrl   = ''
  let chapterUrl = ''
  try { mangaUrl   = decodeURIComponent(mangaId)   } catch { mangaUrl   = mangaId }
  try { chapterUrl = decodeURIComponent(chapterId) } catch { chapterUrl = chapterId }
  return <BrowseReader source={source} mangaUrl={mangaUrl} chapterUrl={chapterUrl} />
}

// Reader uses bare chrome — no sidebar/header/bottomnav.
export const Route = createFileRoute('/browse/$source/manga/$mangaId/chapter/$chapterId')({
  component: ChapterReaderPage,
  staticData: { chrome: 'bare' },
})
