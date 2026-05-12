import { createFileRoute, redirect } from '@tanstack/react-router'
import { TitleHub } from '@features/title/TitleHub'

function TitleHubPage() {
  const { entryId } = Route.useParams()
  const id = Number(entryId)
  return <TitleHub entryId={id} />
}

export const Route = createFileRoute('/title/$entryId')({
  beforeLoad: ({ params }) => {
    const id = Number(params.entryId)
    if (!Number.isFinite(id) || id <= 0) {
      throw redirect({ to: '/library' })
    }
  },
  component: TitleHubPage,
})
