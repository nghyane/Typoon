import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { Plus } from 'lucide-react'
import { projectsApi, projectKeys } from '../../api/projects'
import { Skeleton } from '../../components/ui/Skeleton'

export const Route = createFileRoute('/projects/')({
  component: ProjectsPage,
})

function ProjectsPage() {
  const { data, isLoading } = useQuery({
    queryKey: projectKeys.all(),
    queryFn:  projectsApi.list,
  })

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 h-14 border-b border-(--color-border) shrink-0">
        <h1 className="text-sm font-semibold text-(--color-text-1)">Dự án</h1>
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-(--color-accent) text-(--color-accent-text) hover:bg-(--color-accent-hover) transition-colors">
          <Plus size={14} />
          Thêm dự án
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 flex flex-col gap-2">
        {isLoading
          ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-16 rounded-xl" />)
          : data?.map((p) => (
              <Link
                key={p.project_id}
                to="/projects/$id"
                params={{ id: String(p.project_id) }}
                className="flex items-center gap-4 px-4 py-3 rounded-xl border border-(--color-border) bg-(--color-bg) hover:bg-(--color-surface-1) transition-colors"
              >
                <div className="w-10 h-10 rounded-lg shrink-0 flex items-center justify-center text-lg font-bold bg-(--color-surface-2) text-(--color-text-3)">
                  {p.title[0]}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate text-(--color-text-1)">{p.title}</p>
                  <p className="text-xs mt-0.5 text-(--color-text-3)">
                    {p.source_lang} → {p.target_lang}
                  </p>
                </div>
              </Link>
            ))}
      </div>
    </div>
  )
}
