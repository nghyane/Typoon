import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { Plus, ChevronRight } from 'lucide-react'
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
    <div className="max-w-3xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-(--color-text-1)">Projects</h1>
        <button className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md bg-(--color-accent) text-white hover:bg-(--color-accent-hover) transition-colors">
          <Plus size={14} />
          New project
        </button>
      </div>

      <div className="border border-(--color-border) rounded-lg overflow-hidden">
        {isLoading
          ? Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex items-center gap-3 px-4 py-3 border-b border-(--color-border) last:border-0">
                <Skeleton className="w-8 h-8 rounded-md shrink-0" />
                <div className="flex-1 flex flex-col gap-1.5">
                  <Skeleton className="h-3.5 w-40 rounded" />
                  <Skeleton className="h-3 w-20 rounded" />
                </div>
              </div>
            ))
          : data?.map((p) => (
              <Link
                key={p.project_id}
                to="/projects/$id"
                params={{ id: String(p.project_id) }}
                className="flex items-center gap-3 px-4 py-3 border-b border-(--color-border) last:border-0 hover:bg-(--color-surface) transition-colors group"
              >
                <div className="w-8 h-8 rounded-md shrink-0 flex items-center justify-center text-sm font-bold bg-(--color-surface) border border-(--color-border) text-(--color-text-2)">
                  {p.title[0]}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-(--color-text-1) truncate">{p.title}</p>
                  <p className="text-xs text-(--color-text-3) mt-0.5">
                    {p.source_lang.toUpperCase()} → {p.target_lang.toUpperCase()}
                  </p>
                </div>
                <ChevronRight size={14} className="text-(--color-text-3) opacity-0 group-hover:opacity-100 transition-opacity" />
              </Link>
            ))}
      </div>
    </div>
  )
}
