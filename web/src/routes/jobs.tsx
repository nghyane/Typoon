// /jobs — activity feed (last 7d, server-truth via /me/jobs).

import { createFileRoute } from '@tanstack/react-router'

import { useMyJobs, useDeleteJob } from '@features/jobs/queries'
import { useWorksByIds } from '@features/works/queries'
import { JobCard } from '@features/jobs/JobCard'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'

function JobsPage() {
  const jobs = useMyJobs()
  const del  = useDeleteJob()
  const workLookup = useWorksByIds(jobs.data?.map(j => j.work_id) ?? [])

  if (jobs.isPending) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner size={20} />
      </div>
    )
  }

  if (!jobs.data || jobs.data.length === 0) {
    return (
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10">
        <EmptyState
          title="Chưa có hoạt động"
          hint="Thả một file zip vào trang chủ để bắt đầu dịch."
        />
      </div>
    )
  }

  const lookupTitle = (id: string | null): string | null =>
    !id ? null : (workLookup.data?.get(id)?.title ?? null)

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-6 space-y-4">
      <header>
        <h1 className="text-lg font-semibold text-text">Hoạt động</h1>
        <p className="text-xs text-text-muted">7 ngày gần nhất · {jobs.data.length} job</p>
      </header>
      <div className="space-y-1.5">
        {jobs.data.map(job => (
          <JobCard
            key={job.id}
            job={job}
            workTitle={lookupTitle(job.work_id)}
            variant="full"
            onDelete={(id) => del.mutate(id)}
          />
        ))}
      </div>
    </div>
  )
}

export const Route = createFileRoute('/jobs')({
  component: JobsPage,
  staticData: { auth: 'required' },
})
