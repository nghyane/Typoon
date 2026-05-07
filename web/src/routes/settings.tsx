import { createFileRoute } from '@tanstack/react-router'
import { Link } from '@tanstack/react-router'
import { Settings as Cog, ArrowRight } from 'lucide-react'

function SettingsPage() {
  return (
    <div className="px-6 py-10 max-w-2xl">
      <div className="flex items-center gap-3 mb-6">
        <div className="size-10 rounded-xl bg-zinc-100 flex items-center justify-center">
          <Cog size={18} className="text-zinc-500" />
        </div>
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900">Cài đặt</h1>
          <p className="text-sm text-zinc-500">Cấu hình ứng dụng</p>
        </div>
      </div>

      <p className="text-sm text-zinc-500 mb-3">
        Cài đặt mỗi dự án nằm trong tab <span className="font-medium text-zinc-700">Cài đặt</span> của
        chính dự án đó.
      </p>

      <Link
        to="/projects"
        className="inline-flex items-center gap-1.5 h-9 px-4 rounded-lg border border-zinc-200 text-sm text-zinc-600 hover:bg-zinc-50 hover:border-zinc-300 transition-colors cursor-pointer"
      >
        Mở danh sách dự án
        <ArrowRight size={13} />
      </Link>
    </div>
  )
}

export const Route = createFileRoute('/settings')({
  component: SettingsPage,
})
