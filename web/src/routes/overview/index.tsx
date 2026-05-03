import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import {
  FolderOpen, CheckCircle2, RefreshCw, AlertTriangle,
  ChevronRight, MoreHorizontal,
} from 'lucide-react'
import { projectsApi, projectKeys } from '../../api/projects'
import { ProgressBar } from '../../components/ui/ProgressBar'

export const Route = createFileRoute('/overview/')({
  component: OverviewPage,
})

// ── Stat card ────────────────────────────────────────────────────────
function StatCard({ icon, label, value, iconClass }: {
  icon: React.ReactNode; label: string; value: string | number; iconClass: string
}) {
  return (
    <div className="flex flex-col gap-3 p-4 rounded-xl border border-(--color-border) bg-(--color-bg)">
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center ${iconClass}`}>
        {icon}
      </div>
      <div>
        <p className="text-xs text-(--color-text-3)">{label}</p>
        <p className="text-2xl font-bold text-(--color-text) mt-0.5">{value}</p>
      </div>
    </div>
  )
}

// ── Pipeline bar ──────────────────────────────────────────────────────
function PipelineRow({ label, value, max, variant }: {
  label: string; value: number; max: number; variant: 'done' | 'running' | 'pending' | 'purple'
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 text-sm text-(--color-text-2) shrink-0">{label}</span>
      <ProgressBar value={(value / max) * 100} variant={variant} className="flex-1" />
      <span className="w-8 text-sm text-right text-(--color-text-3) shrink-0">{value}</span>
    </div>
  )
}

// ── Activity item ─────────────────────────────────────────────────────
function ActivityItem({ icon, text, time }: { icon: React.ReactNode; text: string; time: string }) {
  return (
    <div className="flex items-center gap-3 py-2.5 border-b border-(--color-border) last:border-0">
      <div className="shrink-0">{icon}</div>
      <p className="flex-1 text-sm text-(--color-text)">{text}</p>
      <span className="text-xs text-(--color-text-3) shrink-0">{time}</span>
    </div>
  )
}

export function OverviewPage() {
  const { data: projects = [] } = useQuery({
    queryKey: projectKeys.all(),
    queryFn: projectsApi.list,
  })

  return (
    <div className="flex h-full">
      {/* Main */}
      <div className="flex-1 min-w-0 px-8 py-7 overflow-y-auto">
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-(--color-text)">Tổng quan</h1>
            <p className="text-sm text-(--color-text-3) mt-0.5">Theo dõi tiến độ dịch, worker và hoạt động gần đây</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-(--color-border)">
            <span className="w-2 h-2 rounded-full bg-(--color-green)" />
            <div className="text-xs">
              <p className="font-medium text-(--color-text)">Workers đang chạy</p>
              <p className="text-(--color-text-3)">4 worker hoạt động</p>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-3 mb-7">
          <StatCard
            icon={<FolderOpen size={18} className="text-(--color-text-2)" />}
            iconClass="bg-(--color-surface)"
            label="Dự án"
            value={projects.length}
          />
          <StatCard
            icon={<CheckCircle2 size={18} className="text-(--color-green)" />}
            iconClass="bg-green-50"
            label="Chương đã dịch"
            value="1,243"
          />
          <StatCard
            icon={<RefreshCw size={18} className="text-(--color-blue)" />}
            iconClass="bg-blue-50"
            label="Đang xử lý"
            value="87"
          />
          <StatCard
            icon={<AlertTriangle size={18} className="text-(--color-orange)" />}
            iconClass="bg-orange-50"
            label="Lỗi cần xem"
            value="3"
          />
        </div>

        {/* Projects progress */}
        <div className="rounded-xl border border-(--color-border) mb-6">
          <div className="flex items-center justify-between px-5 py-4 border-b border-(--color-border)">
            <h2 className="text-sm font-semibold text-(--color-text)">Tiến độ dự án</h2>
          </div>
          <div className="divide-y divide-(--color-border)">
            {projects.slice(0, 4).map(p => (
              <Link
                key={p.project_id}
                to="/projects/$id"
                params={{ id: String(p.project_id) }}
                className="flex items-center gap-4 px-5 py-3.5 hover:bg-(--color-surface) transition-colors"
              >
                <div className="w-10 h-10 rounded-lg bg-(--color-surface) border border-(--color-border) flex items-center justify-center text-sm font-bold text-(--color-text-3) shrink-0">
                  {p.title[0]}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-(--color-text) truncate">{p.title}</span>
                    <span className="text-xs text-(--color-text-3)">{p.source_lang.toUpperCase()} → {p.target_lang.toUpperCase()}</span>
                  </div>
                  <ProgressBar value={60} variant="running" />
                </div>
                <span className="text-xs text-(--color-text-3) shrink-0">— / — chương</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-green-50 text-(--color-green) shrink-0">Đang chạy</span>
                <MoreHorizontal size={15} className="text-(--color-text-3) shrink-0" />
              </Link>
            ))}
          </div>
        </div>

        {/* Recent activity */}
        <div className="rounded-xl border border-(--color-border)">
          <div className="px-5 py-4 border-b border-(--color-border)">
            <h2 className="text-sm font-semibold text-(--color-text)">Hoạt động gần đây</h2>
          </div>
          <div className="px-5">
            <ActivityItem
              icon={<CheckCircle2 size={16} className="text-(--color-green)" />}
              text="Chap 1050 — One Piece hoàn thành"
              time="12 phút trước"
            />
            <ActivityItem
              icon={<RefreshCw size={16} className="text-(--color-blue)" />}
              text="Chap 1049 — One Piece đang render"
              time="28 phút trước"
            />
            <ActivityItem
              icon={<AlertTriangle size={16} className="text-(--color-red)" />}
              text="OCR lỗi ở Chainsaw Man chap 96"
              time="1 giờ trước"
            />
            <ActivityItem
              icon={<CheckCircle2 size={16} className="text-(--color-green)" />}
              text="Attack on Titan chap 138 được duyệt"
              time="2 giờ trước"
            />
          </div>
          <div className="px-5 py-3 border-t border-(--color-border)">
            <button className="flex items-center gap-1 text-sm text-(--color-blue) hover:underline">
              Xem tất cả hoạt động <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Right sidebar */}
      <div className="w-80 shrink-0 border-l border-(--color-border) px-5 py-7 overflow-y-auto flex flex-col gap-6">
        {/* Pipeline stages — từ chapters đang running */}
        <div>
          <h3 className="text-sm font-semibold text-(--color-text) mb-4">Pipeline hôm nay</h3>
          {(() => {
            const scanning   = projects.length  // placeholder — chưa có per-stage count từ API
            const total      = Math.max(scanning, 1)
            return (
              <div className="flex flex-col gap-3">
                <PipelineRow label="Scan"     value={0} max={total} variant="running" />
                <PipelineRow label="Dịch"     value={0} max={total} variant="done" />
                <PipelineRow label="Render"   value={0} max={total} variant="purple" />
                <PipelineRow label="Hoàn thành" value={0} max={total} variant="done" />
              </div>
            )
          })()}
          <p className="text-xs text-(--color-text-3) mt-3">Cần API trả stage count để hiển thị chính xác</p>
        </div>

        {/* Hàng chờ */}
        <div>
          <h3 className="text-sm font-semibold text-(--color-text) mb-2">Hàng chờ</h3>
          <p className="text-sm text-(--color-text-3)">Chưa có API worker status</p>
          <Link to="/pipeline" className="mt-3 w-full flex items-center justify-center gap-2 h-9 rounded-lg border border-(--color-border) text-sm text-(--color-text-2) hover:bg-(--color-surface) transition-colors">
            Mở Pipeline
          </Link>
        </div>

        {/* Storage */}
        <div className="flex items-center gap-3 p-3 rounded-xl border border-(--color-border)">
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-(--color-text)">Dung lượng</p>
            <p className="text-xs text-(--color-text-3) mt-0.5">128.6 GB / 500 GB</p>
            <ProgressBar value={25} variant="running" className="mt-2" />
          </div>
          <ChevronRight size={15} className="text-(--color-text-3) shrink-0" />
        </div>
      </div>
    </div>
  )
}
