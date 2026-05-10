import type { UploadProgress } from '@typoon/upload-sdk'

// Minimal upload progress footer — replaces action buttons during
// async submission. The user clicked Submit then almost certainly
// switched tabs; the footer's job is to confirm "still going" and
// show the one number that answers "how much longer", not to spell
// out internal phases.
//
// Three states map to one label each:
//   packing    → "Đang đóng gói…"
//   uploading  → "Đang tải lên · {pct}%"
//   finalizing → "Engine đang xử lý…"
//
// One progress bar drives the visual. Indeterminate phases show a
// grow-from-zero pulse instead of a separate shimmer overlay; the
// width transition is enough motion in a static modal that's
// already focus-trapped.

export function UploadProgressFooter({ progress }: { progress: UploadProgress }) {
  const { phase, bytesSent, bytesTotal } = progress

  const pct = phase === 'uploading' && bytesTotal > 0
    ? Math.min(100, (bytesSent / bytesTotal) * 100)
    : phase === 'finalizing'
      ? 100
      : 6  // hint of motion while packing

  const label = phase === 'packing'
    ? 'Đang đóng gói…'
    : phase === 'uploading'
      ? `Đang tải lên · ${Math.round(pct)}%`
      : 'Engine đang xử lý…'

  return (
    <div className="px-5 py-3 space-y-2">
      <p className="text-xs text-text-muted tabular">{label}</p>
      <div className="relative h-[3px] rounded-full bg-surface-2 overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-accent rounded-full transition-[width] duration-300 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
