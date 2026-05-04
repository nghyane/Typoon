import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({
  component: () => (
    <div className="p-8 text-zinc-500">Chọn một mục từ sidebar.</div>
  ),
})
