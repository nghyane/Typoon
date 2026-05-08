import type { ApiChapter } from '@shared/api/api'

export type Filter = 'all' | 'idle' | 'running' | 'done' | 'error'

export const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all',     label: 'Tất cả'    },
  { key: 'running', label: 'Đang chạy' },
  { key: 'idle',    label: 'Chờ xử lý' },
  { key: 'done',    label: 'Xong'      },
  { key: 'error',   label: 'Lỗi'       },
]

export function matchFilter(ch: ApiChapter, f: Filter): boolean {
  if (f === 'all')     return true
  if (f === 'idle')    return ch.state === 'idle'
  if (f === 'running') return ch.state === 'running' || ch.state === 'pending'
  return ch.state === f
}

export type Tab = 'chapters' | 'glossary' | 'settings'
