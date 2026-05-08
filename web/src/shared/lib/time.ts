// API serves timestamps as RFC 3339 in UTC (e.g. '2026-05-07T22:43:03Z').
// `Date(...)` parses that directly. Older SQLite-era values without a
// timezone suffix get treated as UTC for backward compatibility.
function parseUtc(s: string): Date {
  return new Date(s.includes('T') ? s : s.replace(' ', 'T') + 'Z')
}

export function timeAgo(s: string | null | undefined): string {
  if (!s) return ''
  const then = parseUtc(s).getTime()
  const diff = Date.now() - then
  if (Number.isNaN(diff)) return ''

  const sec = Math.max(1, Math.round(diff / 1000))
  if (sec < 60)        return `${sec} giây trước`
  const min = Math.round(sec / 60)
  if (min < 60)        return `${min} phút trước`
  const hr = Math.round(min / 60)
  if (hr  < 24)        return `${hr} giờ trước`
  const day = Math.round(hr / 24)
  if (day < 30)        return `${day} ngày trước`
  const mo = Math.round(day / 30)
  if (mo  < 12)        return `${mo} tháng trước`
  return `${Math.round(mo / 12)} năm trước`
}
