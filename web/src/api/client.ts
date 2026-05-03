const BASE = import.meta.env.VITE_API_URL ?? ''

class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new ApiError(res.status, text)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export const client = {
  get:    <T>(path: string)                     => request<T>(path),
  post:   <T>(path: string, body?: unknown)     => request<T>(path, { method: 'POST',   body: body ? JSON.stringify(body) : undefined }),
  delete: <T>(path: string)                     => request<T>(path, { method: 'DELETE' }),
}

export { ApiError }
export const sseUrl = (path: string) => `${BASE}${path}`
