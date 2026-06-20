// Source fetch — rewrites upstream manga/source URLs through a browser-safe
// gateway, and fetches them with proxy headers injected out-of-band.
//
// `createSourceFetch()` is pure: settings enter as explicit origin list and
// the returned object owns URL building/fetching for that config snapshot.
//
// Wire format:
//
//   `fetch(url, opts)`          — fetch via `X-Proxy-Headers: <blob>`.
//   `toBrowserUrl(url, headers?)` — rewrite for `<img src>`. Adds
//                                   `?_h=<blob>` when headers given.
//
// `<blob>` is base64url(JSON of `{ HeaderName: value, ... }`). The
// server forwards every entry verbatim, minus a small denylist
// (Host, Connection, Cf-*, X-Forwarded-*, …).

export interface FetchOptions {
  /** Arbitrary HTTP headers to forward to the upstream. Keys are
   *  plain header names (Referer, User-Agent, Authorization, …);
   *  values are strings. The gateway encodes them server-side. */
  headers?: Record<string, string>
  /** Standard fetch init — method, body, signal, etc. */
  init?:    RequestInit
}

export interface SourceFetch {
  toBrowserUrl(url: string, headers?: Record<string, string>): string
  fetch(url: string, opts?: FetchOptions): Promise<Response>
}

export function createSourceFetch(origins: readonly string[]): SourceFetch {
  const gateways = origins.filter(isGatewayOrigin)

  function gatewayFor(key: string): string {
    if (gateways.length === 0) return '/cdn/c'
    const origin = gateways[hash(key) % gateways.length]
    return `${origin}/cdn/c`
  }

  function toBrowserUrl(url: string, headers?: Record<string, string>): string {
    let u: URL
    try { u = new URL(url) } catch { return url }
    if (u.protocol !== 'https:' && u.protocol !== 'http:') return url

    const path   = `${gatewayFor(`${u.host}${u.pathname}`)}/${u.host}${u.pathname}`
    const params = new URLSearchParams()
    for (const [k, v] of u.searchParams) params.append(k, v)
    if (headers && Object.keys(headers).length > 0) {
      params.set('_h', encodeHeaderBlob(headers))
    }
    const qs = params.toString()
    return qs ? `${path}?${qs}` : path
  }

  function fetch(url: string, opts: FetchOptions = {}): Promise<Response> {
    const h = new Headers(opts.init?.headers)
    if (opts.headers && Object.keys(opts.headers).length > 0) {
      h.set('X-Proxy-Headers', encodeHeaderBlob(opts.headers))
    }
    return window.fetch(toBrowserUrl(url), { ...opts.init, headers: h })
  }

  return { toBrowserUrl, fetch }
}

/** base64url(JSON(headers)). Stable key order to keep cache hits
 *  consistent across calls with identical header sets. */
function encodeHeaderBlob(headers: Record<string, string>): string {
  const sorted: Record<string, string> = {}
  for (const k of Object.keys(headers).sort()) sorted[k] = headers[k]!
  const json  = JSON.stringify(sorted)
  const bytes = new TextEncoder().encode(json)
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function isGatewayOrigin(value: string): boolean {
  return /^https?:\/\/[^/?#]+$/i.test(value)
}

function hash(value: string): number {
  let h = 2166136261
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

// ── Legacy same-origin fallback for adapters not yet injected ─────
const sameOriginFetch = createSourceFetch([])
export const toBrowserUrl = sameOriginFetch.toBrowserUrl
export const fetchSource = sameOriginFetch.fetch
