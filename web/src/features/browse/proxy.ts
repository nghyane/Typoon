// Proxy helper — every browse-mode network call goes through here.
//
// Inside the DA iframe (`*.discordsays.com`) we use same-origin
// paths; outside we hit `VITE_PUBLIC_BASE_URL` cross-origin. CORS on
// bunle-cdn is wide open.
//
// Two entry points share ONE wire format:
//
//   `pfetch(url, opts)`         — fetch via `X-Proxy-Headers: <blob>`.
//   `proxify(url, headers?)`    — rewrite for `<img src>`. Adds
//                                 `?_h=<blob>` when headers given.
//
// `<blob>` is base64url(JSON of `{ HeaderName: value, ... }`). The
// server forwards every entry verbatim, minus a small denylist
// (Host, Connection, Cf-*, X-Forwarded-*, …). No allowlist — adapters
// can pass any custom auth/sign/referer/cookie they need.

const BASE = window.location.hostname.endsWith('.discordsays.com')
  ? ''
  : ((import.meta.env.VITE_PUBLIC_BASE_URL as string | undefined) ?? '')
const PROXY_BASE = `${BASE.replace(/\/+$/, '')}/cdn/c`

export interface ProxyOpts {
  /** Arbitrary HTTP headers to forward to the upstream. Keys are
   *  plain header names (Referer, User-Agent, Authorization, …);
   *  values are strings. The proxy encodes them server-side. */
  headers?: Record<string, string>
  /** Standard fetch init — method, body, signal, etc. */
  init?:    RequestInit
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

/** Rewrite an absolute upstream URL into the proxy URL.
 *  `https://m.happymh.com/page.jpg?v=2`
 *    → `/cdn/c/m.happymh.com/page.jpg?v=2`
 *
 *  Optional `headers` get encoded as `?_h=<blob>` so an `<img>` can
 *  carry overrides the proxy applies server-side. */
export function proxify(
  url: string,
  headers?: Record<string, string>,
): string {
  let u: URL
  try { u = new URL(url) } catch { return url }
  if (u.protocol !== 'https:' && u.protocol !== 'http:') return url

  const path   = `${PROXY_BASE}/${u.host}${u.pathname}`
  const params = new URLSearchParams()
  for (const [k, v] of u.searchParams) params.append(k, v)
  if (headers && Object.keys(headers).length > 0) {
    params.set('_h', encodeHeaderBlob(headers))
  }
  const qs = params.toString()
  return qs ? `${path}?${qs}` : path
}

/** Generic proxied fetch — HTML/JSON/blob/POST. Forwards `headers`
 *  through `X-Proxy-Headers` so they don't bloat the URL or the
 *  CORS preflight surface. */
export function pfetch(url: string, opts: ProxyOpts = {}): Promise<Response> {
  const h = new Headers(opts.init?.headers)
  if (opts.headers && Object.keys(opts.headers).length > 0) {
    h.set('X-Proxy-Headers', encodeHeaderBlob(opts.headers))
  }
  return fetch(proxify(url), { ...opts.init, headers: h })
}
