export interface HuggingFaceModelRepositoryOptions {
  readonly repo: string
  readonly revision?: string
  readonly manifestPath?: string
  readonly proxyBase?: string
}

/** Build a raw Hugging Face resolve URL for any file in a repo. */
export function huggingFaceResolveUrl(
  repo: string,
  path: string,
  revision = 'main',
  proxyBase?: string,
): string {
  return proxyHuggingFaceUrl(`https://huggingface.co/${repo}/resolve/${revision}/${path}`, proxyBase)
}

/** Build the manifest URL for a Hugging Face model repo. */
export function huggingFaceManifestUrl(
  options: HuggingFaceModelRepositoryOptions,
): string {
  return huggingFaceResolveUrl(
    options.repo,
    options.manifestPath ?? 'manifest.json',
    options.revision ?? 'main',
    options.proxyBase,
  )
}

/**
 * Route any https URL through a proxy gateway as `${proxyBase}/<host><path><search>`.
 * Host-agnostic version of proxyHuggingFaceUrl — used for self-hosted model CDNs
 * (e.g. R2). The discordsays `/cdn/c` gateway proxies arbitrary upstream hosts,
 * so this works inside the Discord Activity sandbox as well as plain browsers.
 */
export function proxyUrl(url: string, proxyBase: string | undefined): string {
  if (!proxyBase) return url
  let upstream: URL
  try {
    upstream = new URL(url)
  } catch {
    return url
  }
  if (upstream.protocol !== 'https:') return url
  return `${proxyBase.replace(/\/+$/u, '')}/${upstream.hostname}${upstream.pathname}${upstream.search}`
}

export function proxyHuggingFaceUrl(url: string, proxyBase: string | undefined): string {
  if (!proxyBase) return url

  let upstream: URL
  try {
    upstream = new URL(url)
  } catch {
    return url
  }
  if (upstream.protocol !== 'https:' || upstream.hostname !== 'huggingface.co') return url

  return `${proxyBase.replace(/\/+$/u, '')}/${upstream.hostname}${upstream.pathname}${upstream.search}`
}
