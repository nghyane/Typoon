export interface HuggingFaceModelRepositoryOptions {
  readonly repo: string
  readonly revision?: string
  readonly manifestPath?: string
}

/** Build a raw Hugging Face resolve URL for any file in a repo. */
export function huggingFaceResolveUrl(
  repo: string,
  path: string,
  revision = 'main',
): string {
  return `https://huggingface.co/${repo}/resolve/${revision}/${path}`
}

/** Build the manifest URL for a Hugging Face model repo. */
export function huggingFaceManifestUrl(
  options: HuggingFaceModelRepositoryOptions,
): string {
  return huggingFaceResolveUrl(
    options.repo,
    options.manifestPath ?? 'manifest.json',
    options.revision ?? 'main',
  )
}
