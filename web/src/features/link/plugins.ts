// Built-in link plugins — bundled at build time so the auto-enrich
// flow has something to fan out to from a cold start.
//
// Same pattern as `features/browse/sources.ts`: Vite's `glob` pulls
// every JSON in `packages/link-plugins/` and exposes the parsed
// modules as a static array. Adding a plugin = drop a new JSON +
// declare it in `packages/link-plugins/index.json` (or just keep the
// glob — Vite picks it up either way).

import type { LinkPlugin } from './runtime'


const BUNDLED: Record<string, { default: LinkPlugin }> =
  import.meta.glob('../../../../packages/link-plugins/*.json', { eager: true })


function isLinkPlugin(mod: { default: unknown }): mod is { default: LinkPlugin } {
  const p = mod.default as Partial<LinkPlugin> | undefined
  return !!p?.id && !!p?.namespace && !!p?.endpoints?.search
}


export const bundledLinkPlugins: LinkPlugin[] = Object.values(BUNDLED)
  .filter(isLinkPlugin)
  .map((m) => m.default)
