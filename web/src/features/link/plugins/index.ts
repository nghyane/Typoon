// Built-in link plugins — bundled at build time.
//
// Plugin order is significant: when two plugins commit competing
// `title_locale` values, the FIRST one in this list wins. We list
// MangaBaka first because it's the aggregator (Anilist + MAL +
// MangaUpdates + Kitsu in one lookup) and tends to have the cleanest
// native_title / romanized_title pair. MangaDex follows for its
// own UUID namespace.
//
// Anilist used to have a standalone adapter; we removed it after
// MangaBaka started forwarding the Anilist id reliably, and after
// Anilist's API entered extended "temporarily disabled" outages.
// Re-introduce a dedicated adapter only if MangaBaka loses Anilist
// coverage or develops a quality gap we need to compensate for.
//
// Adapters are plain TypeScript modules (not JSON) because each
// upstream has its own response quirks that a declarative JSONPath
// schema can't express cleanly. The runtime contract
// (`LinkPluginAdapter`) is small and stable; adding a plugin = drop
// a file in this folder + push it onto the array below.

import type { LinkPluginAdapter } from '../runtime'
import { mangabakaPlugin } from './mangabaka'
import { mangadexPlugin }  from './mangadex'


export const bundledLinkPlugins: LinkPluginAdapter[] = [
  mangabakaPlugin,
  mangadexPlugin,
]
