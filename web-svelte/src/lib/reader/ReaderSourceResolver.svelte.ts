// reader/ReaderSourceResolver.svelte.ts — source state, switching, and preference persistence.

import { getSource } from '$lib/source/registry';
import { fetchChapterPages } from '$lib/source/runtime/endpoints';
import { localSettings } from '$lib/localSettings.svelte';
import type { ReaderData, ReaderSourceVersion } from '$lib/types';

interface SourceResolverDeps {
  data: () => ReaderData | null;
  /** Called before switching source (e.g. to stop translator). */
  onBeforeSwitch?: () => void;
  /** Called after successful switch (e.g. to reset scroll position). */
  onAfterSwitch?: () => void;
}

export class ReaderSourceResolver {
  activeUrls = $state<string[]>([]);
  activeTokens = $state<string[] | null>(null);
  activePageHeaders = $state<Record<string, string> | null>(null);
  activeSourceId = $state<string | undefined>(undefined);
  activeSourceName = $state<string | undefined>(undefined);
  activeSourceLang = $state<string | undefined>(undefined);
  activeVersionKey = $state<string | null>(null);
  sourceSwitching = $state(false);
  sourceError = $state('');

  #data: () => ReaderData | null;
  #onBeforeSwitch?: () => void;
  #onAfterSwitch?: () => void;
  #lastRouteKey = '';
  #appliedPrefRouteKey = '';
  #loadGeneration = 0;

  constructor(deps: SourceResolverDeps) {
    this.#data = deps.data;
    this.#onBeforeSwitch = deps.onBeforeSwitch;
    this.#onAfterSwitch = deps.onAfterSwitch;

    // Sync from data when route changes
    $effect(() => {
      const d = this.#data();
      const routeKey = d ? `${d.workId}:${d.chapterRef}` : '';
      if (!d || !routeKey || routeKey === this.#lastRouteKey) return;
      this.#lastRouteKey = routeKey;
      const version = initialVersion(d);
      const hasUrls = (d.urls?.length ?? 0) > 0;
      const generation = ++this.#loadGeneration;
      this.activeUrls = d.urls ?? [];
      this.activeTokens = d.pageTokens ?? null;
      this.activePageHeaders = d.pageHeaders ?? null;
      this.activeSourceId = d.sourceId ?? version?.sourceId;
      this.activeSourceName = d.sourceName ?? version?.sourceName;
      this.activeSourceLang = d.sourceLang ?? version?.lang;
      this.activeVersionKey = d.selectedVersionKey ?? version?.key ?? null;
      this.sourceSwitching = !hasUrls && !!version;
      this.sourceError = '';
      if (!hasUrls && version) void this.loadVersionPages(d, version, generation, { callAfterSwitch: false, persist: false });
    });

    // Apply saved source preference
    $effect(() => {
      const d = this.#data();
      if (!d || this.sourceSwitching) return;
      const pref = localSettings.state.reader_source_prefs[d.workId];
      const prefKey = pref?.kind === 'raw' ? pref.versionKey : null;
      const routePrefKey = `${d.workId}:${d.chapterRef}:${prefKey ?? 'auto'}`;
      if (!prefKey || routePrefKey === this.#appliedPrefRouteKey) return;
      this.#appliedPrefRouteKey = routePrefKey;
      const version = d.versions.find((item) => item.key === prefKey);
      if (version && version.key !== this.activeVersionKey) {
        void this.switchSource(version, { persist: false });
      }
    });
  }

  async switchSource(
    version: ReaderSourceVersion,
    options: { persist?: boolean } = {},
  ): Promise<void> {
    const d = this.#data();
    if (!d || (version.key === this.activeVersionKey && this.activeUrls.length > 0)) return;

    this.#onBeforeSwitch?.();
    const generation = ++this.#loadGeneration;
    this.sourceSwitching = true;
    this.sourceError = '';
    await this.loadVersionPages(d, version, generation, { callAfterSwitch: true, persist: options.persist });
  }

  private async loadVersionPages(
    d: ReaderData,
    version: ReaderSourceVersion,
    generation: number,
    options: { callAfterSwitch: boolean; persist?: boolean },
  ): Promise<void> {
    this.activeSourceId = version.sourceId;
    this.activeSourceName = version.sourceName;
    this.activeSourceLang = version.lang;
    this.activeVersionKey = version.key;

    try {
      const source = getSource(version.sourceId);
      if (!source) throw new Error(`Nguồn ${version.sourceName} không khả dụng.`);

      const nextPages = await fetchChapterPages(source.manifest, version.url);
      if (nextPages.pages.length === 0) throw new Error('Nguồn này không có trang đọc.');
      if (generation !== this.#loadGeneration) return;

      this.activeUrls = nextPages.pages;
      this.activeTokens = nextPages.tokens ?? null;
      this.activePageHeaders = nextPages.pageHeaders ?? null;

      if (options.persist !== false) {
        localSettings.update({
          reader_source_prefs: {
            ...localSettings.state.reader_source_prefs,
            [d.workId]: { kind: 'raw', versionKey: version.key },
          },
        });
      }

      if (options.callAfterSwitch) this.#onAfterSwitch?.();
    } catch (err) {
      if (generation !== this.#loadGeneration) return;
      this.sourceError = err instanceof Error ? err.message : String(err);
    } finally {
      if (generation === this.#loadGeneration) this.sourceSwitching = false;
    }
  }
}

function initialVersion(data: ReaderData): ReaderSourceVersion | null {
  if (!data.versions.length) return null;
  return data.versions.find((version) => version.key === data.selectedVersionKey) ?? data.versions[0] ?? null;
}
