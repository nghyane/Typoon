// $lib/hooks/useChapterPages.svelte.ts
// Svelte 5 runes — $state + $effect replace useState + useEffect.
// No dependency arrays.  No double-mount bugs.  No ref tricks.

import { useSourceFetch } from '$lib/sourceFetch.svelte';

export interface PageBlobs {
  blobs: (Blob | null)[];
  done: number;
  total: number;
}

class ChapterPages {
  blobs = $state<(Blob | null)[]>([]);
  done = $state(0);
  total = $state(0);

  #sourceFetch = useSourceFetch();
  #ac: AbortController | null = null;

  constructor(
    rawUrls: () => readonly string[],
    key: () => string,
    pageHeaders: () => Record<string, string> | null | undefined = () => undefined,
  ) {
    $effect(() => {
      const urls = rawUrls();
      const _k = key(); // trigger re-run when key changes
      const headers = pageHeaders() ?? undefined;
      if (!urls.length) return;

      const ac = new AbortController();
      this.#ac = ac;
      this.total = urls.length;
      this.blobs = new Array(urls.length).fill(null);
      this.done = 0;

      const CONCURRENCY = 6;
      let next = 0;
      let done = 0;

      const fetchOne = async (i: number) => {
        try {
          const sf = this.#sourceFetch;
          const proxied = sf.toBrowserUrl(urls[i]!, headers);
          const res = await fetch(proxied, { signal: ac.signal });
          if (!res.ok) throw new Error(`${res.status}`);
          const blob = await res.blob();
          if (!ac.signal.aborted) {
            this.blobs[i] = blob;
            done++;
            this.done = done;
            // Trigger reactivity
            this.blobs = [...this.blobs];
          }
        } catch {
          if (!ac.signal.aborted) {
            done++;
            this.done = done;
          }
        }
      };

      const worker = async () => {
        while (!ac.signal.aborted) {
          const i = next++;
          if (i >= urls.length) break;
          await fetchOne(i);
        }
      };

      Promise.allSettled(
        Array.from({ length: Math.min(CONCURRENCY, urls.length) }, worker),
      );

      return () => ac.abort();
    });
  }

  destroy() {
    this.#ac?.abort();
  }
}

export function useChapterPages(
  rawUrls: () => readonly string[],
  key: () => string,
  pageHeaders?: () => Record<string, string> | null | undefined,
) {
  return new ChapterPages(rawUrls, key, pageHeaders);
}
