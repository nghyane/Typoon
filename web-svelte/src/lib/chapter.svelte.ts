// $lib/chapter.svelte.ts — reactive streaming chapter page cache.

import { untrack } from 'svelte';
import { useSourceFetch } from './sourceFetch.svelte';
import type { PageBlob } from './types';

type ResolvePageUrl = (index: number, rawUrl: string) => Promise<string>;
type PageHeaders = () => Record<string, string> | null | undefined;

// Abandon a slow/stalled gateway and fail over to the next mirror. Generous
// enough that a normal large page on a slow connection still completes.
const GATEWAY_TIMEOUT_MS = 12_000;

export class ChapterPages {
  blobs = $state<PageBlob[]>([]);
  pageSizes = $state<Array<{ width: number; height: number } | null>>([]);
  done = $state(0);
  total = $state(0);
  loading = $state(true);
  error = $state('');

  #ac: AbortController | null = null;
  #doneIndexes = new Set<number>();
  #generation = 0;
  #inflight = new Map<number, Promise<Blob>>();
  #key = '';
  #resolveUrl: ResolvePageUrl;
  #pageHeaders: PageHeaders;
  #sf = useSourceFetch();
  #headers: Record<string, string> | undefined;
  #urls: readonly string[] = [];

  constructor(
    urls: () => readonly string[],
    resolveUrl: ResolvePageUrl = (_index, rawUrl) => Promise.resolve(rawUrl),
    pageHeaders: PageHeaders = () => undefined,
  ) {
    this.#resolveUrl = resolveUrl;
    this.#pageHeaders = pageHeaders;

    $effect(() => {
      const list = urls();
      const headers = this.#pageHeaders() ?? undefined;
      untrack(() => {
        const key = `${list.join('\n')}\n\u0000${stableHeaders(headers)}`;
        if (key === this.#key) return;
        this.#key = key;
        this.#start(list, headers);
      });
      return () => this.#ac?.abort();
    });
  }

  readPage(index: number, signal?: AbortSignal): Promise<Blob> {
    if (index < 0 || index >= this.#urls.length) return Promise.reject(new Error(`missing page ${index + 1}`));
    const blob = this.blobs[index];
    if (blob) return Promise.resolve(blob);
    const ac = this.#ac;
    if (!ac || ac.signal.aborted) return Promise.reject(new Error('chapter pages are not active'));
    return raceAbort(this.#loadPage(index, this.#generation, ac.signal), signal);
  }

  destroy(): void {
    this.#ac?.abort();
  }

  #start(list: readonly string[], headers: Record<string, string> | undefined): void {
    this.#ac?.abort();
    this.#generation += 1;
    this.#urls = [...list];
    this.#headers = headers;
    this.#inflight.clear();
    this.#doneIndexes.clear();

    if (!list.length) {
      this.blobs = [];
      this.pageSizes = [];
      this.done = 0;
      this.total = 0;
      this.loading = false;
      this.error = '';
      this.#ac = null;
      return;
    }

    const ac = new AbortController();
    const generation = this.#generation;
    this.#ac = ac;
    this.total = list.length;
    this.blobs = new Array<PageBlob>(list.length).fill(null);
    this.pageSizes = new Array<{ width: number; height: number } | null>(list.length).fill(null);
    this.done = 0;
    this.loading = true;
    this.error = '';

    void this.#downloadSequentially(generation, ac.signal);
  }

  async #downloadSequentially(generation: number, signal: AbortSignal): Promise<void> {
    for (let index = 0; index < this.#urls.length && !signal.aborted; index += 1) {
      await this.#loadPage(index, generation, signal).catch(() => undefined);
    }
  }

  #loadPage(index: number, generation: number, signal: AbortSignal): Promise<Blob> {
    const cached = this.blobs[index];
    if (cached) return Promise.resolve(cached);
    const existing = this.#inflight.get(index);
    if (existing) return existing;

    const promise = this.#fetchPage(index, generation, signal)
      .finally(() => {
        if (this.#generation === generation) this.#inflight.delete(index);
      });
    this.#inflight.set(index, promise);
    return promise;
  }

  async #fetchPage(index: number, generation: number, signal: AbortSignal): Promise<Blob> {
    try {
      const rawUrl = await this.#resolveUrl(index, this.#urls[index] ?? '');
      if (!rawUrl) throw new Error(`missing page url ${index + 1}`);
      throwIfAborted(signal);
      const blob = await this.#fetchBlob(rawUrl, signal);
      const size = await readImageSize(blob);
      throwIfAborted(signal);
      if (this.#generation !== generation) throw new Error('stale page fetch');
      this.blobs[index] = blob;
      this.pageSizes[index] = size;
      this.#markDone(index);
      return blob;
    } catch (err) {
      if (!signal.aborted && this.#generation === generation) {
        this.error ||= String(err);
        this.#markDone(index);
      }
      throw err;
    }
  }

  // Page images must honor the same gateway fallback as covers and source
  // metadata: the primary gateway (discordsays) leads but gets rate-limited on
  // non-Activity deploys (observed: HTTP 503, and concurrent loads stalling for
  // 10s+), so a single-gateway fetch leaves the reader blank while covers (which
  // retry) still render. Walk every gateway, failing a slow/stalled one over to
  // the next via a per-attempt timeout. The last gateway runs untimed so a
  // genuinely slow network on the final mirror still completes.
  async #fetchBlob(rawUrl: string, signal: AbortSignal): Promise<Blob> {
    const count = Math.max(1, this.#sf.gatewayCount);
    let lastError: unknown;
    for (let attempt = 0; attempt < count; attempt += 1) {
      throwIfAborted(signal);
      const isLast = attempt === count - 1;
      const ac = new AbortController();
      const onAbort = () => ac.abort(signal.reason);
      signal.addEventListener('abort', onAbort, { once: true });
      const timer = isLast ? null : setTimeout(() => ac.abort(new Error('gateway timeout')), GATEWAY_TIMEOUT_MS);
      try {
        const res = await fetch(this.#sf.toBrowserUrl(rawUrl, this.#headers, undefined, attempt), { signal: ac.signal });
        if (res.ok) return await res.blob();
        lastError = new Error(`${res.status}`);
      } catch (err) {
        if (signal.aborted) throw err;
        lastError = err;
      } finally {
        if (timer) clearTimeout(timer);
        signal.removeEventListener('abort', onAbort);
      }
    }
    throw lastError ?? new Error('all gateways failed');
  }

  #markDone(index: number): void {
    if (this.#doneIndexes.has(index)) return;
    this.#doneIndexes.add(index);
    this.done = this.#doneIndexes.size;
    this.loading = this.done < this.total;
  }
}

function stableHeaders(headers: Record<string, string> | undefined): string {
  if (!headers || Object.keys(headers).length === 0) return '';
  return JSON.stringify(Object.fromEntries(Object.entries(headers).sort(([a], [b]) => a.localeCompare(b))));
}

async function readImageSize(blob: Blob): Promise<{ width: number; height: number }> {
  // `from-image`: match EXIF orientation applied by <img> and the OCR canvas so
  // the page-frame aspect ratio agrees with the displayed pixels.
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  try {
    return { width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}

function raceAbort<T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError(signal));
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
  });
}

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return;
  throw abortError(signal);
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error('operation aborted');
}
