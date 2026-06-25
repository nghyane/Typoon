// reader/translation.svelte.ts — Svelte wrapper for ReaderTranslation.

import { ReaderTranslation, type ReaderTranslationChapter, type ReaderTranslationState, type TranslationProvider } from '@typoon/client/web-reader';

export type { ReaderTranslationChapter, ReaderTranslationState, TranslationProvider };

export class SvelteReaderTranslation {
  state = $state<ReaderTranslationState>(empty);
  #rt = new ReaderTranslation();
  #unsubscribe: (() => void) | null = null;

  setChapter(chapter: ReaderTranslationChapter): void { this.#rt.setChapter(chapter); }
  clear(): void { this.#rt.clear(); }

  registerContentHost(el: HTMLElement): () => void { return this.#rt.registerContentHost(el); }
  registerPage(pageIndex: number, el: HTMLElement): () => void { return this.#rt.registerPage(pageIndex, el); }
  translate(): void { this.#rt.translate(); }
  setProvider(provider: TranslationProvider): void { this.#rt.setProvider(provider); }
  setHidden(hidden: boolean): void { this.#rt.setHidden(hidden); }
  cancel(): void { this.#rt.cancel(); }
  dispose(): void {
    this.#unsubscribe?.();
    this.#unsubscribe = null;
    this.#rt.dispose();
  }

  constructor() {
    // Bind engine state straight into the rune. Svelte 5 batches DOM updates, so
    // the previous rAF-coalescing layer was a redundant reimplementation.
    this.#unsubscribe = this.#rt.subscribe(s => { this.state = s; });
  }
}

const empty: ReaderTranslationState = {
  phase: 'idle', prepare: { done: 0, total: 0, preparedPages: 0 }, translate: { done: 0, total: 0 }, model: { state: 'idle' }, sourceLanguage: null, targetLanguage: 'vi', hidden: false, failed: 0,
};
