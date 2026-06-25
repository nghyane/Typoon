// reader/translation.svelte.ts — Svelte wrapper for ReaderTranslation.

import { ReaderTranslation, type ReaderTranslationChapter, type ReaderTranslationState, type TranslationProvider } from '@typoon/client/web-reader';

export type { ReaderTranslationChapter, ReaderTranslationState, TranslationProvider };

export class SvelteReaderTranslation {
  state = $state<ReaderTranslationState>(empty);
  #rt = new ReaderTranslation();
  #frame = 0;
  #pending: ReaderTranslationState | null = null;
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
    if (this.#frame) cancelAnimationFrame(this.#frame);
    this.#frame = 0;
    this.#pending = null;
    this.#rt.dispose();
  }

  constructor() {
    this.#unsubscribe = this.#rt.subscribe(s => { this.#queueState(s); });
  }

  #queueState(state: ReaderTranslationState): void {
    if (typeof requestAnimationFrame !== 'function') {
      this.state = state;
      return;
    }
    this.#pending = state;
    if (this.#frame) return;
    this.#frame = requestAnimationFrame(() => {
      this.#frame = 0;
      const next = this.#pending;
      this.#pending = null;
      if (next) this.state = next;
    });
  }
}

const empty: ReaderTranslationState = {
  phase: 'idle', prepare: { done: 0, total: 0, preparedPages: 0 }, translate: { done: 0, total: 0 }, model: { state: 'idle' }, sourceLanguage: null, targetLanguage: 'vi', hidden: false, failed: 0,
};
