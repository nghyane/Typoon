<script lang="ts">
  import {
    attachOverlay,
    ComlinkVisionRuntime,
    DeepLTranslateWeb,
    detectBrowserCapabilities,
    ensureMangaFontLoaded,
    LensTextRecognizer,
    MainThreadVisionRuntime,
    TranslationRuntime,
    type PageOverlay,
    type PreparedPageHandle,
    type PreparedTextResult,
    type PreparedTranslationResult,
    type TranslationRunEvent,
  } from '@typoon/client/web-reader';

  type Slot = 'a' | 'b' | 'seam';

  let sourceLang = $state('zh');
  let targetLang = $state('vi');
  let busy = $state(false);
  let logs = $state<string[]>([]);
  let runtime = $state<TranslationRuntime | null>(null);
  let session = $state<Awaited<ReturnType<TranslationRuntime['createStageSession']>> | null>(null);

  let fileA = $state<File | null>(null);
  let fileB = $state<File | null>(null);
  let handleA = $state<PreparedPageHandle | null>(null);
  let handleB = $state<PreparedPageHandle | null>(null);
  let seamHandle = $state<PreparedPageHandle | null>(null);
  let selected = $state<Slot>('a');
  let textResult = $state<PreparedTextResult | null>(null);
  let translated = $state<PreparedTranslationResult | null>(null);
  let overlays = $state<readonly PageOverlay[]>([]);

  let canvasA = $state<HTMLCanvasElement | null>(null);
  let canvasB = $state<HTMLCanvasElement | null>(null);
  let hostA = $state<HTMLDivElement | null>(null);
  let hostB = $state<HTMLDivElement | null>(null);
  let sizeA = $state<[number, number] | null>(null);
  let sizeB = $state<[number, number] | null>(null);

  function log(message: string): void {
    logs = [`${new Date().toLocaleTimeString()} ${message}`, ...logs].slice(0, 80);
  }

  async function runStep(label: string, fn: () => Promise<void>): Promise<void> {
    if (busy) return;
    busy = true;
    try {
      log(`→ ${label}`);
      await fn();
      log(`✓ ${label}`);
    } catch (error) {
      log(`✗ ${label}: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      busy = false;
    }
  }

  async function ensureSession(): Promise<NonNullable<typeof session>> {
    if (session) return session;
    log('init: font');
    await ensureMangaFontLoaded();
    log('init: caps');
    const caps = detectBrowserCapabilities();
    log('init: vision');
    const vision = typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined'
      ? new ComlinkVisionRuntime(new Worker(new URL('@typoon/client/vision/vision.worker.ts', import.meta.url), { type: 'module' }))
      : new MainThreadVisionRuntime({});
    log('init: runtime');
    runtime = new TranslationRuntime({
      vision,
      recognizer: new LensTextRecognizer({ requestTimeoutMs: 20_000 }),
      translator: new DeepLTranslateWeb({ maxSessions: caps.isMobile ? 2 : 3 }),
      concurrency: { load: 2, prepare: 1, ocr: 2, detect: 1, translate: 2, compose: 2, maxPreparedPages: 3 },
    });
    log('init: beginPreparation');
    session = await runtime.createStageSession({ type: 'identity' });
    log(`session ${session.runId}`);
    return session;
  }

  function pickFile(event: Event, slot: 'a' | 'b'): void {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    if (slot === 'a') { fileA = file; handleA = null; }
    else { fileB = file; handleB = null; }
    if (file) void drawFile(file, slot);
  }

  async function drawFile(file: File, slot: 'a' | 'b'): Promise<void> {
    const canvas = slot === 'a' ? canvasA : canvasB;
    if (!canvas) return;
    const bitmap = await createImageBitmap(file);
    try {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d')?.drawImage(bitmap, 0, 0);
      if (slot === 'a') sizeA = [bitmap.width, bitmap.height];
      else sizeB = [bitmap.width, bitmap.height];
    } finally {
      bitmap.close();
    }
  }

  async function prepare(slot: 'a' | 'b'): Promise<void> {
    const s = await ensureSession();
    const file = slot === 'a' ? fileA : fileB;
    if (!file) throw new Error(`missing page ${slot.toUpperCase()}`);
    const [handle] = await s.preparePage({ index: slot === 'a' ? 0 : 1, blob: file });
    if (!handle) throw new Error('prepare emitted no handle');
    if (slot === 'a') handleA = handle;
    else handleB = handle;
    selected = slot;
    log(`${slot.toUpperCase()} handle ${handle.preparedPageId} ${handle.size.width}x${handle.size.height}`);
  }

  async function createSeam(): Promise<void> {
    const s = await ensureSession();
    if (!handleA || !handleB) throw new Error('prepare A and B first');
    seamHandle = await s.createSeamRepair(handleA, handleB, 320);
    if (!seamHandle) throw new Error('seam has no text-like content');
    selected = 'seam';
    log(`seam handle ${seamHandle.preparedPageId} ${seamHandle.size.width}x${seamHandle.size.height}`);
  }

  function currentHandle(): PreparedPageHandle {
    const handle = selected === 'a' ? handleA : selected === 'b' ? handleB : seamHandle;
    if (!handle) throw new Error(`missing ${selected} handle`);
    return handle;
  }

  async function recognize(): Promise<void> {
    const s = await ensureSession();
    textResult = await s.recognize(currentHandle(), sourceLang || null);
    translated = null;
    overlays = [];
    log(`OCR blocks=${textResult.recognized.blocks.length} units=${textResult.textUnits.length}`);
  }

  async function translate(): Promise<void> {
    const s = await ensureSession();
    if (!textResult) throw new Error('OCR first');
    translated = await s.translate(textResult, sourceLang || null, targetLang);
    log(`translated=${translated.translations.filter(unit => unit.kind !== 'skip').length}`);
  }

  async function compose(): Promise<void> {
    const s = await ensureSession();
    if (!translated) throw new Error('translate first');
    overlays = await s.composeOverlay(currentHandle(), translated);
    attachAll(overlays);
    log(`overlays=${overlays.length}`);
  }

  async function runFullRuntime(): Promise<void> {
    if (!fileA || !fileB) throw new Error('upload A and B first');
    await ensureMangaFontLoaded();
    const caps = detectBrowserCapabilities();
    const rt = new TranslationRuntime({
      vision: typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined'
        ? new ComlinkVisionRuntime(new Worker(new URL('@typoon/client/vision/vision.worker.ts', import.meta.url), { type: 'module' }))
        : new MainThreadVisionRuntime({}),
      recognizer: new LensTextRecognizer({ requestTimeoutMs: 20_000 }),
      translator: new DeepLTranslateWeb({ maxSessions: caps.isMobile ? 2 : 3 }),
      concurrency: { load: 2, prepare: 1, ocr: 2, detect: 1, translate: 2, compose: 2, maxPreparedPages: 3 },
    });
    const files = [fileA, fileB] as const;
    const run = rt.createTranslationRun({
      pageCount: files.length,
      readPage: async (index: number) => ({ index, blob: files[index]! }),
    }, {
      sourceLanguage: sourceLang || null,
      targetLanguage: targetLang,
      scope: 'all',
      priority: { aroundPageIndex: 0 },
      preparation: { type: 'continuous-strip' },
    });
    const unsub = run.subscribe((event: TranslationRunEvent) => {
      if (event.type === 'page-status') log(`run page ${event.pageIndex + 1}: ${event.status}${event.error ? ` ${event.error.message}` : ''}`);
      if (event.type === 'progress') log(`run progress ${event.progress.done}/${event.progress.total}`);
      if (event.type === 'page-overlay') log(`run overlay page ${event.overlay.pageIndex + 1} placements=${event.overlay.placements.length}`);
      if (event.type === 'failed') log(`run failed ${event.error.message}`);
      if (event.type === 'completed') log(`run completed overlays=${event.overlays.length}`);
    });
    try {
      run.start();
      const result = await run.done;
      overlays = result;
      attachAll(result);
    } finally {
      unsub();
      rt.dispose();
    }
  }

  function attachAll(items: readonly PageOverlay[]): void {
    hostA?.querySelectorAll('[data-typoon-overlay="true"]').forEach(node => node.remove());
    hostB?.querySelectorAll('[data-typoon-overlay="true"]').forEach(node => node.remove());
    for (const overlay of items) {
      const host = overlay.pageIndex === 0 ? hostA : overlay.pageIndex === 1 ? hostB : null;
      if (!host) continue;
      attachOverlay(host, {
        pageSize: [overlay.pageSize.width, overlay.pageSize.height],
        placements: overlay.placements,
        translations: overlay.translations,
        placementMargins: overlay.placementMargins,
      }, { debug: { showDrawable: true, showTextBoxes: true, showTextBounds: true } });
    }
  }

  function reset(): void {
    session?.dispose();
    runtime?.dispose();
    session = null;
    runtime = null;
    handleA = null;
    handleB = null;
    seamHandle = null;
    textResult = null;
    translated = null;
    overlays = [];
    hostA?.querySelectorAll('[data-typoon-overlay="true"]').forEach(node => node.remove());
    hostB?.querySelectorAll('[data-typoon-overlay="true"]').forEach(node => node.remove());
    log('reset');
  }
</script>

<svelte:head><title>Runtime Dev — Hội Mê Truyện</title></svelte:head>

<div class="mx-auto max-w-7xl px-4 py-6 text-text space-y-4">
  <div class="flex items-center justify-between gap-3">
    <div>
      <h1 class="text-lg font-semibold">Runtime Dev</h1>
      <p class="text-sm text-text-subtle">Test từng stage: prepare → seam → OCR → translate → compose.</p>
    </div>
    <button class="h-8 px-3 rounded-sm bg-surface-2 text-sm" onclick={reset}>Reset</button>
  </div>

  <div class="grid md:grid-cols-[1fr_20rem] gap-4">
    <div class="grid md:grid-cols-2 gap-4">
      <section class="rounded-md border border-border-soft bg-surface p-3 space-y-2">
        <div class="flex items-center justify-between gap-2">
          <h2 class="text-sm font-semibold">Page A</h2>
          <input type="file" accept="image/*" onchange={(event) => pickFile(event, 'a')} class="text-xs" />
        </div>
        <div class="relative w-full overflow-hidden rounded-sm bg-bg" style={`aspect-ratio:${sizeA ? `${sizeA[0]}/${sizeA[1]}` : '3/4'}`}>
          <canvas bind:this={canvasA} class="absolute inset-0 h-full w-full"></canvas>
          <div bind:this={hostA} class="absolute inset-0 pointer-events-none"></div>
        </div>
      </section>

      <section class="rounded-md border border-border-soft bg-surface p-3 space-y-2">
        <div class="flex items-center justify-between gap-2">
          <h2 class="text-sm font-semibold">Page B</h2>
          <input type="file" accept="image/*" onchange={(event) => pickFile(event, 'b')} class="text-xs" />
        </div>
        <div class="relative w-full overflow-hidden rounded-sm bg-bg" style={`aspect-ratio:${sizeB ? `${sizeB[0]}/${sizeB[1]}` : '3/4'}`}>
          <canvas bind:this={canvasB} class="absolute inset-0 h-full w-full"></canvas>
          <div bind:this={hostB} class="absolute inset-0 pointer-events-none"></div>
        </div>
      </section>
    </div>

    <aside class="space-y-3 rounded-md border border-border-soft bg-surface p-3">
      <div class="grid grid-cols-2 gap-2">
        <label class="text-xs text-text-subtle">Source<input bind:value={sourceLang} class="mt-1 w-full rounded-sm bg-bg px-2 py-1 text-text" /></label>
        <label class="text-xs text-text-subtle">Target<input bind:value={targetLang} class="mt-1 w-full rounded-sm bg-bg px-2 py-1 text-text" /></label>
      </div>

      <div class="grid grid-cols-2 gap-2">
        <button class="h-8 rounded-sm bg-surface-2 text-sm disabled:opacity-50" disabled={busy || !fileA} onclick={() => runStep('prepare A', () => prepare('a'))}>Prepare A</button>
        <button class="h-8 rounded-sm bg-surface-2 text-sm disabled:opacity-50" disabled={busy || !fileB} onclick={() => runStep('prepare B', () => prepare('b'))}>Prepare B</button>
      </div>

      <button class="h-8 w-full rounded-sm bg-surface-2 text-sm disabled:opacity-50" disabled={busy || !handleA || !handleB} onclick={() => runStep('create seam', createSeam)}>Create seam window</button>

      <div class="grid grid-cols-3 gap-2 text-sm">
        <button class:selected={selected === 'a'} class="h-8 rounded-sm bg-bg" onclick={() => { selected = 'a'; }}>A</button>
        <button class:selected={selected === 'b'} class="h-8 rounded-sm bg-bg" onclick={() => { selected = 'b'; }}>B</button>
        <button class:selected={selected === 'seam'} class="h-8 rounded-sm bg-bg" onclick={() => { selected = 'seam'; }}>Seam</button>
      </div>

      <div class="grid gap-2">
        <button class="h-8 rounded-sm bg-accent text-accent-fg text-sm disabled:opacity-50" disabled={busy} onclick={() => runStep(`ocr ${selected}`, recognize)}>OCR selected</button>
        <button class="h-8 rounded-sm bg-accent text-accent-fg text-sm disabled:opacity-50" disabled={busy || !textResult} onclick={() => runStep('translate', translate)}>Translate</button>
        <button class="h-8 rounded-sm bg-accent text-accent-fg text-sm disabled:opacity-50" disabled={busy || !translated} onclick={() => runStep('compose overlay', compose)}>Compose overlay</button>
        <button class="h-8 rounded-sm bg-success-bg text-success-text text-sm disabled:opacity-50" disabled={busy || !fileA || !fileB} onclick={() => runStep('strip runtime A+B', runFullRuntime)}>Run strip runtime A+B</button>
      </div>

      <div class="rounded-sm bg-bg p-2 text-xs space-y-1 max-h-48 overflow-auto">
        {#each logs as line}<div>{line}</div>{/each}
      </div>
    </aside>
  </div>

  {#if textResult}
    <section class="rounded-md border border-border-soft bg-surface p-3">
      <h2 class="text-sm font-semibold mb-2">OCR text</h2>
      <pre class="text-xs whitespace-pre-wrap text-text-subtle">{textResult.textUnits.map(unit => unit.sourceText).join('\n\n')}</pre>
    </section>
  {/if}
</div>

<style>
  .selected { outline: 1px solid currentColor; }
</style>
