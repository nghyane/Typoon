<script lang="ts">
  import { Languages, ScrollText } from 'lucide-svelte';
  import { localSettings, READER_PAGE_WIDTH_MAX, READER_PAGE_WIDTH_MIN } from '$lib/localSettings.svelte';
  import ReaderDropdown from './ReaderDropdown.svelte';

  let { open, anchor, onClose }: { open: boolean; anchor: HTMLElement | null; onClose: () => void } = $props();

  function setPageWidth(value: number): void {
    localSettings.update({ reader_page_width: value });
  }
</script>

<ReaderDropdown {open} {anchor} {onClose} align="end" width="min(23rem, calc(100vw - 1rem))" widthPx={368}>
  <div class="px-4 py-3 border-b border-divider">
    <div class="text-sm font-semibold text-text">Cài đặt đọc</div>
    <div class="text-xs text-text-subtle">Tối ưu cho dịch overlay</div>
  </div>

  <div class="px-4 py-4 space-y-5 overflow-y-auto max-h-[min(70dvh,30rem)]">
    <section class="space-y-3">
      <div class="flex items-center gap-2 rounded-md bg-surface-2 px-3 py-3">
        <div class="grid size-9 shrink-0 place-items-center rounded-md bg-accent-bg text-accent">
          <ScrollText size={18} />
        </div>
        <div class="min-w-0">
          <div class="text-sm font-semibold text-text">Cuộn dọc tối ưu dịch thuật</div>
          <p class="mt-0.5 text-xs leading-relaxed text-text-muted">
            Reader dùng một luồng page-local duy nhất để overlay dịch, OCR và vị trí chữ bám đúng từng trang.
          </p>
        </div>
      </div>
    </section>

    <div class="border-t border-border-soft"></div>

    <section class="space-y-3">
      <div class="flex items-baseline justify-between gap-2">
        <h3 class="text-xs uppercase tracking-wider text-text-subtle font-semibold">Hiển thị</h3>
      </div>
      <div class="space-y-3">
        <div class="flex items-start gap-2 rounded-md border border-border-soft bg-surface px-3 py-2 text-xs text-text-muted">
          <Languages size={14} class="mt-0.5 shrink-0 text-accent" />
          <span>Dịch chạy theo vùng đang đọc trước, sau đó hoàn tất phần còn lại của chương.</span>
        </div>
        <div class="flex items-baseline justify-between gap-2">
          <span class="text-sm text-text">Bề rộng tối đa</span>
          <span class="text-xs tabular-nums text-text-muted h-6 px-2 inline-flex items-center rounded-full bg-surface-2">
            {localSettings.state.reader_page_width} px
          </span>
        </div>
        <input
          type="range"
          min={READER_PAGE_WIDTH_MIN}
          max={READER_PAGE_WIDTH_MAX}
          step="20"
          value={localSettings.state.reader_page_width}
          oninput={(event) => setPageWidth(Number(event.currentTarget.value))}
          class="reader-slider w-full h-1 appearance-none rounded-full bg-surface-2 cursor-pointer accent-accent"
        />
      </div>
    </section>
  </div>
</ReaderDropdown>
