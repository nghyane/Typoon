<script lang="ts">
  import { Loader2, RefreshCw, X } from 'lucide-svelte';
  import { pwaUpdate } from './updatePrompt.svelte';

  const applying = $derived(pwaUpdate.state.phase === 'applying');
  const stalled = $derived(pwaUpdate.state.phase === 'stalled');
</script>

{#if pwaUpdate.state.available}
  <div class="fixed inset-x-0 top-[max(0.5rem,var(--sait))] z-[80] flex justify-center px-3 pointer-events-none">
    <div class="pointer-events-auto flex max-w-[min(42rem,calc(100vw-1.5rem))] items-center gap-2 rounded-md border border-accent/20 bg-surface/95 px-3 py-2 text-sm text-text shadow-lg backdrop-blur">
      <span class="inline-flex size-7 shrink-0 items-center justify-center rounded-full bg-accent-bg text-accent-text">
        {#if applying}
          <Loader2 size={14} class="animate-spin" />
        {:else}
          <RefreshCw size={14} />
        {/if}
      </span>
      <span class="min-w-0 flex-1 leading-tight">
        <span class="block font-medium">
          {stalled ? 'Cập nhật lâu hơn bình thường' : applying ? 'Đang cập nhật…' : 'Có bản cập nhật mới'}
        </span>
        <span class="block text-xs text-text-subtle">
          {#if stalled}
            {pwaUpdate.state.error ?? 'Bạn có thể tải lại ngay để dùng bản mới.'}
          {:else if applying}
            Đang kích hoạt bản mới. Trang sẽ tự tải lại ngay khi sẵn sàng.
          {:else}
            Bản mới đã sẵn sàng, cập nhật chỉ mất vài giây.
          {/if}
        </span>
      </span>
      <button type="button" onclick={() => pwaUpdate.apply()} disabled={applying} class="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-sm bg-accent px-3 text-xs font-semibold text-accent-fg transition hover:brightness-110 active:scale-[0.98] disabled:cursor-wait disabled:opacity-80 cursor-pointer">
        {#if applying}<Loader2 size={13} class="animate-spin" />{/if}
        {applying ? 'Đang cập nhật' : stalled ? 'Thử lại' : 'Cập nhật'}
      </button>
      {#if stalled}
        <button type="button" onclick={() => pwaUpdate.forceReload()} class="h-8 shrink-0 rounded-sm bg-surface-2 px-3 text-xs font-semibold text-text transition hover:bg-hover cursor-pointer">
          Tải lại ngay
        </button>
      {/if}
      {#if !applying}
        <button type="button" aria-label="Ẩn thông báo cập nhật" onclick={() => pwaUpdate.dismiss()} class="inline-flex size-8 shrink-0 items-center justify-center rounded-sm text-text-subtle hover:bg-hover hover:text-text transition-colors cursor-pointer">
          <X size={15} />
        </button>
      {/if}
    </div>
  </div>
{/if}
