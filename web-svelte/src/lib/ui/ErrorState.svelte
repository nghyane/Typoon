<script lang="ts">
  import { RotateCw } from 'lucide-svelte';

  let {
    title = 'Có lỗi xảy ra',
    message = '',
    onRetry,
    retrying = false,
  }: {
    title?: string;
    message?: string;
    onRetry?: () => void;
    retrying?: boolean;
  } = $props();
</script>

<div class="flex flex-col items-center justify-center gap-3 py-16 text-center">
  <p class="text-sm font-medium text-error-text">{title}</p>
  {#if message}<p class="max-w-sm text-xs leading-relaxed text-text-subtle">{message}</p>{/if}
  {#if onRetry}
    <button
      type="button"
      onclick={onRetry}
      disabled={retrying}
      class="inline-flex h-8 items-center gap-1.5 rounded-sm bg-surface-2 px-3 text-sm font-medium text-text transition-colors hover:bg-hover disabled:opacity-60 disabled:cursor-wait cursor-pointer"
    >
      <RotateCw size={14} class={retrying ? 'animate-spin' : ''} /> Thử lại
    </button>
  {/if}
</div>
