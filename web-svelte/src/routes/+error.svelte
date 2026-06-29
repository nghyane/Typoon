<script lang="ts">
  import { page } from '$app/stores';
  import { RotateCw, Home } from 'lucide-svelte';
</script>

<svelte:head><title>Lỗi — Hội Mê Truyện</title></svelte:head>

<div class="min-h-dvh flex flex-col items-center justify-center gap-4 bg-bg px-6 text-center text-text">
  <p class="text-4xl font-bold tabular-nums text-text-subtle">{$page.status}</p>
  <div class="space-y-1">
    <p class="text-base font-semibold">Đã có lỗi xảy ra</p>
    <p class="max-w-sm text-sm text-text-muted">{$page.error?.message ?? 'Trang gặp sự cố không mong muốn.'}</p>
  </div>
  <div class="flex items-center gap-2">
    <button type="button" onclick={() => location.reload()} class="inline-flex h-9 items-center gap-1.5 rounded-sm bg-accent px-4 text-sm font-semibold text-accent-fg transition hover:brightness-110 cursor-pointer">
      <RotateCw size={15} /> Tải lại
    </button>
    <a href="/home" class="inline-flex h-9 items-center gap-1.5 rounded-sm bg-surface-2 px-4 text-sm font-medium text-text transition-colors hover:bg-hover">
      <Home size={15} /> Trang chủ
    </a>
  </div>
</div>
