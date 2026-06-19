<script lang="ts">
  import { X } from 'lucide-svelte';
  import { cn } from '$lib/cn';

  let {
    open,
    onClose,
    title,
    size = 'md',
    children,
    footer,
    footerLeft,
  }: {
    open: boolean;
    onClose: () => void;
    title: string;
    size?: 'sm' | 'md' | 'lg';
    children: import('svelte').Snippet;
    footer?: import('svelte').Snippet;
    footerLeft?: import('svelte').Snippet;
  } = $props();

  const sizes = {
    sm: 'max-w-md',
    md: 'max-w-2xl',
    lg: 'max-w-4xl',
  } as const;

  $effect(() => {
    if (!open) return;
    const onKeydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeydown);
    return () => document.removeEventListener('keydown', onKeydown);
  });
</script>

{#if open}
  <div
    class="fixed inset-0 flex items-center justify-center bg-black/60 p-4 pt-[max(1rem,var(--sait))] pb-[max(1rem,var(--saib))] pl-[max(1rem,var(--sail))] pr-[max(1rem,var(--sair))] z-50"
    onmousedown={(event) => { if (event.target === event.currentTarget) onClose(); }}
    role="presentation"
  >
    <div class={cn('w-full bg-surface text-text rounded-md border border-border-soft flex flex-col max-h-[88vh] overflow-hidden', sizes[size])}>
      <header class="flex items-center justify-between px-5 h-[52px] border-b border-border-soft shrink-0">
        <h2 class="text-base font-semibold text-text tracking-tight truncate">{title}</h2>
        <button type="button" onclick={onClose} class="size-8 rounded-sm flex items-center justify-center text-text-subtle hover:text-text hover:bg-hover transition-colors cursor-pointer" aria-label="Đóng">
          <X size={15} />
        </button>
      </header>

      <div class="flex-1 overflow-auto overscroll-contain">
        {@render children()}
      </div>

      {#if footer || footerLeft}
        <footer class="flex items-center gap-3 px-5 py-3 border-t border-border-soft bg-bg/40 shrink-0">
          <div class="flex-1 min-w-0 text-xs text-text-subtle truncate">
            {#if footerLeft}{@render footerLeft()}{/if}
          </div>
          <div class="flex items-center gap-2 shrink-0">
            {#if footer}{@render footer()}{/if}
          </div>
        </footer>
      {/if}
    </div>
  </div>
{/if}
