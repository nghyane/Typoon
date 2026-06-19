<script lang="ts">
  import { cn } from '$lib/cn';

  let {
    open,
    anchor,
    onClose,
    align = 'end',
    side = 'auto',
    width = 'min(24rem, calc(100vw - 1rem))',
    children,
  }: {
    open: boolean;
    anchor: HTMLElement | null;
    onClose: () => void;
    align?: 'start' | 'end';
    side?: 'auto' | 'top' | 'bottom';
    width?: string;
    children: import('svelte').Snippet;
  } = $props();

  let panel = $state<HTMLDivElement | null>(null);
  let style = $state('');

  function updatePosition(): void {
    if (!anchor) return;
    const rect = anchor.getBoundingClientRect();
    const gap = 8;
    const margin = 8;
    const panelWidth = Math.min(384, window.innerWidth - margin * 2);
    const spaceBelow = window.innerHeight - rect.bottom - gap - margin;
    const spaceAbove = rect.top - gap - margin;
    const openTop = side === 'top' || (side === 'auto' && spaceAbove > spaceBelow && spaceBelow < 220);
    const maxHeight = Math.max(160, openTop ? spaceAbove : spaceBelow);
    const left = align === 'start'
      ? Math.max(margin, Math.min(rect.left, window.innerWidth - panelWidth - margin))
      : Math.max(margin, Math.min(rect.right - panelWidth, window.innerWidth - panelWidth - margin));
    const measuredHeight = Math.min(panel?.offsetHeight || Math.min(360, maxHeight), maxHeight);
    const top = openTop
      ? Math.max(margin, rect.top - gap - measuredHeight)
      : Math.min(rect.bottom + gap, window.innerHeight - margin);
    style = `left:${left}px;top:${top}px;width:${width};max-width:calc(100vw - ${margin * 2}px);max-height:${maxHeight}px;`;
  }

  $effect(() => {
    if (!open || !anchor) return;
    updatePosition();
    const frame = requestAnimationFrame(updatePosition);
    const onKeydown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && (panel?.contains(target) || anchor.contains(target))) return;
      onClose();
    };
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    document.addEventListener('keydown', onKeydown);
    document.addEventListener('pointerdown', onPointerDown, true);
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
      document.removeEventListener('keydown', onKeydown);
      document.removeEventListener('pointerdown', onPointerDown, true);
      cancelAnimationFrame(frame);
    };
  });
</script>

{#if open && anchor}
  <div
    bind:this={panel}
    class={cn('fixed z-50 overflow-hidden rounded-lg border border-divider bg-surface text-text shadow-2xl')}
    style={style}
    role="menu"
  >
    {@render children()}
  </div>
{/if}
