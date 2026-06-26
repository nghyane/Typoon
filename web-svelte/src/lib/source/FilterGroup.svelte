<!--
  FilterGroup — one source filter rendered as chips.
    - nsfw options    → standalone toggle chips (e.g. 18+)
    - the rest (≥ 2)  → a trigger chip + dropdown checklist
  `select` filters show radio indicators (one active); `multi` show checkboxes.
  `inject: 'path'` filters are "required": clicking the active option keeps it,
  so the `{filterPath}` placeholder never resolves empty.
-->
<script lang="ts">
  import { Check, ChevronDown } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import ReaderDropdown from '$lib/reader/ReaderDropdown.svelte';
  import type { Filter } from './types';

  type State = Record<string, string | string[]>;

  let { filter, selection, onChange }: {
    filter: Filter;
    selection: State;
    onChange: (next: State) => void;
  } = $props();

  let open = $state(false);
  let anchor = $state<HTMLButtonElement | null>(null);

  const active = $derived.by((): string[] => {
    const v = selection[filter.id];
    if (!v) return [];
    return Array.isArray(v) ? v : [v];
  });

  const nsfwOpts = $derived(filter.options.filter((o) => o.nsfw));
  const normalOpts = $derived(filter.options.filter((o) => !o.nsfw));
  const activeNormal = $derived(active.filter((id) => normalOpts.some((o) => o.id === id)));

  const label = $derived(
    activeNormal.length === 1
      ? (normalOpts.find((o) => o.id === activeNormal[0])?.label ?? filter.label)
      : activeNormal.length > 1
        ? `${activeNormal.length} ${filter.label.toLowerCase()}`
        : filter.label,
  );

  function toggle(optId: string): void {
    const next = { ...selection };
    if (filter.type === 'select') {
      const required = filter.inject === 'path';
      if (selection[filter.id] === optId) {
        if (!required) delete next[filter.id];
      } else {
        next[filter.id] = optId;
      }
    } else {
      const updated = active.includes(optId) ? active.filter((x) => x !== optId) : [...active, optId];
      if (updated.length === 0) delete next[filter.id];
      else next[filter.id] = updated;
    }
    onChange(next);
  }
</script>

{#if normalOpts.length > 1}
  <button
    type="button"
    bind:this={anchor}
    onclick={() => (open = !open)}
    class={cn(
      'inline-flex items-center gap-1 h-8 px-3 rounded-full text-sm font-medium whitespace-nowrap shrink-0 cursor-pointer transition-colors',
      activeNormal.length > 0 || open ? 'bg-surface-2 text-text' : 'text-text-muted hover:text-text',
    )}
  >
    {label}
    <ChevronDown size={12} class={cn('transition-transform', open && 'rotate-180')} />
  </button>

  <ReaderDropdown {open} {anchor} onClose={() => (open = false)} align="start" width="min(16rem, calc(100vw - 1rem))" widthPx={256}>
    <div class="py-1 max-h-72 overflow-y-auto">
      {#each normalOpts as opt (opt.id)}
        {@const checked = active.includes(opt.id)}
        <button
          type="button"
          onclick={() => toggle(opt.id)}
          role="menuitemcheckbox"
          aria-checked={checked}
          class={cn(
            'w-full flex items-center gap-2.5 px-3 py-1.5 text-sm text-left transition-colors cursor-pointer',
            checked ? 'text-text' : 'text-text-muted hover:text-text hover:bg-hover',
          )}
        >
          {#if filter.type === 'multi'}
            <span class={cn('size-3.5 rounded-xs border flex items-center justify-center shrink-0 transition-colors', checked ? 'bg-accent border-accent' : 'border-border-strong')}>
              {#if checked}<Check size={9} strokeWidth={3} class="text-accent-text" />{/if}
            </span>
          {:else}
            <span class={cn('size-3.5 rounded-full border flex items-center justify-center shrink-0 transition-colors', checked ? 'border-accent' : 'border-border-strong')}>
              {#if checked}<span class="size-2 rounded-full bg-accent"></span>{/if}
            </span>
          {/if}
          {opt.label}
        </button>
      {/each}
    </div>
  </ReaderDropdown>
{/if}

{#each nsfwOpts as opt (opt.id)}
  {@const on = active.includes(opt.id)}
  <button
    type="button"
    onclick={() => toggle(opt.id)}
    class={cn(
      'inline-flex items-center h-8 px-3 rounded-full text-sm font-medium whitespace-nowrap shrink-0 cursor-pointer transition-colors',
      on ? 'bg-error-bg text-error-text' : 'text-text-muted hover:text-text',
    )}
  >{opt.label}</button>
{/each}
