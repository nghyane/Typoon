<script lang="ts">
  import type { Component } from 'svelte';

  let {
    icon: Icon,
    title,
    hint,
    action,
  }: {
    icon?: Component<{ size?: number; class?: string }>;
    title: string;
    hint?: string | null;
    action?: import('svelte').Snippet;
  } = $props();
</script>

<div class="py-16 flex flex-col items-center text-center">
  {#if Icon}
    <div class="size-12 rounded-md bg-surface-2 flex items-center justify-center mb-3">
      <Icon size={20} class="text-text-subtle" />
    </div>
  {/if}
  <p class="text-sm font-medium text-text">{title}</p>
  {#if hint}<p class="text-xs text-text-subtle mt-1 max-w-sm">{hint}</p>{/if}
  {#if action}<div class="mt-4">{@render action()}</div>{/if}
</div>
