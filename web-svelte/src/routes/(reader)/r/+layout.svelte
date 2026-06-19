<script lang="ts">
  let { children }: { children: import('svelte').Snippet } = $props();
</script>

<div class="min-h-dvh bg-bg text-text text-sm">
  {@render children()}
</div>
