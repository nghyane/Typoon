<script lang="ts">
  import AppShell from '$lib/app/AppShell.svelte';
  let { children }: { children: import('svelte').Snippet } = $props();
</script>

<AppShell>
  {@render children()}
</AppShell>
