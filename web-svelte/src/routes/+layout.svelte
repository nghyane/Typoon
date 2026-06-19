<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { QueryClientProvider } from '@tanstack/svelte-query';
  import { queryClient } from '$lib/queryClient';
  import { pwaInstall } from '$lib/pwa/installPrompt.svelte';
  import { registerServiceWorker } from '$lib/pwa/serviceWorker';
  let { children }: { children: import('svelte').Snippet } = $props();

  onMount(() => {
    document.documentElement.classList.toggle('da-host', window.location.hostname.endsWith('.discordsays.com'));
    pwaInstall.init();
    registerServiceWorker();
  });
</script>

<QueryClientProvider client={queryClient}>
  {@render children()}
</QueryClientProvider>
