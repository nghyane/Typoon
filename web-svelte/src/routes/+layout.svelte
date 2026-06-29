<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { afterNavigate } from '$app/navigation';
  import { QueryClientProvider } from '@tanstack/svelte-query';
  import { queryClient } from '$lib/queryClient';
  import { isDiscordActivity } from '$lib/auth/api';
  import { initAnalytics, trackPageView } from '$lib/analytics/client';
  import { pwaInstall } from '$lib/pwa/installPrompt.svelte';
  import { pwaUpdate } from '$lib/pwa/updatePrompt.svelte';
  import UpdateBanner from '$lib/pwa/UpdateBanner.svelte';
  import ToastHost from '$lib/ui/ToastHost.svelte';
  import { registerServiceWorker } from '$lib/pwa/serviceWorker';
  let { children }: { children: import('svelte').Snippet } = $props();

  afterNavigate(({ to }) => {
    const url = to?.url ?? new URL(window.location.href);
    trackPageView(`${url.pathname}${url.search}`);
  });

  onMount(() => {
    document.documentElement.classList.toggle('da-host', isDiscordActivity);
    initAnalytics();
    pwaUpdate.init();
    pwaInstall.init();
    registerServiceWorker();
  });
</script>

<QueryClientProvider client={queryClient}>
  <UpdateBanner />
  <ToastHost />
  {@render children()}
</QueryClientProvider>
