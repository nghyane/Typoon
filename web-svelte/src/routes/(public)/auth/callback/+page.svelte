<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { exchangeDiscordCallback } from '$lib/auth/api';
  import { session } from '$lib/auth/session.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';

  let error = $state('');
  let fired = $state(false);

  $effect(() => {
    if (fired) return;
    fired = true;
    const fragment = new URLSearchParams($page.url.hash.replace(/^#/, ''));
    const query = $page.url.searchParams;
    if (!fragment.get('access_token') && !fragment.get('error') && !query.get('code') && !query.get('error')) {
      error = 'Callback Discord thiếu access token hoặc authorization code.';
      return;
    }
    exchangeDiscordCallback(fragment, query)
      .then(async (returnTo) => {
        await session.refresh();
        goto(returnTo, { replaceState: true });
      })
      .catch((err: Error) => { error = err.message || 'Không thể hoàn tất đăng nhập.'; });
  });
</script>

<svelte:head><title>Đăng nhập — Hội Mê Truyện</title></svelte:head>

{#if error}
  <div class="min-h-screen flex items-center justify-center bg-bg p-4">
    <div class="bg-surface rounded-md p-6 border border-border-soft text-center space-y-4 w-full max-w-sm">
      <div class="text-sm text-error-text">Lỗi đăng nhập: {error}</div>
      <a href={`/login?error=${encodeURIComponent(error)}`} class="inline-flex px-4 py-2 rounded-sm bg-[#5865F2] text-white text-sm hover:bg-[#4752C4]">Quay lại trang đăng nhập</a>
    </div>
  </div>
{:else}
  <div class="min-h-screen flex items-center justify-center bg-bg"><Spinner size={24} /></div>
{/if}
