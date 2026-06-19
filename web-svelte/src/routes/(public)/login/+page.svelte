<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { AlertCircle } from 'lucide-svelte';
  import { discordActivityLogin, isDiscordActivity, safeReturnTo, startDiscordLogin } from '$lib/auth/api';
  import { session } from '$lib/auth/session.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';

  let authorizing = $state(false);
  let activityError = $state<string | null>(null);
  let daFired = $state(false);

  const redirect = $derived(safeReturnTo($page.url.searchParams.get('redirect')));
  const searchError = $derived($page.url.searchParams.get('error'));
  const error = $derived(activityError ?? searchError);

  onMount(() => { void session.load(); });

  $effect(() => {
    if (session.state.status === 'authenticated') goto(redirect, { replaceState: true });
  });
  $effect(() => {
    if (!isDiscordActivity || session.state.status !== 'unauthenticated' || daFired) return;
    daFired = true;
    authorizing = true;
    withTimeout(discordActivityLogin(), 15_000)
      .then(async () => {
        await session.refresh();
        goto(redirect, { replaceState: true });
      })
      .catch((err: Error) => {
        activityError = err.message;
        authorizing = false;
      });
  });

  function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => reject(new Error('Discord không phản hồi, thử lại.')), ms);
      promise.then(resolve, reject).finally(() => window.clearTimeout(timer));
    });
  }

  function beginLogin(): void {
    if (authorizing) return;
    authorizing = true;
    startDiscordLogin(redirect).catch((err: Error) => {
      activityError = err.message || 'Không thể bắt đầu đăng nhập Discord.';
      authorizing = false;
    });
  }
</script>

<svelte:head><title>Đăng nhập — Hội Mê Truyện</title></svelte:head>

{#if isDiscordActivity && session.state.status === 'loading'}
  <div class="min-h-screen flex items-center justify-center bg-bg p-4">
    <div class="w-full max-w-xs p-6 bg-surface rounded-md text-center space-y-4">
      <Spinner size={24} class="text-text mx-auto" />
      <p class="text-sm text-text-muted">Đang kiểm tra đăng nhập…</p>
    </div>
  </div>
{:else if isDiscordActivity && session.state.status === 'error'}
  <div class="min-h-screen flex items-center justify-center bg-bg">
    <div class="w-full max-w-xs p-6 bg-surface rounded-md text-center space-y-4">
      <div class="text-sm text-error-text">{session.state.error.message}</div>
      <button type="button" onclick={() => session.refresh()} class="w-full h-9 rounded-sm bg-[#5865F2] text-white text-sm font-medium cursor-pointer">Thử lại</button>
    </div>
  </div>
{:else if isDiscordActivity && !error}
  <div class="min-h-screen flex items-center justify-center bg-bg p-4">
    <div class="w-full max-w-xs p-6 bg-surface rounded-md text-center space-y-4">
      <Spinner size={24} class="text-text mx-auto" />
      <p class="text-sm text-text-muted">Đang đăng nhập Discord…</p>
    </div>
  </div>
{:else if isDiscordActivity && error}
  <div class="min-h-screen flex items-center justify-center bg-bg">
    <div class="w-full max-w-xs p-6 bg-surface rounded-md text-center space-y-4">
      <div class="text-sm text-error-text">{error}</div>
      <button type="button" onclick={() => { activityError = null; daFired = false; }} class="w-full h-9 rounded-sm bg-[#5865F2] text-white text-sm font-medium cursor-pointer">Thử lại</button>
    </div>
  </div>
{:else}
  <div class="min-h-screen flex items-center justify-center bg-bg p-4">
    <div class="w-full max-w-sm">
      <div class="bg-surface rounded-md p-6 border border-border-soft">
        {#if error}
          <div class="mb-4 p-3 rounded-sm bg-error-bg text-sm text-error-text">
            <div class="flex items-start gap-2"><AlertCircle size={14} class="shrink-0 mt-0.5" /><span class="break-words">{error}</span></div>
          </div>
        {/if}
        <p class="text-sm text-text-muted mb-4">Đăng nhập bằng Discord để tiếp tục.</p>
        <button type="button" aria-disabled={authorizing} onclick={beginLogin} class="w-full inline-flex items-center justify-center gap-2 h-10 px-4 rounded-sm bg-[#5865F2] text-white text-sm font-medium hover:bg-[#4752C4] active:scale-[0.98] aria-disabled:opacity-60 aria-disabled:cursor-not-allowed transition-all cursor-pointer">
          {@render DiscordIcon()}
          {authorizing ? 'Đang chuyển hướng…' : 'Đăng nhập với Discord'}
        </button>
      </div>
    </div>
  </div>
{/if}

{#snippet DiscordIcon()}
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
  </svg>
{/snippet}
