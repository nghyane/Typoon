<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { version } from '$app/environment';
  import { Download, Globe2, LogOut, Search, Sparkles } from 'lucide-svelte';
  import { session } from '$lib/auth/session.svelte';
  import { languageName, languageSummary } from '$lib/lang';
  import { localSettings, type ThemeMode } from '$lib/localSettings.svelte';
  import type { TranslationProvider } from '$lib/reader/translation.svelte';
  import { enableDefaultSources, listSources, setSourceEnabled } from '$lib/source/registry';
  import type { InstalledSource } from '$lib/source/types';
  import { cn } from '$lib/cn';
  import Button from '$lib/ui/Button.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import { pwaInstall } from '$lib/pwa/installPrompt.svelte';

  type Tab = 'account' | 'read' | 'sources';

  let tab = $state<Tab>('account');
  let sources = $state<InstalledSource[]>([]);
  let avatarFailed = $state(false);

  let sourceQuery = $state('');
  let sourceLang = $state('all');
  let sourceVisibility = $state<'all' | 'sfw' | 'nsfw'>('all');

  const user = $derived(session.state.status === 'authenticated' ? session.state.user : null);
  const targetLang = $derived(user?.preferred_target_lang ?? localSettings.state.default_target_lang ?? 'vi');
  const installedSourceCount = $derived(sources.filter((source) => source.enabled).length);

  const sourceLanguages = $derived(
    [...new Set(sources.flatMap((source) => source.manifest.languages))].sort((a, b) =>
      languageName(a).localeCompare(languageName(b)),
    ),
  );

  const filteredSources = $derived.by(() => {
    const q = sourceQuery.trim().toLowerCase();
    return sources.filter((source) => {
      const { name, host, id, languages, nsfw } = source.manifest;
      if (q && !`${name} ${host} ${id}`.toLowerCase().includes(q)) return false;
      if (sourceLang !== 'all' && !languages.includes(sourceLang)) return false;
      if (sourceVisibility === 'sfw' && nsfw) return false;
      if (sourceVisibility === 'nsfw' && !nsfw) return false;
      return true;
    });
  });

  const tabs = $derived<Array<{ id: Tab; label: string }>>([
    { id: 'account', label: 'Tài khoản' },
    { id: 'read', label: 'Đọc & Dịch' },
    { id: 'sources', label: `Nguồn (${installedSourceCount}/${sources.length})` },
  ]);

  function parseTab(value: string | null): Tab {
    return value === 'account' || value === 'read' || value === 'sources' ? value : 'account';
  }

  onMount(() => {
    localSettings.load();
    tab = parseTab($page.url.searchParams.get('tab'));
    sources = listSources();
  });

  function refreshSources(): void {
    sources = listSources();
  }

  function toggleSource(source: InstalledSource): void {
    setSourceEnabled(source.manifest.id, !source.enabled);
    refreshSources();
  }

  function installDefaults(): void {
    enableDefaultSources();
    refreshSources();
  }

  function updateTargetLang(lang: string): void {
    localSettings.update({ default_target_lang: lang });
  }

  type ProviderOption = { id: TranslationProvider; label: string; hint: string };
  const providerOptions: ProviderOption[] = [
    { id: 'deepl', label: 'DeepL', hint: 'Chất lượng cao, ổn định' },
    { id: 'google', label: 'Google Dịch', hint: 'Nhanh, phủ nhiều ngôn ngữ' },
  ];

  function updateProvider(provider: TranslationProvider): void {
    localSettings.update({ translation_provider: provider });
  }

  const versionLabel = $derived.by(() => {
    const ts = Number(version);
    if (!Number.isFinite(ts)) return 'Hội Mê Truyện';
    const d = new Date(ts);
    const date = d.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' });
    const time = d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    return `Hội Mê Truyện · ${date} ${time}`;
  });

  const inputCls = 'h-8 w-full px-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors';
</script>

<svelte:head><title>Cài đặt — Hội Mê Truyện</title></svelte:head>

{#if session.state.status === 'loading' || !user}
  <div class="flex items-center justify-center py-16">
    <Spinner size={20} />
  </div>
{:else}
  <div class="max-w-3xl mx-auto px-4 sm:px-6 py-6 space-y-6">
    <header class="space-y-3">
      <h1 class="text-lg font-semibold text-text">Cài đặt</h1>

      <div class="flex flex-wrap gap-2" role="tablist">
        {#each tabs as item (item.id)}
          {@const selected = item.id === tab}
          <button
            type="button"
            role="tab"
            aria-selected={selected}
            onclick={() => { tab = item.id; }}
            class={cn(
              'h-7 px-3 rounded-full text-xs font-medium transition-colors',
              selected ? 'bg-accent-bg text-accent-text' : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {item.label}
          </button>
        {/each}
      </div>
    </header>

    {#if tab === 'account'}
      <section class="space-y-6">
        <div class="rounded-md bg-surface p-4 space-y-3">
          <div class="flex items-center gap-3">
            {#if user.avatar_url && !avatarFailed}
              <img src={user.avatar_url} alt={user.display_name} class="size-10 rounded-full bg-surface-2" onerror={() => { avatarFailed = true; }} />
            {:else}
              <span class="inline-grid place-items-center size-10 rounded-full bg-accent text-accent-fg font-semibold text-sm">
                {(user.display_name.trim().charAt(0) || '?').toUpperCase()}
              </span>
            {/if}
            <div class="min-w-0 flex-1">
              <div class="text-sm font-medium text-text truncate">{user.display_name}</div>
              {#if user.email}<div class="text-xs text-text-muted truncate">{user.email}</div>{/if}
            </div>
            {#if user.tier}
              <span class="rounded-full bg-surface-2 px-2 py-1 text-xs text-text-subtle">{user.tier.name}</span>
            {/if}
          </div>
          <Button variant="ghost" size="sm" onclick={() => session.signOut().then(() => goto('/login'))}>
            <LogOut size={14} /> Đăng xuất
          </Button>
        </div>

        <div class="rounded-md bg-surface p-4 space-y-3">
          <div class="flex items-start justify-between gap-3">
            <div class="space-y-1">
              <div class="flex items-center gap-2 text-sm font-medium text-text"><Download size={14} class="text-accent" /> Cài Hội Mê Truyện</div>
              <p class="text-xs leading-relaxed text-text-muted">Cài như app riêng để mở nhanh, chạy full-screen và giữ cache shell khi mạng yếu.</p>
              {#if pwaInstall.state.installed}
                <p class="text-xs text-success-text">Ứng dụng đã được cài trên thiết bị này.</p>
              {:else if !pwaInstall.state.ready}
                <p class="text-xs text-text-subtle">Mở menu trình duyệt rồi chọn “Cài đặt ứng dụng” nếu nút chưa sẵn sàng.</p>
              {:else if pwaInstall.state.lastOutcome === 'dismissed'}
                <p class="text-xs text-text-subtle">Bạn có thể cài lại từ menu trình duyệt nếu nút chưa hiện.</p>
              {/if}
            </div>
            {#if !pwaInstall.state.installed && pwaInstall.state.ready}
              <Button variant="primary" size="sm" onclick={() => pwaInstall.install()}>Cài app</Button>
            {:else}
              <Button variant="secondary" size="sm" disabled>
                {pwaInstall.state.installed ? 'Đã cài' : 'Chưa khả dụng'}
              </Button>
            {/if}
          </div>
        </div>
      </section>
    {/if}

    {#if tab === 'sources'}
      <section class="space-y-3">
        <div class="rounded-md bg-surface p-4 space-y-3">
          <div class="flex items-center justify-between gap-3">
            <div class="flex items-center gap-2 text-sm">
              <Globe2 size={14} class="text-accent" />
              <span class="font-medium text-text">Nguồn truyện</span>
            </div>
            <Button variant="secondary" size="sm" onclick={installDefaults}>Cài bộ mặc định</Button>
          </div>
          <p class="text-xs text-text-muted">
            Bạn tự chọn cài hoặc gỡ từng nguồn. Nguồn 18+ phải tự cài thủ công; dùng bộ lọc 18+ để tìm nhanh.
          </p>

          {#if sources.length > 0}
            <div class="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div class="relative flex-1">
                <Search size={14} class="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
                <input
                  type="search"
                  placeholder="Tìm theo tên hoặc host…"
                  bind:value={sourceQuery}
                  class={cn(inputCls, 'pl-8')}
                />
              </div>
              <select class={cn(inputCls, 'sm:w-40')} bind:value={sourceLang}>
                <option value="all">Mọi ngôn ngữ</option>
                {#each sourceLanguages as code (code)}
                  <option value={code}>{languageName(code)}</option>
                {/each}
              </select>
              <select class={cn(inputCls, 'sm:w-32')} bind:value={sourceVisibility}>
                <option value="all">Tất cả</option>
                <option value="sfw">An toàn</option>
                <option value="nsfw">18+</option>
              </select>
            </div>
          {/if}

          {#if sources.length === 0}
            <div class="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
              <p class="text-sm text-text-muted">Chưa cài nguồn nào</p>
              <p class="text-xs text-text-subtle mt-1">Bấm “Cài bộ mặc định” để cài nhanh các nguồn phổ biến.</p>
            </div>
          {:else if filteredSources.length === 0}
            <div class="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
              <p class="text-sm text-text-muted">Không có nguồn khớp bộ lọc</p>
              <p class="text-xs text-text-subtle mt-1">Thử đổi từ khóa, ngôn ngữ hoặc bộ lọc 18+.</p>
            </div>
          {:else}
            <ul class="space-y-2">
              {#each filteredSources as source (source.manifest.id)}
                <li class="flex items-center justify-between gap-3 rounded-md bg-surface-2 px-3 py-2">
                  <div class="min-w-0">
                    <div class="flex items-center gap-1.5">
                      <span class="text-sm font-medium text-text truncate">{source.manifest.name}</span>
                      {#if source.manifest.nsfw}
                        <span class="shrink-0 rounded-full bg-error-bg px-1.5 py-0.5 text-[10px] font-semibold text-error-text">18+</span>
                      {/if}
                    </div>
                    <div class="text-xs text-text-muted truncate">
                      {source.manifest.host} · {languageSummary(source.manifest.languages)} · {source.origin}
                    </div>
                  </div>
                  <button
                    type="button"
                    onclick={() => toggleSource(source)}
                    class={cn(
                      'h-7 rounded-full px-3 text-xs font-medium transition-colors cursor-pointer',
                      source.enabled ? 'bg-surface text-text-muted hover:bg-hover hover:text-text' : 'bg-accent-bg text-accent-text hover:opacity-90',
                    )}
                  >
                    {source.enabled ? 'Gỡ' : 'Cài'}
                  </button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      </section>
    {/if}

    {#if tab === 'read'}
      <section class="space-y-4">
        <Field label="Giao diện">
          <select class={inputCls} value={localSettings.state.theme} onchange={(event) => localSettings.update({ theme: event.currentTarget.value as ThemeMode })}>
            <option value="system">Theo hệ thống</option>
            <option value="dark">Tối</option>
            <option value="light">Sáng</option>
          </select>
        </Field>

        <div class="rounded-md bg-surface p-4 space-y-4">
          <div class="flex items-center gap-2 text-sm">
            <Sparkles size={14} class="text-accent" />
            <span class="font-medium text-text">Dịch trong reader</span>
          </div>

          <Field label="Ngôn ngữ đích">
            <select class={inputCls} value={targetLang} onchange={(event) => updateTargetLang(event.currentTarget.value)}>
              <option value="vi">Tiếng Việt</option>
              <option value="en">Tiếng Anh</option>
            </select>
          </Field>

          <div class="space-y-2">
            <div class="text-xs font-medium text-text-muted">Công cụ dịch</div>
            <div class="grid gap-2 sm:grid-cols-2">
              {#each providerOptions as option (option.id)}
                {@const active = localSettings.state.translation_provider === option.id}
                <button
                  type="button"
                  onclick={() => updateProvider(option.id)}
                  aria-pressed={active}
                  class={cn(
                    'flex flex-col items-start gap-0.5 rounded-md border px-3 py-2 text-left transition-colors cursor-pointer',
                    active ? 'border-accent bg-accent-bg' : 'border-transparent bg-surface-2 hover:bg-hover',
                  )}
                >
                  <span class={cn('text-sm font-medium', active ? 'text-accent-text' : 'text-text')}>{option.label}</span>
                  <span class="text-xs text-text-subtle">{option.hint}</span>
                </button>
              {/each}
              <div
                class="flex flex-col items-start gap-0.5 rounded-md border border-dashed border-border-soft bg-surface-2/50 px-3 py-2 opacity-60"
                aria-disabled="true"
              >
                <span class="text-sm font-medium text-text-muted">AI / LLM</span>
                <span class="text-xs text-text-subtle">Sắp có · custom gateway</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    {/if}

    <footer class="mt-8 border-t border-border-soft pt-4">
      <p class="text-xs text-text-subtle tabular-nums">{versionLabel}</p>
    </footer>
  </div>
{/if}
