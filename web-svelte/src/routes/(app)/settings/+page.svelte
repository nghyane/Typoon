<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { Database, Download, Globe2, HardDrive, LogOut, Sparkles } from 'lucide-svelte';
  import { session } from '$lib/auth/session.svelte';
  import { languageSummary } from '$lib/lang';
  import { localSettings, type ThemeMode } from '$lib/localSettings.svelte';
  import { listSources, setSourceEnabled } from '$lib/source/registry';
  import type { InstalledSource } from '$lib/source/types';
  import { cn } from '$lib/cn';
  import Button from '$lib/ui/Button.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import { pwaInstall } from '$lib/pwa/installPrompt.svelte';

  type Tab = 'account' | 'sources' | 'reader' | 'translation' | 'storage';

  let tab = $state<Tab>('account');
  let sources = $state<InstalledSource[]>([]);
  let avatarFailed = $state(false);

  const user = $derived(session.state.status === 'authenticated' ? session.state.user : null);
  const targetLang = $derived(user?.preferred_target_lang ?? localSettings.state.default_target_lang ?? 'vi');
  const enabledSourceCount = $derived(sources.filter((source) => source.enabled).length);

  const tabs = $derived<Array<{ id: Tab; label: string }>>([
    { id: 'account', label: 'Tài khoản' },
    { id: 'sources', label: `Nguồn (${enabledSourceCount}/${sources.length})` },
    { id: 'reader', label: 'Trình đọc' },
    { id: 'translation', label: 'Dịch' },
    { id: 'storage', label: 'Lưu trữ' },
  ]);

  onMount(() => {
    localSettings.load();
    sources = listSources();
  });

  function refreshSources(): void {
    sources = listSources();
  }

  function toggleSource(source: InstalledSource): void {
    setSourceEnabled(source.manifest.id, !source.enabled);
    refreshSources();
  }

  function enableBundled(): void {
    for (const source of sources) setSourceEnabled(source.manifest.id, true);
    refreshSources();
  }

  function updateTargetLang(lang: string): void {
    localSettings.update({ default_target_lang: lang });
  }

  const inputCls = 'h-8 w-full px-3 rounded-sm bg-surface-2 border border-transparent text-sm text-text placeholder:text-text-subtle hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none transition-colors';
</script>

<svelte:head><title>Cài đặt — Hội Mê Truyện</title></svelte:head>

{#if session.state.status === 'loading' || !user}
  <div class="flex items-center justify-center py-16">
    <Spinner size={20} />
  </div>
{:else}
  <div class="max-w-3xl mx-auto px-4 sm:px-6 py-6 space-y-6">
    <header>
      <h1 class="text-lg font-semibold text-text">Cài đặt</h1>
    </header>

    <div class="flex flex-wrap gap-1.5" role="tablist">
      {#each tabs as item (item.id)}
        {@const selected = item.id === tab}
        <button
          type="button"
          role="tab"
          aria-selected={selected}
          onclick={() => { tab = item.id; }}
          class={cn(
            'h-7 px-3 rounded-full text-xs font-medium transition-colors',
            selected ? 'bg-accent-bg text-accent-text' : 'bg-surface text-text-muted hover:bg-hover hover:text-text',
          )}
        >
          {item.label}
        </button>
      {/each}
    </div>

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

        <Field label="Ngôn ngữ đích mặc định">
          <select class={inputCls} value={targetLang} onchange={(event) => updateTargetLang(event.currentTarget.value)}>
            <option value="vi">Tiếng Việt</option>
            <option value="en">Tiếng Anh</option>
          </select>
        </Field>

        <div class="rounded-md bg-surface p-4 space-y-3">
          <div class="flex items-start justify-between gap-3">
            <div class="space-y-1">
              <div class="flex items-center gap-2 text-sm font-medium text-text"><Download size={14} class="text-accent" /> Cài Hội Mê Truyện</div>
              <p class="text-xs leading-relaxed text-text-muted">Cài như app riêng để mở nhanh, chạy full-screen và giữ cache shell khi mạng yếu.</p>
              {#if pwaInstall.state.installed}
                <p class="text-xs text-success-text">Ứng dụng đã được cài trên thiết bị này.</p>
              {:else if pwaInstall.state.lastOutcome === 'dismissed'}
                <p class="text-xs text-text-subtle">Bạn có thể cài lại từ menu trình duyệt nếu nút chưa hiện.</p>
              {/if}
            </div>
            <Button variant="primary" size="sm" disabled={!pwaInstall.state.ready || pwaInstall.state.installed} onclick={() => pwaInstall.install()}>
              {pwaInstall.state.installed ? 'Đã cài' : pwaInstall.state.ready ? 'Cài app' : 'Chờ trình duyệt'}
            </Button>
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
            <Button variant="secondary" size="sm" onclick={enableBundled}>Cài nguồn mặc định</Button>
          </div>
          <p class="text-xs text-text-muted">
            Bật ít nhất một nguồn để tìm và thêm truyện. Nguồn mặc định đi kèm ứng dụng, không cần cài ngoài.
          </p>

          {#if sources.length === 0}
            <div class="rounded-md bg-surface-2 border border-dashed border-border-soft px-4 py-6 text-center">
              <p class="text-sm text-text-muted">Chưa cài nguồn nào</p>
              <p class="text-xs text-text-subtle mt-1">Bấm “Cài nguồn mặc định” để khôi phục danh sách nguồn.</p>
            </div>
          {:else}
            <ul class="space-y-2">
              {#each sources as source (source.manifest.id)}
                <li class="flex items-center justify-between gap-3 rounded-md bg-surface-2 px-3 py-2">
                  <div class="min-w-0">
                    <div class="text-sm font-medium text-text truncate">{source.manifest.name}</div>
                    <div class="text-xs text-text-muted truncate">
                      {source.manifest.host} · {languageSummary(source.manifest.languages)} · {source.origin}
                    </div>
                  </div>
                  <button
                    type="button"
                    onclick={() => toggleSource(source)}
                    class={cn(
                      'h-7 rounded-full px-3 text-xs font-medium transition-colors cursor-pointer',
                      source.enabled ? 'bg-accent-bg text-accent-text' : 'bg-surface text-text-muted hover:bg-hover hover:text-text',
                    )}
                  >
                    {source.enabled ? 'Đang bật' : 'Đã tắt'}
                  </button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      </section>
    {/if}

    {#if tab === 'reader'}
      <section class="space-y-4">
        <div class="rounded-md bg-surface p-4 space-y-2">
          <div class="text-sm font-medium text-text">Kiểu đọc mặc định</div>
          <div class="text-sm text-text-muted">Cuộn dọc tối ưu dịch thuật</div>
          <p class="text-xs leading-relaxed text-text-subtle">
            Chế độ từng page đã được bỏ để overlay dịch, OCR và vị trí chữ luôn dùng cùng một hệ tọa độ page-local.
          </p>
        </div>

        <Field label="Giao diện">
          <select class={inputCls} value={localSettings.state.theme} onchange={(event) => localSettings.update({ theme: event.currentTarget.value as ThemeMode })}>
            <option value="system">Theo hệ thống</option>
            <option value="dark">Tối</option>
            <option value="light">Sáng</option>
          </select>
        </Field>
      </section>
    {/if}

    {#if tab === 'translation'}
      <section class="space-y-3">
        <div class="rounded-md bg-surface p-4 space-y-3">
          <div class="flex items-center gap-2 text-sm">
            <Sparkles size={14} class="text-accent" />
            <span class="font-medium text-text">Dịch trong reader</span>
          </div>
          <div class="grid sm:grid-cols-2 gap-3">
            <Field label="Ngôn ngữ đích">
              <select class={inputCls} value={targetLang} onchange={(event) => updateTargetLang(event.currentTarget.value)}>
                <option value="vi">Tiếng Việt</option>
                <option value="en">Tiếng Anh</option>
              </select>
            </Field>
            <div class="rounded-md bg-surface-2 px-3 py-2">
              <div class="text-xs text-text-subtle">Trạng thái</div>
              <div class="mt-1 inline-flex items-center gap-1.5 text-sm font-medium text-accent-text">
                <Sparkles size={13} /> Realtime
              </div>
            </div>
          </div>
        </div>
      </section>
    {/if}

    {#if tab === 'storage'}
      <section class="space-y-3">
        <div class="rounded-md bg-surface p-4 space-y-2">
          <div class="flex items-center gap-2 text-sm">
            <HardDrive size={14} class="text-text-subtle" />
            <span class="font-medium text-text">Bộ nhớ thiết bị</span>
          </div>
          <div class="text-xs text-text-muted space-y-1">
            <div>Tổng: 0.0 MB · 0 gói offline</div>
            <div>Raw offline: 0</div>
          </div>
        </div>

        <div class="rounded-md bg-surface p-4 space-y-2">
          <div class="flex items-center gap-2 text-sm">
            <Database size={14} class="text-text-subtle" />
            <span class="font-medium text-text">Dữ liệu dịch realtime</span>
          </div>
          <p class="text-xs text-text-muted">Cache bản dịch trong reader.</p>
        </div>
      </section>
    {/if}
  </div>
{/if}
