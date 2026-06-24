<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { Check, ChevronLeft, ChevronRight, Compass, Globe, Home, Library, LogOut, Settings, Shield } from 'lucide-svelte';
  import { cn } from '$lib/cn';
  import { safeReturnTo } from '$lib/auth/api';
  import { session } from '$lib/auth/session.svelte';
  import { BRAND } from '$lib/brand';
  import { localSettings } from '$lib/localSettings.svelte';

  let { children }: { children: import('svelte').Snippet } = $props();
  let collapsed = $state(true);
  let logoFailed = $state(false);
  let menuOpen = $state(false);
  let pane = $state<'root' | 'lang'>('root');
  let menuEl = $state<HTMLDivElement | null>(null);
  let avatarFailed = $state(false);

  const currentPath = $derived($page.url.pathname);
  const currentHref = $derived(`${$page.url.pathname}${$page.url.search}${$page.url.hash}`);
  const isDevRoute = $derived(currentPath.startsWith('/dev/'));
  const user = $derived(session.state.status === 'authenticated' ? session.state.user : null);
  const currentLang = $derived(localSettings.state.default_target_lang ?? 'vi');

  $effect(() => {
    if (!isDevRoute && session.state.status === 'unauthenticated') {
      goto(`/login?redirect=${encodeURIComponent(safeReturnTo(currentHref))}`, { replaceState: true });
    }
  });

  $effect(() => {
    if (!menuOpen) pane = 'root';
  });

  $effect(() => {
    if (!menuOpen) return;
    const onClick = (event: MouseEvent) => {
      if (menuEl && !menuEl.contains(event.target as Node)) menuOpen = false;
    };
    const onKeydown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (pane !== 'root') pane = 'root';
      else menuOpen = false;
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKeydown);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKeydown);
    };
  });

  onMount(() => {
    void session.load();
    localSettings.load();
    const saved = localStorage.getItem('typoon_sidebar');
    collapsed = saved === null ? window.innerWidth < 640 : saved === 'true';
    const onUnauth = () => {
      goto(`/login?redirect=${encodeURIComponent(safeReturnTo(currentHref))}`, { replaceState: true });
    };
    window.addEventListener('typoon:unauthorized', onUnauth);
    return () => window.removeEventListener('typoon:unauthorized', onUnauth);
  });

  function setCollapsed(value: boolean): void {
    collapsed = value;
    localStorage.setItem('typoon_sidebar', String(value));
  }

  const avatarPalette = ['#4F88E6', '#23A55A', '#F0B232', '#F47B67', '#EC4899', '#0EA5E9', '#10B981', '#F47B67'] as const;

  function avatarColor(name: string): string {
    let h = 0;
    for (let i = 0; i < name.length; i += 1) h = (h << 5) - h + name.charCodeAt(i);
    return avatarPalette[Math.abs(h) % avatarPalette.length]!;
  }

  function active(to: string): boolean {
    return to === '/' ? currentPath === '/' : currentPath.startsWith(to);
  }

  const primary = [
    { to: '/', label: 'Trang chủ', icon: Home },
    { to: '/library', label: 'Thư viện', icon: Library },
    { to: '/explore', label: 'Khám phá', icon: Compass },
  ];

  const bottom = [
    { to: '/', label: 'Nhà', icon: Home },
    { to: '/library', label: 'Thư viện', icon: Library },
    { to: '/explore', label: 'Khám phá', icon: Compass },
    { to: '/settings', label: 'Cài đặt', icon: Settings },
  ];

  const langOptions = [
    { code: 'vi', label: 'Tiếng Việt' },
    { code: 'en', label: 'English' },
  ];
</script>

{#if isDevRoute}
  {@render children()}
{:else if session.state.status === 'error'}
  <div class="min-h-dvh flex items-center justify-center bg-bg text-text p-4">
    <div class="w-full max-w-sm bg-surface border border-border-soft rounded-md p-6 text-center space-y-4">
      <div class="text-sm text-error-text">Không thể kiểm tra đăng nhập: {session.state.error.message}</div>
      <button type="button" onclick={() => session.refresh()} class="h-9 px-4 rounded-sm bg-accent text-accent-fg text-sm font-medium cursor-pointer">
        Thử lại
      </button>
    </div>
  </div>
{:else if user}
  <div class="flex h-dvh overflow-hidden bg-bg text-text text-sm pt-[var(--sait)] pl-[var(--sail)] pr-[var(--sair)]">
    <aside
      style={`width: ${collapsed ? 60 : 240}px; transition: width 180ms ease-in-out`}
      class="hidden sm:flex flex-col h-full shrink-0 overflow-hidden bg-surface"
    >
      <div class="flex items-center h-bar shrink-0">
        <div style="width: 60px" class="h-full flex items-center justify-center shrink-0">
          <button
            type="button"
            onclick={() => { if (collapsed) setCollapsed(false); }}
            class={cn('group relative size-7 rounded-sm flex items-center justify-center overflow-hidden bg-accent text-accent-fg text-[10px] font-bold leading-none', collapsed ? 'cursor-pointer' : 'cursor-default')}
            title={collapsed ? 'Mở rộng' : undefined}
          >
            <span class={cn('transition-opacity', collapsed && 'group-hover:opacity-0')}>{BRAND.monogram}</span>
            {#if BRAND.logoUrl && !logoFailed}
              <img
                src={BRAND.logoUrl}
                alt=""
                onerror={() => { logoFailed = true; }}
                class={cn('absolute inset-0 size-full object-cover transition-opacity', collapsed && 'group-hover:opacity-0')}
              />
            {/if}
            {#if collapsed}<ChevronRight size={12} class="absolute opacity-0 group-hover:opacity-100 transition-opacity text-text" />{/if}
          </button>
        </div>
        <span class="flex-1 min-w-0 truncate text-sm font-semibold text-text transition-opacity duration-150" style={`opacity: ${collapsed ? 0 : 1}`}>{BRAND.name}</span>
        <button type="button" onclick={() => setCollapsed(true)} title="Thu gọn" tabindex={collapsed ? -1 : 0} class="size-7 mr-2 rounded-sm flex items-center justify-center text-text-subtle hover:text-text hover:bg-hover cursor-pointer shrink-0 transition-opacity duration-150" style={`opacity: ${collapsed ? 0 : 1}; pointer-events: ${collapsed ? 'none' : 'auto'}`}>
          <ChevronLeft size={14} />
        </button>
      </div>

      <nav class="px-2 py-2 flex flex-col gap-0.5">
        {#each primary as item}
          {@const isActive = active(item.to)}
          <a href={item.to} title={collapsed ? item.label : undefined} class={cn('group relative flex items-center h-8 w-full rounded-sm select-none cursor-pointer transition-colors duration-150', isActive ? 'bg-accent-bg text-accent-text font-medium' : 'text-text-muted hover:bg-hover hover:text-text')}>
            {#if isActive}<span class="absolute left-1 top-1.5 bottom-1.5 w-0.5 rounded-full bg-accent"></span>{/if}
            <span style="width: 44px" class="h-full flex items-center justify-center shrink-0"><item.icon size={16} class={cn(isActive && 'text-accent')} /></span>
            <span class="flex-1 min-w-0 truncate pr-2 text-sm transition-opacity duration-150" style={`opacity: ${collapsed ? 0 : 1}`}>{item.label}</span>
          </a>
        {/each}
      </nav>
      <div class="flex-1"></div>
      <div class="px-2 pb-2 pt-2 flex flex-col gap-0.5">
        {#if user.is_admin}
          <a href="/admin/ops" class="group relative flex items-center h-8 w-full rounded-sm select-none cursor-pointer transition-colors duration-150 text-text-muted hover:bg-hover hover:text-text">
            <span style="width: 44px" class="h-full flex items-center justify-center shrink-0"><Shield size={16} /></span>
            <span class="flex-1 min-w-0 truncate pr-2 text-sm transition-opacity duration-150" style={`opacity: ${collapsed ? 0 : 1}`}>Quản trị</span>
          </a>
        {/if}
        <a href="/settings" class={cn('group relative flex items-center h-8 w-full rounded-sm select-none cursor-pointer transition-colors duration-150', active('/settings') ? 'bg-accent-bg text-accent-text font-medium' : 'text-text-muted hover:bg-hover hover:text-text')}>
          <span style="width: 44px" class="h-full flex items-center justify-center shrink-0"><Settings size={16} /></span>
          <span class="flex-1 min-w-0 truncate pr-2 text-sm transition-opacity duration-150" style={`opacity: ${collapsed ? 0 : 1}`}>Cài đặt</span>
        </a>
      </div>
    </aside>

    <div class="flex flex-col flex-1 min-w-0 overflow-hidden">
      <header class="flex items-center gap-2 px-3 sm:px-5 h-bar bg-bg shrink-0">
        <div class="flex-1 min-w-0"></div>
        <div bind:this={menuEl} class="relative">
          <button type="button" onclick={() => { menuOpen = !menuOpen; }} class="flex items-center gap-2 pl-1 pr-2 h-8 rounded-sm hover:bg-hover transition-colors cursor-pointer">
            {#if user.avatar_url && !avatarFailed}
              <span class="size-7 rounded-full overflow-hidden shrink-0 flex items-center justify-center bg-surface-2">
                <img src={user.avatar_url} alt={user.display_name} class="w-full h-full object-cover" onerror={() => { avatarFailed = true; }} />
              </span>
            {:else}
              <span class="inline-grid place-items-center size-7 rounded-full text-white font-semibold text-xs flex-none" style={`background: ${avatarColor(user.display_name)}`}>{(user.display_name.trim().charAt(0) || '?').toUpperCase()}</span>
            {/if}
            <span class="text-sm text-text max-w-32 truncate hidden md:inline">{user.display_name}</span>
          </button>

          {#if menuOpen}
            <div class="absolute right-0 top-full mt-1.5 w-64 rounded-md bg-surface border border-border-soft overflow-hidden z-50">
              {#if pane === 'root'}
                <div>
                  <header class="px-3.5 py-3 border-b border-border-soft">
                    <p class="text-sm font-medium text-text truncate">{user.display_name}</p>
                    {#if user.is_admin}
                      <span class="inline-flex items-center gap-1 mt-1.5 text-xs font-semibold uppercase tracking-wider text-success-text bg-success-bg rounded px-1.5 py-0.5"><Shield size={9} />Admin</span>
                    {/if}
                  </header>
                  <nav class="py-1">
                    <button type="button" onclick={() => { pane = 'lang'; }} class="w-full flex items-center gap-2.5 px-3.5 py-2 text-sm text-left cursor-pointer transition-colors text-text hover:bg-hover">
                      <span class="text-text-subtle"><Globe size={14} /></span>
                      <span class="flex-1">Đọc bằng</span>
                      <span class="text-xs text-text-subtle truncate max-w-[80px]">{langOptions.find((item) => item.code === currentLang)?.label ?? '—'}</span>
                      <ChevronRight size={12} class="text-text-subtle" />
                    </button>
                    <a href="/settings" onclick={() => { menuOpen = false; }} class="w-full flex items-center gap-2.5 px-3.5 py-2 text-sm text-left cursor-pointer transition-colors text-text hover:bg-hover">
                      <span class="text-text-subtle"><Settings size={14} /></span>
                      <span class="flex-1">Cài đặt</span>
                      <ChevronRight size={12} class="text-text-subtle" />
                    </a>
                  </nav>
                  <div class="border-t border-border-soft py-1">
                    <button type="button" onclick={() => { menuOpen = false; session.signOut().then(() => goto('/login')); }} class="w-full flex items-center gap-2.5 px-3.5 py-2 text-sm text-left cursor-pointer transition-colors text-text hover:bg-error/10 hover:text-error-text">
                      <span class="text-text-subtle"><LogOut size={14} /></span>
                      <span class="flex-1">Đăng xuất</span>
                    </button>
                  </div>
                </div>
              {:else}
                <div>
                  <header class="flex items-center gap-1 px-2 py-2 border-b border-border-soft">
                    <button type="button" onclick={() => { pane = 'root'; }} class="inline-flex items-center gap-1 h-7 px-2 rounded-sm text-sm text-text-muted hover:bg-hover hover:text-text cursor-pointer transition-colors">
                      <ChevronLeft size={14} />Quay lại
                    </button>
                  </header>
                  <ul role="listbox" class="py-1">
                    {#each langOptions as option (option.code)}
                      {@const selected = currentLang === option.code}
                      <li>
                        <button type="button" role="option" aria-selected={selected} onclick={() => localSettings.update({ default_target_lang: option.code })} class={cn('w-full flex items-center gap-2 px-3.5 py-2 text-sm text-left cursor-pointer transition-colors', selected ? 'text-text' : 'text-text-muted hover:bg-hover hover:text-text')}>
                          <span class="flex-1">{option.label}</span>
                          {#if selected}<Check size={14} class="text-accent" />{/if}
                        </button>
                      </li>
                    {/each}
                  </ul>
                </div>
              {/if}
            </div>
          {/if}
        </div>
      </header>
      <main class="flex-1 overflow-auto sm:pb-[var(--saib)]">{@render children()}</main>
      <nav class="sm:hidden flex items-stretch bg-surface border-t border-border-soft shrink-0 h-[calc(3.5rem+var(--saib))] pb-[var(--saib)]">
        {#each bottom as item}
          {@const isActive = active(item.to)}
          <a href={item.to} class={cn('flex-1 flex flex-col items-center justify-center gap-0.5 text-xs transition-colors', isActive ? 'text-accent-text' : 'text-text-subtle hover:text-text')}>
            <span class={cn('h-6 min-w-9 px-2 rounded-full inline-flex items-center justify-center', isActive && 'bg-accent-bg')}><item.icon size={17} /></span>
            <span>{item.label}</span>
          </a>
        {/each}
      </nav>
    </div>
  </div>
{/if}
