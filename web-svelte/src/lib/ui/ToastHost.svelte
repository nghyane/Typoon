<script lang="ts">
  import { fly } from 'svelte/transition';
  import { quintOut } from 'svelte/easing';
  import { flip } from 'svelte/animate';
  import { CheckCircle2, AlertCircle, Info, Bell, X } from 'lucide-svelte';
  import { goto } from '$app/navigation';
  import { cn } from '$lib/cn';
  import Cover from './Cover.svelte';
  import { toast, type Toast, type ToastVariant } from './toast.svelte';

  // Variant → default icon + accent ring colour for the leading badge. A toast
  // can override the icon via `icon`; the colour still follows the variant.
  const VARIANTS: Record<ToastVariant, { icon: typeof Info; tone: string }> = {
    default: { icon: Bell, tone: 'bg-accent-bg text-accent-text' },
    success: { icon: CheckCircle2, tone: 'bg-success-bg text-success-text' },
    error: { icon: AlertCircle, tone: 'bg-error-bg text-error-text' },
    info: { icon: Info, tone: 'bg-accent-bg text-accent-text' },
  };

  function runAction(t: Toast): void {
    const action = t.action;
    toast.dismiss(t.id);
    if (!action) return;
    if (action.onClick) action.onClick();
    else if (action.href) void goto(action.href);
  }
</script>

<!-- Bottom-right stack. Full-width on phones, a fixed column on larger screens.
     The container ignores pointer events so it never blocks the page; each card
     re-enables them. aria-live announces new toasts without stealing focus. -->
<!-- Bottom padding clears the mobile bottom tab bar (~3.5rem + safe-area, see
     AppShell); on ≥sm there's no bottom nav so it sits just above the edge. -->
<div
  class="pointer-events-none fixed z-[90] flex flex-col gap-2 px-3 sm:px-0
         right-0 bottom-0 w-full sm:w-auto sm:right-4
         pb-[calc(3.5rem+var(--saib)+0.75rem)] sm:pb-[max(1rem,var(--saib))]"
  aria-live="polite"
  aria-atomic="false"
>
  {#each toast.items as t (t.id)}
    {@const variant = VARIANTS[t.variant ?? 'default']}
    {@const Icon = t.icon ?? variant.icon}
    <div
      role="status"
      animate:flip={{ duration: 200 }}
      in:fly={{ x: 24, y: 8, duration: 260, easing: quintOut }}
      out:fly={{ x: 24, duration: 180, easing: quintOut }}
      onmouseenter={() => toast.pause(t.id)}
      onmouseleave={() => toast.resume(t)}
      onfocusin={() => toast.pause(t.id)}
      onfocusout={() => toast.resume(t)}
      class="pointer-events-auto flex w-full items-start gap-3 rounded-lg border border-border-soft
             bg-surface/95 p-3 shadow-lg shadow-black/30 backdrop-blur sm:w-[22rem]"
    >
      {#if t.cover}
        <Cover
          src={t.cover.src}
          headers={t.cover.headers}
          title={t.cover.title}
          fontSize="text-sm"
          class="h-14 w-10 shrink-0 rounded-md ring-1 ring-inset ring-white/10"
        />
      {:else}
        <span class={cn('mt-0.5 inline-flex size-7 shrink-0 items-center justify-center rounded-full', variant.tone)}>
          <Icon size={15} />
        </span>
      {/if}

      <div class="min-w-0 flex-1 leading-snug">
        <p class="text-sm font-medium text-text line-clamp-1">{t.title}</p>
        {#if t.description}
          <p class="mt-0.5 text-xs text-text-subtle line-clamp-2">{t.description}</p>
        {/if}
        {#if t.action}
          <button
            type="button"
            onclick={() => runAction(t)}
            class="mt-2 inline-flex h-7 items-center rounded-sm bg-accent px-3 text-xs font-semibold text-accent-fg transition hover:brightness-110 active:scale-[0.98] cursor-pointer"
          >
            {t.action.label}
          </button>
        {/if}
      </div>

      <button
        type="button"
        aria-label="Ẩn thông báo"
        onclick={() => toast.dismiss(t.id)}
        class="-mr-1 -mt-0.5 inline-flex size-7 shrink-0 items-center justify-center rounded-sm text-text-subtle transition-colors hover:bg-hover hover:text-text cursor-pointer"
      >
        <X size={15} />
      </button>
    </div>
  {/each}
</div>
