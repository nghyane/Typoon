<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import { cn } from '$lib/cn';

  type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
  type ButtonSize = 'sm' | 'md' | 'lg';

  let {
    variant = 'secondary',
    size = 'md',
    icon = false,
    class: cls = '',
    type = 'button',
    children,
    ...rest
  }: HTMLButtonAttributes & {
    variant?: ButtonVariant;
    size?: ButtonSize;
    icon?: boolean;
    class?: string;
    children: import('svelte').Snippet;
  } = $props();

  const base = 'inline-flex items-center justify-center gap-1.5 rounded-sm font-medium transition-[background-color,color,filter] duration-150 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap select-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent';
  const sizes: Record<ButtonSize, string> = {
    sm: 'h-7 px-2.5 text-xs',
    md: 'h-8 px-3 text-sm',
    lg: 'h-10 px-4 text-sm',
  };
  const iconSizes: Record<ButtonSize, string> = {
    sm: 'size-7 p-0',
    md: 'size-8 p-0',
    lg: 'size-10 p-0',
  };
  const variants: Record<ButtonVariant, string> = {
    primary: 'bg-accent text-accent-fg hover:brightness-110',
    secondary: 'bg-surface-2 text-text hover:bg-interactive-hover',
    ghost: 'bg-transparent text-text-muted hover:text-text hover:bg-hover',
    danger: 'bg-error text-white hover:brightness-110',
  };
</script>

<button class={cn(base, icon ? iconSizes[size] : sizes[size], variants[variant], cls)} {type} {...rest}>
  {@render children()}
</button>
