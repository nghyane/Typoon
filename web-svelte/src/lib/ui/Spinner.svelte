<script lang="ts">
  import { cn } from '$lib/cn';
  let { size = 24, class: cls = '' }: { size?: number; class?: string } = $props();
</script>

<span
  class={cn('inline-block ts-spinner-circle', cls)}
  style={`width: ${size}px; height: ${size}px`}
  aria-label="Đang tải"
  role="status"
></span>
