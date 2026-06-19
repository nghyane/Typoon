import { QueryClient } from '@tanstack/svelte-query';

export const queryClient = new QueryClient({
	defaultOptions: {
		queries: { staleTime: 5 * 60_000, gcTime: 30 * 60_000, retry: false },
	},
});
