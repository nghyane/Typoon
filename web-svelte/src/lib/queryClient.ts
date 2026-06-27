import { QueryClient } from '@tanstack/svelte-query';

export const queryClient = new QueryClient({
	defaultOptions: {
		// refetchOnWindowFocus off: browse/library listings change slowly and a
		// background refetch on every tab-return just dims the grid (keepPreviousData)
		// for no real freshness gain. staleTime already governs deliberate refetches.
		queries: { staleTime: 5 * 60_000, gcTime: 30 * 60_000, retry: false, refetchOnWindowFocus: false },
	},
});
