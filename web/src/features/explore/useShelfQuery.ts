import { useInfiniteQuery } from '@tanstack/react-query'
import { fetchBrowse, shelfPageSize } from '@features/browse/manifest/runtime'
import type { InstalledSource, MangaSummary } from '@features/browse/manifest/types'

export interface ShelfInfiniteData {
  items:              MangaSummary[]
  loading:            boolean
  isFetchingNextPage: boolean
  hasNextPage:        boolean
  fetchNextPage:      () => void
  error:              Error | null
}

export function useShelfQuery(
  source:        InstalledSource,
  shelfId:       string,
  filterParams = '',
  filterState:   Record<string, string | string[]> = {},
): ShelfInfiniteData {
  const pageSize = shelfPageSize(source.manifest, shelfId)
  const paginated = pageSize !== Infinity

  const {
    data,
    isPending,
    error,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useInfiniteQuery({
    queryKey:       ['explore', 'shelf', source.manifest.id, shelfId, filterParams],
    queryFn:        ({ pageParam }) =>
      fetchBrowse(source.manifest, shelfId, { page: pageParam as number, filterParams, filterState }),
    initialPageParam: 1,
    getNextPageParam: (lastPage, _allPages, lastPageParam) => {
      // Source không hỗ trợ phân trang hoặc page cuối trả về ít hơn pageSize
      // → không có trang tiếp.
      if (!paginated) return undefined
      if (lastPage.length < pageSize) return undefined
      return (lastPageParam as number) + 1
    },
    staleTime: 5 * 60_000,
    gcTime:    30 * 60_000,  // giữ cache 30 phút sau khi unmount
    retry:     false,
    enabled:   !!shelfId,
  })

  const items = data?.pages.flat() ?? []

  return {
    items,
    loading:            isPending && !data,
    isFetchingNextPage,
    hasNextPage:        hasNextPage ?? false,
    fetchNextPage,
    error:              error as Error | null,
  }
}
