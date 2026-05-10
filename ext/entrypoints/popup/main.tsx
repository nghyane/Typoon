import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider, keepPreviousData } from '@tanstack/react-query'
import { Popup } from './Popup'
import './style.css'

// Match web SPA defaults so cache behavior feels identical when the
// engine is reused across surfaces.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      placeholderData: keepPreviousData,
      retry: 1,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <Popup />
    </QueryClientProvider>
  </StrictMode>,
)
