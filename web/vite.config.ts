import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { TanStackRouterVite } from '@tanstack/router-plugin/vite'

// Dev proxies `/api` and `/files` to the FastAPI backend so the web app can
// stay same-origin in the browser (no CORS, no absolute URL plumbing for
// <img src="/files/...">). Override the target with VITE_API_URL.
export default defineConfig(({ mode }) => {
  const env    = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [
      TanStackRouterVite({ routesDirectory: './src/routes', generatedRouteTree: './src/routeTree.gen.ts' }),
      react(),
      tailwindcss(),
    ],
    server: {
      proxy: {
        '/api':   { target, changeOrigin: true },
        '/files': { target, changeOrigin: true },
      },
    },
  }
})
