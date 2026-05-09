import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { TanStackRouterVite } from '@tanstack/router-plugin/vite'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('.', import.meta.url))

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
    resolve: {
      alias: {
        '@app':      resolve(root, 'src/app'),
        '@shared':   resolve(root, 'src/shared'),
        '@features': resolve(root, 'src/features'),
      },
    },
    server: {
      proxy: {
        '/api':   { target, changeOrigin: true },
        '/files': { target, changeOrigin: true },
        // No /cdn proxy: archive URLs now point directly at the
        // Discord Activity proxy (`<app>.discordsays.com/cdn/t/...`),
        // which is reachable from any origin (CORS = *) and serves
        // the same response in dev and prod.
      },
    },
  }
})
