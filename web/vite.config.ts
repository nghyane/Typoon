import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { TanStackRouterVite } from '@tanstack/router-plugin/vite'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('.', import.meta.url))

// Dev proxies `/api` and `/files` to a backend so the web app can
// stay same-origin in the browser. Default target = local FastAPI
// (`http://localhost:8000`); override with `VITE_PUBLIC_BASE_URL`
// (e.g. point at the production DA host for read-only QA against
// real data).
//
// We do NOT proxy `/cdn` — browse-mode hits bunle-cdn directly via
// `https://927251094806098001.discordsays.com/cdn/c/...` because
// `Access-Control-Allow-Origin: *` lets the browser accept it
// cross-origin from any localhost dev URL.
export default defineConfig(({ mode }) => {
  const env    = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_PUBLIC_BASE_URL || 'http://localhost:8787'

  return {
    plugins: [
      TanStackRouterVite({
        routesDirectory:    './src/routes',
        generatedRouteTree: './src/routeTree.gen.ts',
      }),
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
        '/cdn':   { target, changeOrigin: true },
        '/files': { target, changeOrigin: true },
      },
    },
  }
})
