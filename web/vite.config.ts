import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { TanStackRouterVite } from '@tanstack/router-plugin/vite'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('.', import.meta.url))

// Dev proxies `/api`, `/files`, and `/cdn` so the web app can stay
// same-origin in the browser. `/api` and `/files` default to the local
// backend (`http://localhost:3000`); override with `VITE_PUBLIC_BASE_URL`.
// `/cdn` only forwards local dev traffic to the stable CDN gateway. The
// gateway owns runtime pool selection/configuration.
const cdnTarget = 'https://bunle-cdn-ceu.pages.dev'

export default defineConfig(({ mode }) => {
  const env    = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_PUBLIC_BASE_URL || 'http://localhost:3000'

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
        '/cdn':   {
          target: cdnTarget,
          changeOrigin: true,
        },
        '/deepl': {
          target: cdnTarget,
          changeOrigin: true,
          ws: true,
        },
        '/files': { target, changeOrigin: true },
      },
    },
  }
})
