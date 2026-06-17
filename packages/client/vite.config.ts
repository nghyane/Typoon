import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

const repoRoot = fileURLToPath(new URL('../..', import.meta.url))

export default defineConfig({
  root: 'dev',
  server: {
    port: 5190,
    proxy: {
      '/deepl': {
        target: 'https://927251094806098001.discordsays.com',
        changeOrigin: true,
        ws: true,
      },
    },
    fs: {
      allow: [repoRoot],
    },
  },
})
