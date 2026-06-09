import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

const repoRoot = fileURLToPath(new URL('../..', import.meta.url))

export default defineConfig({
  root: 'dev',
  server: {
    port: 5190,
    fs: {
      allow: [repoRoot],
    },
  },
})
