import { defineConfig } from 'wxt'
import tailwindcss from '@tailwindcss/vite'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('.', import.meta.url))

// API URL is baked at build time. There is exactly one Hội Mê Truyện
// engine and we don't want users to type a URL — the Setup screen only
// asks for the API token. To run against a different host (dev,
// staging), override VITE_API_URL via .env.local.
//
// NOTE: do NOT use the Discord Activity proxy URL
// (`*.discordsays.com/api`) here. That host only routes requests when
// they originate inside the Discord client iframe with Discord auth
// headers — extension fetches against it return 404. Always point at
// the real public engine origin (api.mangalocal.com).
//
// The host the extension will fetch from MUST be reflected in
// `host_permissions` so users grant it implicitly at install time
// (`optional_host_permissions` requires a runtime prompt we no longer
// have a UI for).
const API_URL = process.env.VITE_API_URL ?? 'https://api.mangalocal.com'
const apiOrigin = new URL(API_URL).origin

export default defineConfig({
  modules: ['@wxt-dev/module-react'],

  manifest: {
    name: 'Hội Mê Truyện — Importer',
    description: 'Chọn ảnh trên trang bất kỳ và upload thành chương cho Hội Mê Truyện.',
    permissions: [
      'storage',
      'activeTab',
      'scripting',
      'notifications',
    ],
    // The engine origin is baked in — grant it at install time so the
    // popup never has to call chrome.permissions.request().
    //
    // `<all_urls>` is required so the popup can fetch images directly
    // from any CDN. Without it, content-script fetches hit the manga
    // CDN's CORS gate (which usually has no Access-Control-Allow-
    // Origin) and fail with "Failed to fetch". Extension-context
    // fetches with host_permissions bypass CORS entirely — the same
    // pattern AdBlock / ImageAssistant / DownThemAll use.
    host_permissions: [`${apiOrigin}/*`, '<all_urls>'],
    icons: {
      16:  'icon/16.png',
      32:  'icon/32.png',
      48:  'icon/48.png',
      96:  'icon/96.png',
      128: 'icon/128.png',
    },
    action: {
      default_title: 'Hội Mê Truyện',
      default_icon: {
        16:  'icon/16.png',
        32:  'icon/32.png',
        48:  'icon/48.png',
      },
    },
  },

  vite: () => ({
    plugins: [tailwindcss()],
    define: {
      __API_URL__: JSON.stringify(API_URL),
    },
    resolve: {
      alias: {
        '@core':   resolve(root, 'core'),
        '@shell':  resolve(root, 'shell'),
        '@shared': resolve(root, 'shared'),
      },
    },
  }),
})
