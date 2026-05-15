/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PUBLIC_BASE_URL?: string
  readonly VITE_BRAND_NAME?:      string
  readonly VITE_BRAND_MONOGRAM?:  string
  readonly VITE_BRAND_LOGO_URL?:  string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
