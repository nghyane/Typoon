// =============================================================================
// Brand — app identity (name, logo, monogram).
//
// Brand is decoupled from the Discord guild on purpose. Guild is a data
// scope (member check, feed scoping, upload destination); brand is the
// deploy's visual identity. Mixing the two was tried (commits 4c7bd61 →
// ea6868b → c9f4870) and rolled back — Discord renaming the guild
// shouldn't rename the app, and an outage shouldn't strip its identity.
//
// Brand is build-time configurable so a future "clone deploy for another
// hội nhóm" only needs new env vars + a new logo file in public/brand/.
// =============================================================================

export interface Brand {
  /** Full display name. Used in sidebar, <title>, OG tags. */
  name:     string
  /** Short label (≤4 chars) for the collapsed sidebar tile / favicons. */
  monogram: string
  /** Public URL of the logo asset. Empty string → monogram fallback. */
  logoUrl:  string
}

export const BRAND: Brand = {
  name:     import.meta.env.VITE_BRAND_NAME     ?? 'Hội Mê Truyện',
  monogram: import.meta.env.VITE_BRAND_MONOGRAM ?? 'HMT',
  logoUrl:  import.meta.env.VITE_BRAND_LOGO_URL ?? '/brand/logo.webp',
}
