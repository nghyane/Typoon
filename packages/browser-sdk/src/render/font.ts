export interface FontMetrics {
  readonly unitsPerEm: number
  readonly ascent: number
  readonly descent: number
  readonly lineGap: number
}

export interface FontProfile {
  readonly family: string
  readonly cssFamily: string
  readonly metrics: FontMetrics
  readonly lineHeightRatio: number
}

export const MANGA_FONT_PROFILE: FontProfile = {
  family: 'SamaritanTall',
  cssFamily: 'SamaritanTall, serif',
  metrics: {
    unitsPerEm: 1000,
    ascent: 1318,
    descent: -227,
    lineGap: 0,
  },
  lineHeightRatio: lineHeightRatio({ unitsPerEm: 1000, ascent: 1318, descent: -227, lineGap: 0 }),
}

export const MANGA_FONT_FAMILY = MANGA_FONT_PROFILE.cssFamily

export async function ensureMangaFontLoaded(sizePx = 24): Promise<void> {
  if (!('fonts' in document)) return
  await document.fonts.load(`${sizePx}px ${MANGA_FONT_PROFILE.family}`)
}

function lineHeightRatio(metrics: FontMetrics): number {
  return (metrics.ascent - metrics.descent + metrics.lineGap) / metrics.unitsPerEm
}
