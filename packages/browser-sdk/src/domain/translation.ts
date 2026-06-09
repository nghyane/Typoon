import type { ImagePixels } from './image'
import type { TextRole, TextPlacement } from './planning'

export type TranslationKind = 'dialogue' | 'sfx' | 'skip'

export interface TranslationUnit {
  readonly id: string
  readonly placementId: string
  readonly pageIndex: number
  readonly sourceText: string
  readonly kind: TranslationKind
  readonly role: TextRole
}

export interface TranslatedUnit {
  readonly unitId?: string
  readonly placementId: string
  readonly pageIndex: number
  readonly kind: TranslationKind
  readonly role: TextRole
  readonly sourceText: string
  readonly targetText: string
}

export interface TranslatedPage {
  readonly image: ImagePixels
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly detectedLanguage: string | null
  readonly placements: readonly TextPlacement[]
  readonly units: readonly TranslationUnit[]
  readonly translations: readonly TranslatedUnit[]
  readonly timingMs: Record<string, number>
}
