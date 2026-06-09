import type { TranslationContext } from '../domain/context'
import type { TranslatedUnit, TranslationUnit } from '../domain/translation'

export interface Translator {
  readonly name: string
  translateUnits(args: {
    units: readonly TranslationUnit[]
    sourceLang: string | null
    targetLang: string
    context?: TranslationContext
    signal?: AbortSignal
  }): Promise<readonly TranslatedUnit[]>
}
