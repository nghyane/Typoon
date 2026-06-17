import type { TranslationContext } from '../domain/context'
import type { TranslatedUnit, TranslationUnit } from '../domain/translation'
import type { Translator } from '../translators/translator'

export async function translateSegment(args: {
  units: readonly TranslationUnit[]
  translator: Translator
  sourceLang?: string | null
  targetLang: string
  context?: TranslationContext
  signal?: AbortSignal
}): Promise<readonly TranslatedUnit[]> {
  return args.translator.translateUnits({
    units: args.units,
    sourceLang: args.sourceLang ?? null,
    targetLang: args.targetLang,
    context: args.context,
    signal: args.signal,
  })
}
