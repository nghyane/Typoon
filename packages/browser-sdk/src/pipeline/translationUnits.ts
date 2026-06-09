import type { TextRole } from '../domain/planning'
import type { TextUnit } from '../domain/text'
import type { TranslationKind, TranslationUnit } from '../domain/translation'

export function translationUnitsFromTextUnits(units: readonly TextUnit[]): TranslationUnit[] {
  return units.map(unit => {
    const role = unit.roleHint ?? 'dialogue'
    return {
      id: unit.id,
      pageIndex: unit.pageIndex,
      blockIds: unit.blockIds,
      sourceText: unit.sourceText,
      kind: translationKind(unit.sourceText, role),
      role,
    }
  })
}

function translationKind(sourceText: string, role: TextRole): TranslationKind {
  if (!sourceText.trim()) return 'skip'
  return role === 'sfx' ? 'sfx' : 'dialogue'
}
