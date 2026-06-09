import type { ContextRequest, TranslationContext } from '../domain/context'

export type { AddressRule, CharacterProfile, ContextRequest, GlossaryEntry, TranslationContext } from '../domain/context'

export interface ContextBuilder {
  readonly name: string
  buildContext(args: ContextRequest): Promise<TranslationContext>
}
