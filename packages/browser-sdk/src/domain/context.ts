import type { TranslationUnit } from './translation'

export interface GlossaryEntry {
  readonly source: string
  readonly target: string
  readonly locked?: boolean
  readonly note?: string
}

export interface CharacterProfile {
  readonly id: string
  readonly name: string
  readonly targetName?: string
  readonly voice?: string
  readonly note?: string
}

export interface AddressRule {
  readonly speakerId: string
  readonly listenerId: string
  readonly self: string
  readonly other: string
  readonly confidence?: number
  readonly evidenceUnitIds?: readonly string[]
  readonly note?: string
}

export interface TranslationContext {
  readonly summary?: string
  readonly styleNotes?: readonly string[]
  readonly glossary?: readonly GlossaryEntry[]
  readonly characters?: readonly CharacterProfile[]
  readonly addressRules?: readonly AddressRule[]
}

export interface ContextRequest {
  readonly workId: string
  readonly segmentId: string
  readonly sourceLang: string | null
  readonly targetLang: string
  readonly units: readonly TranslationUnit[]
  readonly signal?: AbortSignal
}
