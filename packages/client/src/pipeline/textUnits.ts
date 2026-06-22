import type { TextBlock, TextUnit } from '../domain/text'
import { classifyTextBlockRole, textRoleContext, type TextRoleContext } from './textRole'

export function textUnitsFromBlocks(blocks: readonly TextBlock[], pageIndex: number, roleContext?: TextRoleContext): TextUnit[] {
  const context = roleContext ?? textRoleContext(blocks)
  return blocks.map((block, index) => ({
    id: blockUnitId(pageIndex, index),
    pageIndex,
    blockIds: [blockUnitId(pageIndex, index)],
    sourceText: block.text,
    roleHint: classifyTextBlockRole(block, context),
  }))
}

export function blockUnitId(pageIndex: number, blockIndex: number): string {
  return `p${pageIndex}-b${blockIndex}`
}
