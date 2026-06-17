import type { TextBlock, TextUnit } from '../domain/text'

export function textUnitsFromBlocks(blocks: readonly TextBlock[], pageIndex: number): TextUnit[] {
  return blocks.map((block, index) => ({
    id: blockUnitId(pageIndex, index),
    pageIndex,
    blockIds: [blockUnitId(pageIndex, index)],
    sourceText: block.text,
    roleHint: classifyBlockRole(block),
  }))
}

export function blockUnitId(pageIndex: number, blockIndex: number): string {
  return `p${pageIndex}-b${blockIndex}`
}

function classifyBlockRole(block: TextBlock): TextUnit['roleHint'] {
  const chars = [...block.text].filter(ch => !/\s/u.test(ch)).length
  const w = Math.max(1, block.bbox[2] - block.bbox[0])
  const h = Math.max(1, block.bbox[3] - block.bbox[1])
  if (Math.abs(block.rotationDeg) > 5) return 'sfx'
  if (chars <= 10 && w / h >= 1.4) return 'sfx'
  if (chars > 30) return 'narration'
  return 'dialogue'
}
