// pipeline/readerNoise.ts — watermark/credit noise filtering for reader OCR.
//
// Extracted from the strip-based chapter translation so the page-anchored
// pipeline can reuse it without depending on chapter/strip layout. A NoiseFrame
// is just the page rect (source px) the text lives in.

import type { BBox } from '../domain/geometry'
import type { RecognizedTextPage } from '../domain/text'

export interface NoiseFrame {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

/** Drop watermark/credit/noise blocks, keeping real text. */
export function removeReaderNoiseBlocks(recognized: RecognizedTextPage, frame: NoiseFrame): RecognizedTextPage {
  return { ...recognized, blocks: recognized.blocks.filter(block => !isReaderNoiseText(block.text, block.bbox, frame)) }
}

export function isReaderNoiseText(text: string, bbox: BBox, frame: NoiseFrame): boolean {
  const source = normalizeText(text)
  if (!source) return true
  if (source === '巅峰' || source === '武炼巅峰' || source === '武煉巔峰') return true
  if (isCreditFragment(source) && isSmallEdgeText(source, bbox, frame)) return true
  if (/^[0-9]{1,4}$/u.test(source) && bboxArea(bbox) < 6000) return true
  if (/^[a-z0-9]$/iu.test(source) && bboxArea(bbox) < 2000) return true
  return false
}

function isCreditFragment(source: string): boolean {
  return source === '腾讯'
    || source === '騰訊'
    || source === '腾'
    || source === '訊'
    || source === '讯'
    || /^(?:腾讯|騰訊)(?:动|動|动漫|動漫|漫|漫画|漫畫)?$/u.test(source)
    || source === 'tencent'
    || source.includes('tencentcomics')
    || source.includes('tencentanime')
    || source === '包子'
    || source.includes('包子漫')
    || source.includes('baozimh')
}

function isSmallEdgeText(source: string, bbox: BBox, frame: NoiseFrame): boolean {
  const width = Math.max(1, frame.width)
  const height = Math.max(1, frame.height)
  const boxWidth = bboxWidth(bbox)
  const boxHeight = bboxHeight(bbox)
  const small = boxWidth <= width * 0.30
    && boxHeight <= Math.min(width * 0.09, height * 0.10)
    && bboxArea(bbox) <= width * height * 0.015
  const localX1 = bbox[0] - frame.x
  const localX2 = bbox[2] - frame.x
  const localY1 = bbox[1] - frame.y
  const localY2 = bbox[3] - frame.y
  const nearSideEdge = localX1 <= width * 0.18 || localX2 >= width * 0.82
  const nearVerticalEdge = localY1 <= height * 0.12 || localY2 >= height * 0.88
  if (source === '包子') return small && nearSideEdge && nearVerticalEdge
  return small && (nearSideEdge || nearVerticalEdge)
}

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
}

function bboxArea(bbox: BBox): number {
  return bboxWidth(bbox) * bboxHeight(bbox)
}

function bboxWidth(bbox: BBox): number {
  return Math.max(0, bbox[2] - bbox[0])
}

function bboxHeight(bbox: BBox): number {
  return Math.max(0, bbox[3] - bbox[1])
}
