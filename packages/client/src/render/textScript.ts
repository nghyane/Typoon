export type TextScript = 'latin' | 'hangul' | 'kana' | 'han' | 'mixed-cjk' | 'numeric-symbol' | 'mixed'

export function classifyTextScript(text: string): TextScript {
  let latin = 0
  let hangul = 0
  let kana = 0
  let han = 0
  let numericSymbol = 0

  for (const char of text) {
    if (/\s/u.test(char)) continue
    if (isHanChar(char)) han += 1
    else if (isKanaChar(char)) kana += 1
    else if (isHangulChar(char)) hangul += 1
    else if (/[\p{Script=Latin}\p{M}]/u.test(char)) latin += 1
    else numericSymbol += 1
  }

  const cjk = han + kana
  const letters = latin + hangul + cjk
  if (!letters) return 'numeric-symbol'
  if (cjk && (latin || hangul)) return 'mixed-cjk'
  if (han && kana) return 'mixed-cjk'
  if (han) return 'han'
  if (kana) return 'kana'
  if (hangul && latin) return 'mixed'
  if (hangul) return 'hangul'
  if (latin) return numericSymbol > latin ? 'mixed' : 'latin'
  return 'mixed'
}

export function canUseVerticalTypesetting(text: string): boolean {
  const script = classifyTextScript(text)
  return script === 'han' || script === 'kana' || script === 'mixed-cjk'
}

export function canBreakTokenPerCharacter(text: string): boolean {
  const script = classifyTextScript(text)
  return script === 'han' || script === 'kana' || script === 'mixed-cjk'
}

function isHanChar(char: string): boolean {
  const cp = char.codePointAt(0)
  if (cp === undefined) return false
  return (cp >= 0x3400 && cp <= 0x4DBF)
    || (cp >= 0x4E00 && cp <= 0x9FFF)
    || (cp >= 0xF900 && cp <= 0xFAFF)
}

function isKanaChar(char: string): boolean {
  const cp = char.codePointAt(0)
  if (cp === undefined) return false
  return cp >= 0x3040 && cp <= 0x30FF
}

function isHangulChar(char: string): boolean {
  const cp = char.codePointAt(0)
  if (cp === undefined) return false
  return (cp >= 0x1100 && cp <= 0x11FF)
    || (cp >= 0x3130 && cp <= 0x318F)
    || (cp >= 0xAC00 && cp <= 0xD7AF)
}
