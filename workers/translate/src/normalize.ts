/**
 * Post-LLM text normalization — port of typoon/stages/translate._normalize_for_render.
 *
 *  1. NFC normalise (LLM providers sometimes emit NFD).
 *  2. Collapse `....` → `…`; `.{7,}` → `……`; `…{3,}` → `……`.
 *  3. Strip per-line whitespace; drop blank lines.
 *  4. Punctuation-only last line is joined onto the previous line.
 */

const PUNCT_ONLY = /^[.!?…,]+$/;

export function normalizeForRender(text: string): string {
  if (!text) return text;
  let s = text.normalize("NFC");
  s = s.replace(/\.{7,}/g, "……");
  s = s.replace(/\.{4,6}/g, "…");
  s = s.replace(/…{3,}/g, "……");
  s = s.split("\n").map(l => l.trim()).join("\n");
  const lines = s.split("\n").filter(l => l !== "");
  // Fold lone punctuation lines back into the previous one.
  for (let i = lines.length - 1; i > 0; i--) {
    if (PUNCT_ONLY.test(lines[i])) {
      lines[i - 1] = lines[i - 1] + lines[i];
      lines.splice(i, 1);
    } else {
      break;
    }
  }
  return lines.join("\n");
}
