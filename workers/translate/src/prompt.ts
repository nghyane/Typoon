/**
 * Prompt assembly — port of typoon/stages/prompt.py + agents/page.md.
 *
 * Bundles the prompt templates at build time via esbuild's text loader.
 * The Python side uses lru_cache + filesystem reads; the worker has the
 * strings inlined as constants.
 */

// @ts-ignore — text loader
import PAGE_TEMPLATE from "./prompts/agents/page.md";

// Source policies
// @ts-ignore
import SRC_JA from "./prompts/sources/ja.md";
// @ts-ignore
import SRC_KO from "./prompts/sources/ko.md";
// @ts-ignore
import SRC_ZH from "./prompts/sources/zh.md";
// @ts-ignore
import SRC_EN from "./prompts/sources/en.md";
// @ts-ignore
import SRC_ES from "./prompts/sources/es.md";
// @ts-ignore
import SRC_VI from "./prompts/sources/vi.md";

// Target policies
// @ts-ignore
import TGT_VI from "./prompts/targets/vi.md";
// @ts-ignore
import TGT_EN from "./prompts/targets/en.md";

const SOURCES: Record<string, string> = {
  ja: SRC_JA, ko: SRC_KO, zh: SRC_ZH, en: SRC_EN, es: SRC_ES, vi: SRC_VI,
};
const TARGETS: Record<string, string> = { vi: TGT_VI, en: TGT_EN };

const LANG_NAMES: Record<string, string> = {
  ja: "Japanese", ko: "Korean", zh: "Chinese", en: "English",
  es: "Spanish",  vi: "Vietnamese",
};

export function langName(code: string): string {
  return LANG_NAMES[code] ?? code;
}

export function loadSourcePolicy(code: string): string {
  return (SOURCES[code] ?? "").trim();
}

/**
 * Mirror of load_target_policy: return only the "Translator mechanics"
 * half of the target file. Legacy files (no marker) return whole.
 */
export function loadTargetPolicy(code: string): string {
  const full = (TARGETS[code] ?? "").trim();
  const marker = "### Translator mechanics";
  const idx = full.indexOf(marker);
  if (idx === -1) return full;
  return full.slice(idx).trim();
}

export interface PageSystemArgs {
  sourceLang: string;
  targetLang: string;
}

export function pageSystem({ sourceLang, targetLang }: PageSystemArgs): string {
  return PAGE_TEMPLATE
    .replace(/\{source_lang_name\}/g, langName(sourceLang))
    .replace(/\{target_lang_name\}/g, langName(targetLang))
    .replace(/\{source_policy\}/g,    loadSourcePolicy(sourceLang))
    .replace(/\{target_policy\}/g,    loadTargetPolicy(targetLang));
}
