/**
 * Prompt assembly for the brief vision pass.
 *
 * Port of `typoon/stages/brief._build_brief_prompt` plus `prompt.py`
 * `STORYBOARD_SYSTEM`. The system prompt is the storyboard template
 * + the target-language agent policy (the "Context agent policy"
 * section of `targets/{lang}.md`). The user prompt is the bubble list
 * with deterministic script-foreign annotations.
 *
 * Why we ship script detection in the prompt: a CJK substring in a
 * Spanish chapter is almost always non-diegetic chrome (publisher
 * watermark, scanlation tag). Telling the model "foreign=1" up front
 * gives it permission to mark as NOISE without second-guessing.
 */

// @ts-ignore — esbuild text loader
import STORYBOARD_MD from "./prompts/agents/storyboard.md";

// @ts-ignore
import TGT_VI from "./prompts/targets/vi.md";
// @ts-ignore
import TGT_EN from "./prompts/targets/en.md";

const TARGETS: Record<string, string> = { vi: TGT_VI, en: TGT_EN };

const LANG_NAMES: Record<string, string> = {
  ja: "Japanese", ko: "Korean", zh: "Chinese", en: "English",
  es: "Spanish",  vi: "Vietnamese",
};

export function langName(code: string): string {
  return LANG_NAMES[code] ?? code;
}

/** Same logic as `prompt.load_target_agent_policy`. */
export function loadTargetAgentPolicy(code: string): string {
  const full = (TARGETS[code] ?? "").trim();
  if (!full) return "";
  const agent = "### Context agent policy";
  const translator = "### Translator mechanics";
  const start = full.indexOf(agent);
  const end   = full.indexOf(translator);
  if (start === -1) return "";
  return (end === -1 ? full.slice(start) : full.slice(start, end)).trim();
}

export function storyboardSystem(args: {
  sourceLang: string;
  targetLang: string;
  isColor:    boolean;
}): string {
  return (STORYBOARD_MD as string)
    .replace(/\{source_lang_name\}/g,    langName(args.sourceLang))
    .replace(/\{target_lang_name\}/g,    langName(args.targetLang))
    .replace(/\{is_color\}/g,            args.isColor ? "true" : "false")
    .replace(/\{target_agent_policy\}/g, loadTargetAgentPolicy(args.targetLang));
}

// ─── Script detection ─────────────────────────────────────────────────

const SCRIPT_RANGES: { tag: string; re: RegExp }[] = [
  { tag: "han",      re: /[\u3400-\u9fff\uf900-\ufaff]/u },
  { tag: "kana",     re: /[\u3040-\u30ff]/u },
  { tag: "hangul",   re: /[\uac00-\ud7af\u1100-\u11ff]/u },
  { tag: "cyrillic", re: /[\u0400-\u04ff]/u },
  { tag: "arabic",   re: /[\u0600-\u06ff]/u },
];

// Languages → set of scripts that count as *native* (non-foreign).
// Mirrors `typoon/stages/brief._LANG_NATIVE_SCRIPTS`. Latin counts as
// native everywhere via the early short-circuit below, so it doesn't
// appear here.
//
// Japanese mixes kana + kanji inseparably; Korean uses hangul + legacy
// hanja. A bubble that is all-kanji inside a Japanese chapter (`田中`,
// `東京`) is fully native — flagging it as foreign would inflate the
// vision agent's noise candidates with real dialogue.
const LANG_NATIVE_SCRIPTS: Record<string, ReadonlySet<string>> = {
  zh: new Set(["han"]),
  ja: new Set(["kana", "han"]),
  ko: new Set(["hangul", "han"]),
  ru: new Set(["cyrillic"]),
  ar: new Set(["arabic"]),
};

function detectScripts(text: string): string[] {
  const out: string[] = [];
  for (const { tag, re } of SCRIPT_RANGES) if (re.test(text)) out.push(tag);
  if (out.length === 0) out.push("latin");
  return out;
}

function isForeignScript(scripts: string[], sourceLang: string): boolean {
  if (scripts.includes("latin")) return false;
  const native = LANG_NATIVE_SCRIPTS[sourceLang] ?? new Set(["latin"]);
  return scripts.every(s => !native.has(s));
}

// ─── User prompt ──────────────────────────────────────────────────────

export interface PromptBubble {
  key:        string;
  page_index: number;
  text:       string;
}

export function buildBriefPrompt(args: {
  sourceLang: string;
  targetLang: string;
  bubbles:    PromptBubble[];
}): string {
  const header = [
    `Source language: ${args.sourceLang}`,
    `Target language: ${args.targetLang}`,
    "",
    "Bubble list (one line per bubble). `script` lists writing systems",
    "detected in the OCR text; `foreign=1` means non-native script",
    "(strong signal for watermark/credit — but only mark NOISE if certain).",
    "",
  ];
  const rows = args.bubbles.map(b => formatBubbleRow(b, args.sourceLang));
  return [...header, ...rows].join("\n");
}

function formatBubbleRow(b: PromptBubble, srcLang: string): string {
  const raw     = b.text || "";
  const preview = raw.replace(/\n/g, " ").slice(0, 80) || "(empty)";
  const scripts = detectScripts(raw);
  const foreign = isForeignScript(scripts, srcLang) ? 1 : 0;
  return `@@ ${b.key} page=${b.page_index} script=${scripts.join("+")} foreign=${foreign} text=${JSON.stringify(preview)}`;
}
// @ts-ignore
import SPEAKERS_MD from "./prompts/agents/speakers.md";

export function speakersSystem(args: {
  sourceLang:     string;
  targetLang:     string;
  chapterContext: string;
}): string {
  return (SPEAKERS_MD as string)
    .replace(/\{chapter_context\}/g, args.chapterContext);
}

export function buildSpeakersPrompt(args: {
  sourceLang: string;
  bubbles:    PromptBubble[];
}): string {
  const header = [
    `Source language: ${args.sourceLang}`,
    "",
    "Bubble list for this storyboard chunk:",
    "",
  ];
  const rows = args.bubbles.map(b => formatBubbleRow(b, args.sourceLang));
  return [...header, ...rows].join("\n");
}
