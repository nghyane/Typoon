// $lib/types.ts — shared domain types matching the Go API responses.

export interface WorkCover {
  id: string;
  title: string;
  author?: string;
  cover_url?: string;
  status?: string;
  latest_chapter?: string;
  source_lang?: string;
  target_lang?: string;
}

export interface ChapterItem {
  number: string;
  title?: string;
  pages?: number;
}

export interface PublicSettings {
  sourceFetch: { origins: string[] };
  features: { browse: boolean; translation: boolean };
}

export interface WorkDetail extends WorkCover {
  description?: string;
  tags?: string[];
}

export type TranslationPhase =
  | 'idle' | 'loading-model' | 'translating' | 'done' | 'error';

export interface ReaderChapterLink {
  numberNorm: string;
  number: string;
  label: string;
}

export interface ReaderSourceVersion {
  key: string;
  sourceId: string;
  sourceName: string;
  lang: string;
  url: string;
  scanlator?: string | null;
  date?: string | null;
}

export interface ReaderData {
  workId: string;
  chapterRef: string;
  urls: string[];
  pageTokens?: string[] | null;
  sourceId?: string;
  selectedVersionKey?: string | null;
  targetLang: string;
  workTitle?: string;
  chapterNumber?: string;
  chapterIndex?: number;
  chapterTotal?: number;
  sourceName?: string;
  sourceLang?: string;
  prevRef?: string | null;
  nextRef?: string | null;
  chapters?: ReaderChapterLink[];
  versions?: ReaderSourceVersion[];
}

export type PageBlob = Blob | null;
