// $lib/types.ts — shared domain types.

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
  scanlator: string | null;
  date: string | null;
}

export interface ReaderData {
  workId: string;
  chapterRef: string;
  urls: string[];
  pageTokens: string[] | null;
  pageHeaders: Record<string, string> | null;
  sourceId: string | null;
  selectedVersionKey: string | null;
  targetLang: string;
  workTitle: string | null;
  chapterNumber: string | null;
  chapterIndex: number | null;
  chapterTotal: number | null;
  sourceName: string | null;
  sourceLang: string | null;
  prevRef: string | null;
  nextRef: string | null;
  chapters: ReaderChapterLink[];
  versions: ReaderSourceVersion[];
}

export type PageBlob = Blob | null;
