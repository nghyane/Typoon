import type { PageLoad } from './$types';
import { getWork } from '$lib/works/repo';
import { getSource } from '$lib/source/registry';
import { fetchMangaDetail } from '$lib/source/runtime/endpoints';
import { localSettings } from '$lib/localSettings.svelte';
import type { WorkSource } from '$lib/db';
import { queryClient } from '$lib/queryClient';
import {
  mergeChapters,
  pickBestVersion,
  sortMergedChaptersAsc,
  versionKeyOf,
  type SourceChapterDetail,
} from '$lib/work/chapters';

export const load: PageLoad = async ({ params }) => {
  const work = await getWork(params.workId);

  if (!work || work.sources.length === 0) {
    return {
      workId: params.workId, chapterRef: params.numberNorm, urls: [], targetLang: localSettings.state.default_target_lang,
      workTitle: work?.title ?? null, chapterNumber: null, chapterIndex: null, chapterTotal: null,
      sourceId: null, selectedVersionKey: null, sourceName: null, sourceLang: null,
      pageTokens: null, pageHeaders: null,
      prevRef: null, nextRef: null, chapters: [], versions: [],
    };
  }

  try {
    const sourceChapters = await fetchSourceChapters(work.sources);
    const merged = mergeChapters(sourceChapters, localSettings.state.default_target_lang.toLowerCase());
    const chapter = merged.find((item) => item.numberNorm === params.numberNorm);
    if (!chapter) throw new Error('chapter not found');

    const version = pickBestVersion(chapter, localSettings.state.default_target_lang.toLowerCase());
    if (!version) throw new Error('chapter source not found');

    const sorted = sortMergedChaptersAsc(merged);
    const index = sorted.findIndex((item) => item.numberNorm === chapter.numberNorm);
    const prev = index > 0 ? sorted[index - 1] : null;
    const next = index >= 0 && index < sorted.length - 1 ? sorted[index + 1] : null;
    return {
      workId: params.workId,
      chapterRef: chapter.numberNorm || params.numberNorm,
      urls: [],
      pageTokens: null,
      targetLang: localSettings.state.default_target_lang,
      workTitle: work.title,
      chapterNumber: chapter.number || chapter.numberNorm,
      chapterIndex: index >= 0 ? index + 1 : null,
      chapterTotal: sorted.length,
      sourceId: version.source.manifest.id,
      selectedVersionKey: versionKeyOf(version),
      sourceName: version.source.manifest.name,
      sourceLang: version.lang,
      pageHeaders: null,
      prevRef: prev?.numberNorm ?? null,
      nextRef: next?.numberNorm ?? null,
      chapters: sorted.map((item) => ({
        numberNorm: item.numberNorm,
        number: item.number,
        label: item.label,
      })),
      versions: chapter.sourceVersions.map((item) => ({
        key: versionKeyOf(item),
        sourceId: item.source.manifest.id,
        sourceName: item.source.manifest.name,
        lang: item.lang,
        url: item.ref.url,
        scanlator: item.ref.scanlator,
        date: item.ref.date,
      })),
    };
  } catch {
    return {
      workId: params.workId, chapterRef: params.numberNorm, urls: [], targetLang: localSettings.state.default_target_lang,
      workTitle: work.title, chapterNumber: null, chapterIndex: null, chapterTotal: null,
      sourceId: null, selectedVersionKey: null, sourceName: null, sourceLang: null,
      pageTokens: null, pageHeaders: null,
      prevRef: null, nextRef: null, chapters: [], versions: [],
    };
  }
};

async function fetchSourceChapters(sources: WorkSource[]): Promise<SourceChapterDetail[]> {
  const results = await Promise.allSettled(sources.map(async (origin) => {
    const source = getSource(origin.source);
    if (!source) throw new Error(`Nguồn ${origin.source} không khả dụng.`);
    const detail = await queryClient.ensureQueryData({
      queryKey: ['manga-detail', origin.source, origin.upstream_ref] as const,
      queryFn: () => fetchMangaDetail(source.manifest, origin.upstream_ref),
    });
    return { source, origin, refs: detail.chapters };
  }));

  return results
    .filter((result): result is PromiseFulfilledResult<SourceChapterDetail> => result.status === 'fulfilled')
    .map((result) => result.value);
}
