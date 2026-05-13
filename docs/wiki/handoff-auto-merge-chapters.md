# Handoff — auto-merge multi-source chapter list

## Status khi bàn giao

`/w/$workId` hiện chọn **1 active material** theo URL `?src=`, fetch
**1 manifest** của material đó, merge với translations từ server payload.
Sau khi Works được community-vote merge, 1 Work có thể đính N sibling
materials (HappyMH-zh + OTruyen-vi + MangaDex multilang). User phải click
`SourceChipRail` chuyển qua chuyển lại từng source để xem chapters.

Đó là rò rỉ implementation: "source" là cách app fetch dữ liệu, không
phải khái niệm user. Pattern tương tự MangaDex — chapter list duy nhất,
mỗi chapter row liệt kê các bản (language / scanlator) khả dụng — user
không bao giờ phải chọn source ở level page.

## Việc phải làm

Refactor `useWorkData` + `mergeChapters` để fetch manifest **song song**
từ MỌI source-installed material, union mọi chapters vào 1 list duy
nhất. Bỏ `SourceChipRail`, bỏ `?src=` URL state, bỏ field
`activeMaterial`. VersionLine đã hiển thị source per row → đủ context.

## Decisions cố định (KHÔNG hỏi lại)

| Quyết định                                  | Giá trị                                              |
|---------------------------------------------|------------------------------------------------------|
| `SourceChipRail`                            | XOÁ. Info redundant với VersionLine                  |
| `?src=` URL state                           | XOÁ ở `/w/$workId` + `/r/$workId/$numberNorm`        |
| `activeMaterial` trên WorkData              | XOÁ                                                  |
| Cover priority                              | target_lang match → first material có cover non-null |
| Manifest fetch failure                      | Silent skip (không banner)                           |
| Parallel fetch                              | Tất cả parallel, không concurrency limit             |
| Title resolver chain (đã bỏ activeMaterial) | `title_locale[lang]` → langs match → title_native → materials[0] |
| Reader pick version                         | Giữ logic cũ (translation done > raw target > any raw) |
| Library / recent-reads `onContinue`         | Bỏ `search: { src }`                                 |
| Manifest mất                                | Source plugin chưa cài → silent skip                 |
| MetaStrip / BookmarkButton / StatusPicker   | Pick "primary" material theo cùng resolver (first match) |

## Files dự kiến chạm

```
DELETE
  web/src/features/work/SourceChipRail.tsx

EDIT
  web/src/features/work/useWorkData.ts          — fetch parallel, drop activeMaterial
  web/src/features/work/queries.ts              — không đổi (useMangaDetail giữ nguyên)
  web/src/features/work/title.ts                — bỏ activeMaterial param, thêm resolveWorkCover
  web/src/features/work/WorkHero.tsx            — bỏ SourceChipRail mount, dùng resolvers
  web/src/features/title/mergeChapters.ts       — accept manifestSources[]
  web/src/features/reader/useReader.ts          — bỏ src
  web/src/routes/w.$workId.tsx                  — bỏ SearchParams.src + handleSelectSource
  web/src/routes/r.$workId.$numberNorm.tsx      — bỏ SearchParams.src
  web/src/routes/index.tsx                      — bỏ search.src trong onContinue
```

## Plan tuần tự

### 1. `mergeChapters` refactor

Input cũ:
```ts
mergeChapters({
  work,
  activeMaterialId,
  manifestChapters: MangaChapterRef[],
  activeSource: InstalledSource,
  installedSources,
})
```

Input mới:
```ts
mergeChapters({
  work,
  manifestSources: Array<{
    material: ApiMaterial
    source: InstalledSource
    chapters: MangaChapterRef[]
  }>,
  installedSources,
})
```

Hành vi: cho mỗi `manifestSources[i]`, generate raw HubVersion như cũ
(với `materialId = item.material.id`, `sourceId = item.material.source`).
Key dedup vẫn theo `numberNorm` — N sources có cùng numberNorm → 1
`HubChapter` với N raw versions + N translations từ server payload.
HubVersion key đã có `materialId` → unique tự nhiên.

### 2. `useWorkData` refactor

Bỏ tham số `preferredSrc`. Bỏ field `activeMaterial`.

```ts
export interface WorkData {
  work:             ApiWorkDetail | null
  materials:        ApiMaterial[]
  targetLang:       string | null
  chapters:         HubChapter[]
  workLoading:      boolean
  manifestsLoading: boolean       // any in-flight manifest
  workError:        Error | null
}
```

Fetch logic:

```ts
const installed = useSources((s) => s.sources)
const sourceMaterials = materials.filter(m =>
  m.source && installed[m.source] && m.upstream_ref
)
const manifestQs = useQueries({
  queries: sourceMaterials.map(m => ({
    queryKey: qk.manifest.detail(m.source, m.upstream_ref),
    queryFn:  () => fetchMangaDetail(installed[m.source]!.manifest, m.upstream_ref!),
    staleTime: 5 * 60_000,
    gcTime:    24 * 60 * 60_000,
    retry: 2,
    placeholderData: keepPreviousData,
  })),
})
const manifestSources = sourceMaterials.map((m, i) => {
  const data = manifestQs[i]?.data
  if (!data) return null
  return {
    material: m,
    source:   installed[m.source!]!,
    chapters: data.chapters,
  }
}).filter(Boolean)
```

`manifestsLoading = manifestQs.some(q => q.isPending && q.fetchStatus !== 'idle')`.

`useQueries` từ `@tanstack/react-query`. Một query failed không kéo các
query khác (RQ default). Silent skip = `data === undefined` → loại
khỏi `manifestSources`.

### 3. `title.ts` refactor

Bỏ tham số `activeMaterial`:

```ts
export function resolveWorkTitle(
  materials:  ApiMaterial[],
  targetLang: string | null,
): ResolvedWorkTitle
```

Chain:
1. `materials[i].title_locale[targetLang]` — bất kỳ material có entry.
2. Material in `languages[targetLang]` → `.title`.
3. Material có `title_native` non-null → `.title`.
4. `materials[0].title`.

Thêm function mới:

```ts
export function resolveWorkCover(
  materials:  ApiMaterial[],
  targetLang: string | null,
): { coverUrl: string | null; materialId: number | null }
```

Chain:
1. Material in `languages[targetLang]` có `cover_url` non-null.
2. Material đầu tiên có `cover_url` non-null.
3. Null.

Thêm function helper `pickPrimaryMaterial(materials, targetLang)` —
trả material để mount `<MetaStrip>`, `<StatusPicker.material>`,
`<BookmarkButton>`. Cùng priority như title.

### 4. `WorkHero` refactor

- Bỏ prop `activeMaterial`, thêm prop (nếu cần) hoặc compute inline:
  - `primary = pickPrimaryMaterial(materials, targetLang)`
  - `cover = resolveWorkCover(materials, targetLang)`
  - `title = resolveWorkTitle(materials, targetLang)`
- `<Cover src={cover.coverUrl} ... />`
- `<MetaStrip material={primary} />`
- `<StatusPicker material={primary ? {...} : null} />`
- XOÁ `<SourceChipRail>` mount + "Nguồn:" label.

### 5. Routes refactor

`web/src/routes/w.$workId.tsx`:
- Bỏ `SearchParams.src`.
- Bỏ `validateSearch.src`.
- Bỏ `handleSelectSource` callback + prop.
- `useWorkData(workIdNum)` không tham số 2.
- `handleOpenVersion`: bỏ `search: { src: activeMaterial?.id }`.
- `handleResume`: bỏ.

`web/src/routes/r.$workId.$numberNorm.tsx`:
- Bỏ `SearchParams.src` + `validateSearch.src`.
- Bỏ `src` argument trong call `useReader`.
- Bỏ `src` từ `beforeLoad` redirect search forwarding.

`web/src/routes/index.tsx`:
- `onContinue`: bỏ `search: { src: it.material_id }`.

### 6. `useReader` refactor

```ts
export interface UseReaderInput {
  workId:     number
  numberNorm: string
  lang?:      string
  // KHÔNG còn src
}
```

`resolveNav` bỏ `src` từ `ReaderNavTarget`. URL nav cũng không gửi
`src`.

### 7. Xoá `SourceChipRail.tsx`

`rm web/src/features/work/SourceChipRail.tsx`.

### 8. Grep cleanup

```bash
grep -rn "activeMaterial\|preferredSrc\|\?src=\|search.*src" web/src --include="*.ts" --include="*.tsx"
```

Tất cả phải resolve về:
- 0 reference `activeMaterial`/`preferredSrc`.
- 0 `?src=` URL trong nav handlers.
- `src` chỉ xuất hiện trong manifest plugin context (browse search), không phải `WorkData`.

### 9. Verify

```bash
cd web
./node_modules/.bin/tsc -b
./node_modules/.bin/eslint src/features/work src/features/reader src/features/title \
  src/routes/w.\$workId.tsx src/routes/r.\$workId.\$numberNorm.tsx src/routes/index.tsx
./node_modules/.bin/vite build
```

Pre-existing warnings ở `WorkChapterList`/`useWorkData` (fast-refresh,
exhaustive-deps) — bỏ qua, không touch.

### 10. Smoke test

Mental:

- Mở `/w/123` (work merge có 3 sources). Thấy 1 chap list duy nhất,
  không có chip rail "Nguồn:", VersionLine từng row chỉ source +
  language.
- Reload `/w/123?src=5` (URL cũ user bookmarked). Router không lỗi,
  `?src=5` bị strip do validateSearch không nhận, page render bình
  thường.
- Mở reader từ chapter row → `/r/123/64` không kèm `src`. Reader pick
  version đúng theo lang.

### 11. Commit + deploy

1 commit duy nhất (refactor tightly-coupled, không split sensible).

Template message:

```
refactor(work): auto-merge chapter list from all installed sources

The active-material model forced users to click a source chip rail to
see chapters from one source at a time. After Work merges (community
vote) a single manga can have HappyMH-zh + OTruyen-vi + MangaDex —
clicking through each to compare bản dịch was the only way to see
what's available where, and the empty-source-chip case was a
distraction.

New model: chapter list auto-merges across every installed source
material of the Work. Each chapter row already lists per-version
source via VersionLine; the source-rail above the list was redundant.

Drop:
- `?src=` URL state on /w/$workId + /r/$workId/$numberNorm
- SourceChipRail component + mount
- `activeMaterial` field on WorkData + every consumer

Add:
- useWorkData fetches every installed-source material's manifest in
  parallel (RQ caches per (source, ref)).
- mergeChapters accepts manifestSources[] and unions raws across.
- resolveWorkCover picks target_lang-matching cover (falls back to
  first non-null) — same semantics as resolveWorkTitle.

Failure mode: a manifest fetch that fails contributes no raws but
doesn't break the page. Silent skip; the row still shows whatever
translations + raws other sources provided.
```

Deploy:

```bash
cd /Users/nghiahoang/Dev/MANGA/ComicScan/v2
wrangler pages deploy web/dist \
  --project-name=mangalocal-web \
  --branch=main \
  --commit-dirty=true
```

## Gotchas

### React Query `useQueries` shape

```ts
import { useQueries } from '@tanstack/react-query'

const queries = useQueries({
  queries: items.map(item => ({
    queryKey: [...],
    queryFn:  ...,
    ...
  })),
})
// queries[i] is { data, isPending, isError, error, ... }
```

Mỗi sub-query có lifecycle riêng — failure isolated. Cache hit/miss
độc lập. Cùng `queryKey` ở `useQueries` và `useQuery` (vd component
khác mount `useMangaDetail(same source, same ref)`) → cùng cache
entry.

### Persistence

`qk.manifest.detail(...)` đã trong PERSIST_DOMAINS (`shared/api/persistence.ts`).
IndexedDB cache sẽ rehydrate mỗi manifest individually trên reload —
không có gì cần đổi.

### Render priority + order chapters

Sau khi union N sources, 1 chapter có thể có 3 raw versions (3 sources)
+ 2 translations. Sort order trong row:
- Translation done first.
- Raw target lang.
- Raw lang khác.

`mergeChapters` đã sort versions per chapter qua `score`. Verify logic
vẫn đúng khi N raws có cùng `numberNorm`. Cùng raw lang khác material →
giữ thứ tự append (theo `manifestSources` order).

### Empty work

Work có 0 source material installed (tất cả là extension/upload, hoặc
plugin chưa cài) → `manifestSources = []` → chapters chỉ có translation
data từ server (work_chapters). UI vẫn render OK, chỉ không có raw
links.

### `WorkChapterList` đang nhận `chapters: HubChapter[]` từ `useWorkData`

Không cần thay đổi `WorkChapterList`. Component đã agnostic — nhận
`chapters[]` từ ngoài, không quan tâm chúng đến từ 1 hay N sources.

### `useReader.useWorkData`

`useReader` gọi `useWorkData(workId, src)`. Sau refactor → `useWorkData(workId)`.
Reader tự pick version theo `resolveVersion` đã có; không cần biết
active source.
