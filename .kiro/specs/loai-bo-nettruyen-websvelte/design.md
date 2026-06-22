# Design Document

## Overview

Remove the nettruyen manga source from the project by deleting its manifest file, removing its entry from the source registry, and cleaning up stale code references. This is a pure deletion/cleanup task with no new code or architecture changes.

## Architecture

No architectural changes. The existing source-loading pipeline (`packages/manga-sources/index.json` → individual JSON manifests → consumed by web apps via `import.meta.glob`) remains intact. We simply remove one source from the registry.

## Components Affected

### 1. `packages/manga-sources/nettruyen.json`

**Action:** Delete the file entirely.

### 2. `packages/manga-sources/index.json`

**Action:** Remove the object `{ "id": "nettruyen", "url": "./nettruyen.json" }` from the `sources` array. Keep all other entries and ensure the trailing comma/JSON structure remains valid.

Before:
```json
{ "id": "naver-webtoon", "url": "./naver-webtoon.json" },
{ "id": "nettruyen",     "url": "./nettruyen.json" },
{ "id": "comix",         "url": "./comix.json" }
```

After:
```json
{ "id": "naver-webtoon", "url": "./naver-webtoon.json" },
{ "id": "comix",         "url": "./comix.json" }
```

### 3. `web/src/features/reader/components/SourcePicker.tsx`

**Action:** Remove the ASCII-art comment line that mentions "NetTruyen". The comment block at the top of the file includes an example row `VI · NetTruyen`. Remove that line.

Before:
```typescript
//   │   VI · TruyenQQ                       2h trước │
//   │   VI · NetTruyen                      1d trước │
```

After:
```typescript
//   │   VI · TruyenQQ                       2h trước │
```

## Data Models

No data model changes. The `index.json` schema (`{ version: number, sources: Array<{ id: string, url: string }> }`) is unchanged.

## Error Handling

No new error handling needed. If the app attempts to load a source not listed in `index.json`, it simply won't appear in the UI — this is existing behavior.

## Verification

1. `packages/manga-sources/nettruyen.json` does not exist on disk.
2. `packages/manga-sources/index.json` parses as valid JSON with 12 sources (no nettruyen).
3. `packages/manga-sources/otruyen.json` and `packages/manga-sources/truyenqq.json` still exist.
4. `SourcePicker.tsx` contains no occurrence of "NetTruyen" or "nettruyen".

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

No property-based tests are appropriate for this task. All acceptance criteria are one-time smoke checks (file existence, JSON content, string absence) that do not vary with input. Verification is performed by example-based assertions:

- Assert `nettruyen.json` does not exist.
- Assert `index.json` is valid JSON without a nettruyen entry.
- Assert `otruyen.json` and `truyenqq.json` exist.
- Assert `SourcePicker.tsx` does not contain "NetTruyen" or "nettruyen".
