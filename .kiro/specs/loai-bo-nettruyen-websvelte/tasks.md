# Implementation Plan: Remove NetTruyen Source

## Overview

Delete the nettruyen manga source from the project: remove its manifest file, its registry entry, and stale code references. Verify other Vietnamese sources remain intact.

## Tasks

- [ ] 1. Remove nettruyen source files and references
  - [ ] 1.1 Delete `packages/manga-sources/nettruyen.json`
    - Remove the file from disk
    - _Requirements: 1.1_

  - [ ] 1.2 Remove nettruyen entry from `packages/manga-sources/index.json`
    - Remove the object `{ "id": "nettruyen", "url": "./nettruyen.json" }` from the `sources` array
    - Ensure the resulting JSON is valid (no trailing commas, correct structure)
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 1.3 Remove NetTruyen comment line from `web/src/features/reader/components/SourcePicker.tsx`
    - Delete the ASCII-art comment line containing "NetTruyen"
    - _Requirements: 3.1_

- [ ] 2. Verification checkpoint
  - [ ] 2.1 Verify nettruyen removal and other sources intact
    - Confirm `packages/manga-sources/nettruyen.json` does not exist
    - Confirm `packages/manga-sources/index.json` is valid JSON without nettruyen entry
    - Confirm `packages/manga-sources/otruyen.json` and `packages/manga-sources/truyenqq.json` still exist
    - Confirm `SourcePicker.tsx` contains no "NetTruyen" or "nettruyen" string
    - Ensure all tests pass, ask the user if questions arise.
    - _Requirements: 4.1, 4.2_

## Notes

- This is a pure deletion task — no new code is written
- All changes are file deletions or single-line removals
- The source-loading pipeline architecture is unchanged

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1"] }
  ]
}
```
