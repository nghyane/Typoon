# Requirements Document

## Introduction

Remove the nettruyen manga source entirely from the Typoon project. The source is no longer maintained and should be cleaned from the manifest registry, its JSON definition file, and any remaining code references. Other Vietnamese sources (otruyen, truyenqq) must remain intact.

## Glossary

- **Source_Registry**: The `packages/manga-sources/index.json` file that lists all available manga source definitions
- **Source_Manifest**: A JSON file in `packages/manga-sources/` that declares a single manga source's configuration (host, selectors, headers, etc.)
- **Build_System**: The web-svelte and web applications that consume manga source definitions via `import.meta.glob` or direct imports from `packages/manga-sources/`

## Requirements

### Requirement 1: Delete nettruyen manifest file

**User Story:** As a maintainer, I want the nettruyen manifest removed, so that the dead source no longer ships in the project.

#### Acceptance Criteria

1. WHEN the cleanup task is executed, THE Build_System SHALL no longer contain the file `packages/manga-sources/nettruyen.json`

### Requirement 2: Remove nettruyen from source registry

**User Story:** As a maintainer, I want the registry to reflect only active sources, so that consumers do not attempt to load a missing manifest.

#### Acceptance Criteria

1. WHEN the cleanup task is executed, THE Source_Registry SHALL not contain any entry with id "nettruyen"
2. THE Source_Registry SHALL retain all other source entries unchanged (community, happymh, otruyen, mangadex, baozimh, e-hentai, hitomi, truyenqq, webtoonscan, nhentai, naver-webtoon, comix)
3. THE Source_Registry SHALL remain valid JSON after the removal

### Requirement 3: Remove nettruyen references in application code

**User Story:** As a maintainer, I want stale references cleaned up, so that the codebase does not mention a removed source.

#### Acceptance Criteria

1. WHEN the cleanup task is executed, THE Build_System SHALL not contain the string "NetTruyen" or "nettruyen" in `web/src/features/reader/components/SourcePicker.tsx`

### Requirement 4: Preserve other Vietnamese sources

**User Story:** As a user, I want otruyen and truyenqq to remain functional, so that Vietnamese manga reading is unaffected.

#### Acceptance Criteria

1. THE Source_Registry SHALL contain entries with id "otruyen" and id "truyenqq"
2. THE Build_System SHALL contain the files `packages/manga-sources/otruyen.json` and `packages/manga-sources/truyenqq.json` unchanged
