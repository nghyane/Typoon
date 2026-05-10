# Hội Mê Truyện browser extension

Pick images from any page and upload them as a chapter to Hội Mê Truyện
(engine: typoon). See `docs/rfc/009-browser-extension.md` for the design.

## Dev

```sh
cd ext
bun install
bun run dev            # Chrome
bun run dev:firefox    # Firefox
```

WXT opens a fresh browser profile with the extension loaded. Edit any
file under `entrypoints/` or `core/` and the browser auto-reloads.

## Layout

```
ext/
  entrypoints/         WXT entrypoints — popup, background SW, content, offscreen
  core/                UI- and runtime-agnostic logic (no chrome.*, no react)
  shell/               chrome.* glue — message bus, storage adapters, hooks
  shared/              UI primitives + design tokens copied from web/
```

`core/` is plain TypeScript that depends only on web standards
(`fetch`, `Blob`, `FormData`). It runs in the SW, the offscreen doc, the
popup, and content scripts unchanged.

`shell/` wires `core/` to `chrome.*` APIs (storage, runtime messaging,
notifications). Anything browser-API-specific lives here.

## Build

```sh
bun run build              # → .output/chrome-mv3/
bun run build:firefox      # → .output/firefox-mv2/
bun run zip                # → packaged zip in .output/
```
