# Web Svelte reader port — tóm tắt phiên làm việc

Ngày: 2026-06-14

## Mục tiêu

Chuyển UI/logic reader từ React `../web` sang Svelte `web-svelte`, đặc biệt là luồng dịch realtime vì bản React đang lỗi.

## Những phần đã làm

### 1. Clone UI reader từ React sang Svelte

Đã port các phần chính của React reader:

- Top bar: nút quay lại, tên truyện, chip chương, chip nguồn, progress hairline.
- Bottom pill: chương trước/sau, page indicator, nút dịch, nút `Aa` mở settings.
- Reader modes:
  - `webtoon`: cuộn dọc nhiều trang.
  - `standard` / `rtl` / `vertical`: pager một trang.
- Tap/keyboard navigation cho pager và tap giữa màn hình để ẩn/hiện chrome.
- Settings sheet: chọn mode đọc và max page width.
- Chapter picker: modal danh sách chương + search.
- Source picker: modal chọn version/source của chapter.

File chính đã thêm/sửa:

- `src/routes/(reader)/r/[workId]/[numberNorm]/+page.svelte`
- `src/lib/reader/PageRenderer.svelte`
- `src/lib/reader/ChapterPicker.svelte`
- `src/lib/reader/SourcePicker.svelte`
- `src/lib/reader/ReaderSettingsSheet.svelte`

### 2. Phân tích và port logic translation

React có 2 hướng translation:

1. `useTranslationPipeline.ts`: tạo run nhưng gần như chỉ cập nhật progress, chưa attach overlay đúng vào page host.
2. `translation/session.ts`: đầy đủ hơn — load runtime, tạo vision runtime, OCR, translator, subscribe event, attach overlay, cleanup.

Đã chọn hướng thứ 2 làm mẫu port sang Svelte.

Logic Svelte đã đưa vào `src/lib/translate.svelte.ts`:

- Dynamic import `@typoon/client/web-reader`.
- `ensureMangaFontLoaded()` trước khi render overlay.
- Detect browser capabilities.
- Tạo optional ONNX detector (`OrtRuntime`, `OrtSessionPool`, `ModelRepository`, `MangaTextRegionDetector`).
- Tạo `ComlinkVisionRuntime` nếu có `Worker + OffscreenCanvas`, fallback `MainThreadVisionRuntime`.
- Tạo `TranslationRuntime` với:
  - `LensTextRecognizer`
  - `DeepLTranslateWeb`
  - concurrency mobile/desktop
- Tạo translation run với `preparation: { type: 'continuous-strip' }`.
- Subscribe events:
  - `progress`
  - `page-status`
  - `page-overlay`
  - `failed`
- Register overlay host theo page index từ `PageRenderer.svelte`.
- Attach overlay bằng `attachOverlay(..., { eraseStrategy: 'flat-fill' })`.
- Queue attach overlay ưu tiên page gần viewport.
- `stop()` / `destroy()` cancel run, dispose runtime, remove overlay DOM.

### 3. Data/source cho reader

Đã mở rộng data load để UI reader có đủ thông tin cho picker:

- `chapters`: danh sách chương đã sort asc.
- `versions`: các source version của chapter hiện tại.
- `selectedVersionKey`: source version đang đọc.

File liên quan:

- `src/routes/(reader)/r/[workId]/[numberNorm]/+page.ts`
- `src/lib/types.ts`
- `src/lib/work/chapters.ts`

Đã thêm helper:

- `versionKeyOf(version)` = `${sourceId}:${ref.id}`.

### 4. Local settings mở rộng

Đã thêm vào `src/lib/localSettings.svelte.ts`:

- `reader_page_width`
- `reader_source_prefs`
- `SourcePref`
- clamp width min/max

Mục đích:

- Lưu source version user đã chọn theo work.
- Lưu max width của reader giống React settings.

### 5. Dependencies/runtime

Đã thêm dependency cần cho web translation runtime:

- `onnxruntime-web`
- `chrome-lens-ocr`

Vì `@typoon/client` runtime import các package này.

## Verification đã chạy

Trước khi thử cleanup typing/alias:

- `npm run check`: pass, còn 2 warning cũ ở `LinkSearchModal.svelte`.
- `npm run build`: pass, tạo worker `vision.worker` và bundle `web-reader`.

Sau đó có thử bỏ ambient module shim và map trực tiếp `@typoon/client/*` qua `tsconfig.paths`; cách này không đúng với SvelteKit vì:

- SvelteKit cảnh báo không nên dùng `baseUrl/paths` trong app tsconfig, nên dùng `kit.alias`.
- Typecheck kéo toàn bộ `../packages/client/src` vào workspace và lộ nhiều lỗi type do TS/DOM lib version mismatch (`Uint8Array<ArrayBufferLike>` vs `BodyInit`, `ImageDataArray`, ...).

=> Cần sửa lại ở bước tiếp theo: bỏ `tsconfig.paths` hack hoặc chuẩn hóa package export/types của `@typoon/client`.

## Các điểm còn hacky / sai lệch cần xử lý tiếp

1. `src/lib/translate.svelte.ts` đang quá lớn, vừa là session, runtime factory, overlay manager. Nên tách theo pattern rõ hơn:
   - `translation/runtime.ts`
   - `translation/session.svelte.ts`
   - `translation/overlay.ts`
2. Dynamic import `@typoon/client/web-reader` đang phụ thuộc alias Vite. Cần pattern chuẩn SvelteKit/package:
   - thêm workspace package đúng cách, hoặc
   - thêm `exports`/types trong `packages/client/package.json`, hoặc
   - dùng `kit.alias` thay vì `tsconfig.paths`.
3. Hardcode runtime:
   - HuggingFace repo `nghyane/comic-detr`
   - revision `v1`
   - translator `DeepLTranslateWeb`
   - concurrency numbers
   Nên đưa vào config/factory.
4. Reader route đang ôm nhiều logic:
   - source switching
   - reader navigation
   - scroll progress
   - translation lifecycle
   Nên tách thành Svelte state classes/hooks tương tự `ChapterPages`.
5. Source preference đang apply bằng effect trong page; nên đóng gói thành reader source state/resolver để tránh effect phức tạp.
6. `PageRenderer.svelte` tự decode canvas và register overlay; ổn để chạy, nhưng có thể tách canvas action riêng để reuse và test.
7. `LinkSearchModal.svelte` vẫn có warning cũ về `$state(workTitle)` capture initial prop.

## Next step khuyến nghị

1. Khôi phục/check lại typing đúng chuẩn SvelteKit:
   - remove `baseUrl/paths` khỏi `tsconfig.json` nếu còn.
   - dùng `kit.alias` hoặc package export cho `@typoon/client`.
2. Tách `Translator` thành session/runtime/overlay modules nhỏ.
3. Tách reader route state thành các Svelte classes/hooks.
4. Chạy lại:
   - `npm run check`
   - `npm run build`
5. Sau đó test browser thực tế: load reader, đổi source, bấm dịch, xác nhận overlay attach đúng trang.
