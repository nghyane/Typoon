# RFC-003: Scan Stage

## Status

Proposed

## Problem

`prepare_chapter()` tạo ra `PreparedChapter` nhưng không có stage nào đọc
nó và chạy vision pipeline một cách chính thức. `inspect-vision` là debug
tool, không phải stage — nó không nhận `PreparedChapter`, không trả ra
domain types, và không viết artifacts vào đúng layout.

`translate_pages(pages, session)` đã tồn tại và nhận `list[Page]` với
`Bubble`. Cầu nối còn thiếu là bước chuyển đổi `PreparedChapter` →
`list[Page]`.

## Decision

Tạo `typoon/stages/scan.py` với entry point:

```python
def scan_chapter(
    chapter: PreparedChapter,
    runtime: VisionRuntime,
    *,
    artifacts: ArtifactSink | None = None,
) -> list[Page]:
```

Stage này:
- nhận `PreparedChapter` (không đọc raw source)
- gọi `runtime.scan_page(image)` cho từng page
- convert `VisualTextGroup` → `Bubble`
- trả ra `list[Page]` với `source_text` và masks đã điền
- viết artifacts vào `02_detect/`, `03_group/`, `04_ocr/`

## Mapping

```text
VisualTextGroup.render_polygon  → Bubble.polygon
VisualTextGroup.erase_masks     → Bubble.erase_masks
VisualTextGroup.text_masks      → Bubble.text_masks
VisualTextGroup.text            → Bubble.source_text
VisualTextGroup.confidence      → Bubble.ocr_confidence
```

`Bubble.idx` = thứ tự trong trang. `Bubble.page_index` = `PreparedPage.index`.

## Artifacts

Theo RFC-002, stage phải viết:

```text
02_detect/
  page_{N:04d}_boxes.png      unit detection overlay
  detections.json             tất cả units mỗi page (raw data)

03_group/
  page_{N:04d}_groups.png     groups + scope overlay
  groups.json                 accepted groups mỗi page

04_ocr/
  page_{N:04d}_erased.png     erase preview
  ocr.json                    text + confidence mỗi bubble
```

Overlay helpers đã có trong `vision/inspect.py` — tái dùng, không viết lại.

## CLI

```
typoon scan <prepared_chapter_dir> [--out <scan_dir>] [--run-id <id>]
```

Output mặc định: `<prepared_chapter_dir>/../scan/` hoặc cạnh prepared dir.

## Dependency

```
stages/scan.py → domain (PreparedChapter, Bubble, Page)
stages/scan.py → adapters (VisionRuntime)
stages/scan.py → runs (ArtifactSink)
stages/scan.py → vision/inspect.py (overlay helpers)
```

Không được import từ `cli/` hoặc `translation/`.

## Definition of done

- `scan_chapter()` chạy trên `PreparedChapter` từ `typoon prepare`
- Viết đủ artifacts vào `02_detect/`, `03_group/`, `04_ocr/`
- CLI `typoon scan` chạy được và in path tới artifacts
- Output `list[Page]` có thể truyền thẳng vào `translate_pages()`
- Không có raw source read sau prepare

## Non-goals

- Không xử lý translate hay render trong stage này
- Không tạo `Session` hay gọi LLM
- Không merge page hay cross-page logic
