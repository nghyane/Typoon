# Handoff — scan stage

## Status khi bàn giao

Pipeline hiện tại:

```
RawSource → prepare_chapter() → PreparedChapter   ✓ done
                                     ↓
                               scan_chapter()      ← MISSING
                                     ↓
                               list[Page]          ← translate_pages() nhận cái này ✓
```

`prepare` hoạt động. `translate_pages(pages, session)` đã có và nhận
`list[Page]` với `Bubble` objects. Cầu nối còn thiếu là `scan_chapter` —
stage nhận `PreparedChapter`, chạy vision pipeline trên từng trang, trả ra
`list[Page]` với `Bubble` đã có `source_text` và masks.

## Việc phải làm

Tạo `typoon/stages/scan.py` với hàm:

```python
def scan_chapter(
    chapter: PreparedChapter,
    runtime: VisionRuntime,
    *,
    artifacts: ArtifactSink | None = None,
) -> list[Page]:
```

Mỗi prepared page:
1. Load image từ `chapter.page_path(index)`
2. Gọi `runtime.scan_page(image)` → `list[VisualTextGroup]`
3. Convert mỗi `VisualTextGroup` thành `Bubble`
4. Gói vào `Page(index=..., bubbles=[...])`
5. Viết artifacts ra `02_detect/`, `03_group/`, `04_ocr/`

## Mapping VisualTextGroup → Bubble

```python
Bubble(
    idx=i,
    page_index=page_index,
    polygon=group.render_polygon,
    erase_masks=group.erase_masks,
    text_masks=group.text_masks,
    source_text=group.text,
    ocr_confidence=group.confidence,
)
```

## Artifacts cần viết (theo RFC-002)

```
02_detect/
  page_0000_boxes.png     unit boxes overlay (có thể tái dùng từ vision/inspect.py)
  detections.json

03_group/
  page_0000_groups.png    group + scope overlay
  page_0000_groups.json

04_ocr/
  page_0000_erased.png    kết quả erase preview
  ocr.json                text + confidence mỗi bubble
```

`vision/inspect.py` đã có các hàm vẽ overlay — tái dùng, không viết lại.

## CLI cần thêm

```
typoon scan <prepared_chapter_dir> [--out <dir>] [--run-id <id>]
```

Verify bằng visual E2E: chạy `scan` trên một `PreparedChapter` đã có,
mở `02_detect/`, `03_group/`, `04_ocr/` và kiểm tra ảnh.

## Không làm

- Không tạo `Session` hay gọi `translate_pages` trong scan
- Không đọc raw source files
- Không stitch hoặc merge page
- Không tạo `engine.py`, `service.py`, hoặc bất kỳ broad module nào

## File liên quan cần đọc trước khi code

```
typoon/domain/prepared.py       PreparedChapter, PreparedPage
typoon/domain/bubble.py         Bubble, Page
typoon/vision/types.py          VisualTextGroup, TextMask
typoon/adapters/vision_runtime.py  VisionRuntime.scan_page()
typoon/runs/artifacts.py        ArtifactSink, FileArtifactSink
typoon/vision/inspect.py        overlay helpers để tái dùng
typoon/stages/prepare.py        pattern tham khảo cho stage structure
docs/wiki/lens-native-grouping.md     hiểu output của scan_page()
```
