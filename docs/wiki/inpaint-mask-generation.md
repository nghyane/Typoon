# Inpaint mask generation — stroke detection & hole fill

## Problem

Scan stage `_build_mask` ở `workers/scan/container/main.py` dùng
`cv2.fillPoly(polygon=group.polygon)` khi group không có `erase_masks`.
Polygon thường = bbox rectangle → mask chiếm toàn bộ vùng chữ, kể cả
background. Inpaint trên mask quá rộng gây artifact.

## Pipeline hiện tại

```text
Raw source → prepare → PreparedChapter
    → scan (Lens OCR + spatial_join) → BubbleGroup:
        - polygon: từ bbox OCR, thường là rectangle
        - erase_masks: per-word tight mask (có thể rỗng)
    → _build_mask: OR erase_masks, fallback fillPoly(polygon)
    → close_mask_per_block (Rust): dilate+erode per group bbox
    → build_inpaint_regions: merge groups by distance, cap at 3
    → flat fill / AOT crop-and-stitch
```

## Vấn đề cụ thể

Khi `erase_masks` rỗng, polygon = bbox → mask dày 100% trong bbox.
Inpaint diffusion/NS/Telea hoặc AOT đều kém trên mask hình chữ nhật lớn.

Ví dụ thực tế: job 12 page 0000, group 0 "武炼":
- bbox: `[1190, 2378, 1915, 2828]` (725×450 px)
- polygon: 4 góc bbox
- erase_polygons: `[]`
- mask sau close: **100% density** trong title region

## Giải pháp: Stroke detection từ ảnh (inpaint stage)

Khi mask gốc trong bbox > 85% density → regenerate mask từ ảnh
`PreparedChapter` bằng edge detection, không dùng polygon fill.

### Thuật toán (áp dụng per-group-bbox)

```
1. CROP ảnh quanh bbox + padding (15% short edge, min 8px)
2. Convert RGB → grayscale
3. Edge detection:
   a. Sobel gradient Gx, Gy (kernel 3×3)
   b. Gradient magnitude M = sqrt(Gx² + Gy²)
   c. Double threshold + hysteresis (Canny-style):
      - high threshold → strong edge (255)
      - low threshold → weak edge (128, promoted if 8-connected to strong)
4. Morphological dilate (elliptical kernel, radius ~7-9px, 2 iterations)
   → nối các edge fragment gần nhau
5. Morphological close (elliptical kernel, radius ~15-19px, 1 iteration)
   → lấp khe hở nhỏ trong stroke
6. Flood-fill holes:
   a. BFS/DFS từ tất cả pixel viền crop (=0, chạm biên)
   b. Pixel =0 không chạm được từ viền → đổi thành =1 (mask)
7. OR vào page mask
```

### Tham số (tune theo loại ảnh)

| Tham số | Mặc định | Mô tả |
|---|---|---|
| Edge low threshold | 25 | Gradient magnitude tối thiểu cho weak edge |
| Edge high threshold | 100 | Gradient magnitude tối thiểu cho strong edge |
| Dilate kernel size | 7 | Bán kính elliptical kernel, pixel |
| Dilate iterations | 2 | Số lần dilate liên tiếp |
| Close kernel size | 15 | Bán kính elliptical kernel, pixel |
| Close iterations | 1 | Số lần close |
| Padding fraction | 0.15 | % của short edge, thêm context quanh bbox |
| Density threshold | 0.85 | Nếu mask/bbox > tỉ lệ này → dùng stroke detect |

### Hạn chế đã biết

1. **Text sáng trên nền tối** (武炼): Canny edge detect hoạt động tốt vì
   dựa trên gradient, không phụ thuộc absolute brightness.
2. **Flat color text trên flat background**: Edge yếu → có thể bỏ sót.
   Fallback về dilate/close/Telea nhưng với mask chặt hơn bbox.
3. **Text quá nhỏ (<20px)**: Có thể không cần edge detect vì mask ban
   đầu thường không quá dày.

## Implementation plan

### Rust (`crates/inpaint/src/page.rs`)

Sửa `close_mask_per_block`:
- Thêm param `img: &[u8]` (flat RGB, 3 byte/pixel)
- Với mỗi group, tính `bbox_density = mask_on / bbox_area`
- Nếu `bbox_density > 0.85`: gọi `detect_edges_in_bbox(img, w, h, bbox, pad)`
- Sau đó tiếp tục dilate+erode+hole_fill như bình thường

Cần thêm các hàm:
- `grayscale(img: &[u8], w, h, x0, y0, x1, y1) -> Vec<u8>`
- `sobel_edges(gray: &[u8], w, h, low, high) -> Vec<u8>`
- `dilate_binary(mask: &mut [u8], w, h, radius)`
- `close_binary(mask: &mut [u8], w, h, radius)`
- `fill_enclosed_holes(mask: &mut [u8], w, h)` — đã có

### Python scan container (tương lai)

Nếu `erase_masks` rỗng:
- Thay vì `cv2.fillPoly(polygon=bbox)`, dùng `cv2.Canny + dilate + close`
- Giảm density mask gốc, inpaint có nhiều context hơn

### Test cases

1. job 12 page 0000 "武炼": text sáng trên nền tối, Canny phát hiện edge tốt
2. `fast-bench/*`: manga thông thường, dialogue balloon, mask nhỏ
3. `jpg-bench/*`: manga JPEG artifact, cần kiểm tra nhiễu edge

## Artifacts kiểm chứng

```
debug-runs/inpaint-real-demo-job12-page0000/
  edge-detect-compare.jpg      — original, mask, NS, Telea (title crop)
  edge-allgroups-fullpage.jpg  — full page overview
  edge-allgroups-ns.png        — full page NS inpaint result
  edge-allgroups-telea.png     — full page Telea inpaint result
  edge-allgroups-mask.png      — full page mask
  stroke-detect-demo.jpg       — earlier stroke-detection attempt (dùng màu)
  mask-holefill-demo.jpg       — hole-fill comparison (no effect on full rect)
```

## Routing decision

```text
mask density > 85% in bbox → edge-detect từ ảnh → close → hole fill
                             → inpaint (cv2.INPAINT_NS/INPAINT_TELEA hoặc AOT)

mask density ≤ 85%           → dùng mask gốc → close hiện tại
                             → flat fill nếu nền đơn sắc
                             → AOT crop-and-stitch nếu nền phức tạp
```
