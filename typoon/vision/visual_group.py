"""Canonical visual text group geometry helpers."""

from __future__ import annotations

from dataclasses import replace

from .types import TextMask, VisualTextGroup


def bbox_from_polygon(polygon: list[list[float]]) -> list[int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def offset_mask(mask: TextMask, dy: int) -> TextMask:
    return TextMask(x=mask.x, y=mask.y + dy, image=mask.image)


def offset_polygon(polygon: list[list[float]], dy: int) -> list[list[float]]:
    return [[p[0], p[1] + dy] for p in polygon]


def offset_bbox(box: list[int] | None, dy: int) -> list[int] | None:
    if box is None:
        return None
    return [box[0], box[1] + dy, box[2], box[3] + dy]


def offset_group(group: VisualTextGroup, dy: int) -> VisualTextGroup:
    return replace(
        group,
        text_polygon=offset_polygon(group.text_polygon, dy),
        render_polygon=offset_polygon(group.render_polygon, dy),
        text_bbox=offset_bbox(group.text_bbox, dy),
        mask_bbox=offset_bbox(group.mask_bbox, dy),
        fit_bbox=offset_bbox(group.fit_bbox, dy),
        erase_bbox=offset_bbox(group.erase_bbox, dy),
        scope_bbox=offset_bbox(group.scope_bbox, dy),
        text_masks=[offset_mask(m, dy) for m in group.text_masks],
        erase_masks=[offset_mask(m, dy) for m in group.erase_masks],
    )


def clip_mask_to_y(mask: TextMask, y_start: int, y_end: int) -> TextMask | None:
    mask_h = mask.image.shape[0]
    mask_y1 = mask.y
    mask_y2 = mask.y + mask_h
    overlap_y1 = max(mask_y1, y_start)
    overlap_y2 = min(mask_y2, y_end)
    if overlap_y2 <= overlap_y1:
        return None
    crop_y1 = overlap_y1 - mask_y1
    crop_y2 = overlap_y2 - mask_y1
    return TextMask(
        x=mask.x,
        y=overlap_y1 - y_start,
        image=mask.image[crop_y1:crop_y2].copy(),
    )


def clip_polygon_to_y(polygon: list[list[float]], y_start: int, y_end: int, page_w: int) -> list[list[float]]:
    return [
        [min(max(p[0], 0.0), float(page_w)), min(max(p[1], y_start), y_end) - y_start]
        for p in polygon
    ]


def clip_bbox_to_y(box: list[int] | None, y_start: int, y_end: int, page_w: int) -> list[int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if y2 <= y_start or y1 >= y_end:
        return None
    return [
        max(0, min(x1, page_w)),
        max(y1, y_start) - y_start,
        max(0, min(x2, page_w)),
        min(y2, y_end) - y_start,
    ]


def clip_group_to_slice(group: VisualTextGroup, y_start: int, y_end: int, page_w: int) -> VisualTextGroup | None:
    render_bbox = bbox_from_polygon(group.render_polygon)
    if render_bbox[3] <= y_start or render_bbox[1] >= y_end:
        return None
    text_bbox = clip_bbox_to_y(group.text_bbox, y_start, y_end, page_w)
    mask_bbox = clip_bbox_to_y(group.mask_bbox, y_start, y_end, page_w)
    fit_bbox = clip_bbox_to_y(group.fit_bbox, y_start, y_end, page_w)
    erase_bbox = clip_bbox_to_y(group.erase_bbox, y_start, y_end, page_w)
    if text_bbox is None or mask_bbox is None or fit_bbox is None or erase_bbox is None:
        return None
    return replace(
        group,
        text_polygon=clip_polygon_to_y(group.text_polygon, y_start, y_end, page_w),
        render_polygon=clip_polygon_to_y(group.render_polygon, y_start, y_end, page_w),
        text_bbox=text_bbox,
        mask_bbox=mask_bbox,
        fit_bbox=fit_bbox,
        erase_bbox=erase_bbox,
        scope_bbox=clip_bbox_to_y(group.scope_bbox, y_start, y_end, page_w),
        text_masks=[m for mask in group.text_masks if (m := clip_mask_to_y(mask, y_start, y_end)) is not None],
        erase_masks=[m for mask in group.erase_masks if (m := clip_mask_to_y(mask, y_start, y_end)) is not None],
    )
