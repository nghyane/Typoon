"""Smart pagination — find split points that never cut bubbles.

Used by engine to split long manhwa/webtoon pages into ~1800px
logical pages after detection.
"""

from __future__ import annotations

from itertools import accumulate

_TARGET_HEIGHT = 1800
_MIN_HEIGHT = 800
_MAX_HEIGHT = 2500
_MARGIN = 20


def smart_split(
    total_h: int,
    original_heights: list[int],
    bubble_y_ranges: list[tuple[float, float]],
) -> list[int]:
    """Compute split points (y-coordinates) that never cut bubbles."""
    if len(original_heights) <= 1 and total_h <= _MAX_HEIGHT:
        return []
    if total_h <= _MAX_HEIGHT:
        return list(accumulate(original_heights[:-1]))

    return _repaginate(total_h, bubble_y_ranges)


def _repaginate(total_h: int, bubble_ranges: list[tuple[float, float]]) -> list[int]:
    sorted_ranges = sorted(bubble_ranges)
    cuts: list[int] = []
    y = 0
    while y + _MAX_HEIGHT < total_h:
        ideal = y + _TARGET_HEIGHT
        lo = y + _MIN_HEIGHT
        hi = min(y + _MAX_HEIGHT, total_h)
        cuts.append(_nearest_safe(ideal, sorted_ranges, lo, hi))
        y = cuts[-1]
    return cuts


def _nearest_safe(ideal: int, ranges: list[tuple[float, float]], lo: int, hi: int) -> int:
    forbidden = [
        (max(int(r[0]) - _MARGIN, lo), min(int(r[1]) + _MARGIN, hi))
        for r in ranges if r[0] < hi and r[1] > lo
    ]
    if not forbidden:
        return ideal

    forbidden.sort()
    merged = [forbidden[0]]
    for f_lo, f_hi in forbidden[1:]:
        if f_lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], f_hi))
        else:
            merged.append((f_lo, f_hi))

    safe: list[int] = []
    if lo < merged[0][0]:
        safe.append(min(ideal, merged[0][0] - 1))
    for i in range(len(merged) - 1):
        gap_s, gap_e = merged[i][1], merged[i + 1][0]
        if gap_s < gap_e:
            safe.append(max(gap_s, min(ideal, gap_e - 1)))
    if merged[-1][1] < hi:
        safe.append(max(ideal, merged[-1][1]))

    return min(safe, key=lambda p: abs(p - ideal)) if safe else ideal
