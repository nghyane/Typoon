from __future__ import annotations

import numpy as np

from typoon.vision.chapter_images import LazyPageProvider, StitchedStrip


class _Source:
    def __init__(self, images: list[np.ndarray]) -> None:
        self.images = images

    def page_count(self) -> int:
        return len(self.images)

    def load_page(self, index: int) -> np.ndarray:
        return self.images[index]


def test_lazy_page_provider_matches_stitched_crop():
    images = [
        np.full((10, 4, 3), 10, dtype=np.uint8),
        np.full((12, 6, 3), 20, dtype=np.uint8),
        np.full((8, 4, 3), 30, dtype=np.uint8),
    ]
    strip = StitchedStrip.from_pages(images)
    total_h = sum(strip.heights)
    provider = LazyPageProvider.build(
        _Source(images), strip.heights, strip.width,
        [(5, 18), (18, total_h)],
    )

    np.testing.assert_array_equal(provider.page(0), strip.image[5:18])
    np.testing.assert_array_equal(provider.page(1), strip.image[18:total_h])
