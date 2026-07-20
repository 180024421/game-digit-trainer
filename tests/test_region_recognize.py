"""区域识别：字框解析（训练按字，识别可框区域）。"""
from __future__ import annotations

import numpy as np

from game_digit_trainer.project import PreprocessConfig
from game_digit_trainer.segment import looks_like_region_box, resolve_recognize_boxes


def test_looks_like_region_box():
    assert looks_like_region_box((10, 10, 91, 29))
    assert not looks_like_region_box((10, 10, 12, 28))


def test_resolve_prefers_manual_char_boxes():
    bgr = np.zeros((40, 200, 3), dtype=np.uint8)
    boxes = [(10, 5, 12, 28), (30, 5, 12, 28), (50, 5, 14, 28)]
    out, mode = resolve_recognize_boxes(
        bgr,
        preprocess=PreprocessConfig(),
        roi=None,
        boxes=boxes,
        max_gap=3,
    )
    assert mode == "manual"
    assert out == boxes


def test_resolve_wide_box_as_region():
    # white digits on black in a wide strip
    bgr = np.zeros((40, 120, 3), dtype=np.uint8)
    # three blobs
    bgr[8:32, 10:22] = 255
    bgr[8:32, 40:52] = 255
    bgr[8:32, 70:95] = 255
    wide = (5, 5, 100, 30)
    out, mode = resolve_recognize_boxes(
        bgr,
        preprocess=PreprocessConfig(binarize="none"),
        roi=None,
        boxes=[wide],
        max_gap=3,
    )
    assert mode == "wide_box"
    assert len(out) >= 2
