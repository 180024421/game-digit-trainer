from __future__ import annotations

from pathlib import Path

from game_digit_trainer.box_ops import auto_fix_boxes, merge_neighbor_boxes, merge_tiny_boxes
from game_digit_trainer.studio_pack import LUA_STUB, copy_exports_to_studio


def test_merge_tiny_boxes():
    boxes = [(0, 0, 4, 10), (5, 0, 3, 10), (20, 0, 12, 10)]
    out = merge_tiny_boxes(boxes, max_width=6, max_gap=4)
    assert len(out) == 2
    assert out[0][2] >= 7


def test_merge_neighbor():
    boxes = [(0, 0, 10, 10), (12, 0, 10, 10), (30, 0, 8, 8)]
    out = merge_neighbor_boxes(boxes, 0)
    assert out is not None
    assert len(out) == 2


def test_autofix():
    boxes = [(0, 0, 10, 10), (12, 0, 10, 10), (24, 0, 10, 10), (40, 0, 50, 10)]
    fixed, tips = auto_fix_boxes(boxes)
    assert len(fixed) >= 1
    assert tips  # 宽框应被建议拆


def test_copy_studio(tmp_path: Path):
    exp = tmp_path / "exports"
    exp.mkdir()
    (exp / "digits.onnx").write_bytes(b"onnx")
    (exp / "digits.labels").write_text("0\n1\n", encoding="utf-8")
    (exp / "manifest.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "script" / "models"
    copied = copy_exports_to_studio(exp, dest)
    assert "digits.onnx" in copied
    assert (dest / "recognize_digits.lua").read_text(encoding="utf-8").startswith("-- recognize")
    assert LUA_STUB
