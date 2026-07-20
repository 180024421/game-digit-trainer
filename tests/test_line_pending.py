"""行待审：框选加入 → 填字确认；样本库改金标/删除。"""
from __future__ import annotations

import numpy as np

from game_digit_trainer.line_data import (
    clear_line_samples,
    confirm_line_pending,
    delete_line_sample,
    list_line_labeled,
    list_line_pending,
    save_line_pending,
    update_line_label,
)
from game_digit_trainer.project import create_project


def test_line_pending_roundtrip(tmp_path):
    proj = create_project("line_pending_t", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((40, 120, 3), dtype=np.uint8)
    bgr[10:30, 20:100] = 200
    pending = save_line_pending(proj, bgr, (15, 8, 90, 24), source_name="hud.png")
    assert pending.is_file()
    assert pending in list_line_pending(proj)
    dest = confirm_line_pending(proj, pending, "12万")
    assert dest.is_file()
    assert pending not in list_line_pending(proj)
    labels = (proj.root / "lines" / "labels.jsonl").read_text(encoding="utf-8")
    assert "12万" in labels


def test_line_labeled_manage(tmp_path):
    proj = create_project("line_mgr_t", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((40, 120, 3), dtype=np.uint8)
    bgr[10:30, 20:100] = 200
    p1 = save_line_pending(proj, bgr, (15, 8, 90, 24), source_name="a.png")
    dest = confirm_line_pending(proj, p1, "39万")
    items = list_line_labeled(proj)
    assert len(items) == 1
    assert items[0][1] == "39万"
    new_text = update_line_label(proj, dest, "1.9亿")
    assert new_text == "1.9亿"
    assert list_line_labeled(proj)[0][1] == "1.9亿"
    delete_line_sample(proj, dest)
    assert list_line_labeled(proj) == []
    p2 = save_line_pending(proj, bgr, (15, 8, 90, 24), source_name="b.png")
    confirm_line_pending(proj, p2, "8万")
    assert clear_line_samples(proj) == 1
    assert list_line_labeled(proj) == []
