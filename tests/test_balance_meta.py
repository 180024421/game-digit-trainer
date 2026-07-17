from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.balance import balance_warnings, format_balance_text
from game_digit_trainer.project import create_project
from game_digit_trainer.sample_meta import get_meta, load_meta, record_pending_batch
from game_digit_trainer.segment import CharCrop, move_to_label, save_pending_chars


def test_balance_warnings():
    warns = balance_warnings({"0": 20, "1": 2, "2": 0})
    assert any("不均" in w or "偏少" in w or "无样本" in w for w in warns)
    assert "看起来还行" in format_balance_text({"0": 10, "1": 9, "2": 11})


def test_sample_meta_roundtrip(tmp_path: Path):
    proj = create_project("meta_demo", base=tmp_path)
    src = proj.raw_dir / "hud.png"
    img = np.zeros((40, 80), dtype=np.uint8)
    cv2.imwrite(str(src), img)
    crops = [
        CharCrop(image=np.ones((10, 8), dtype=np.uint8) * 255, x=2, y=4, w=8, h=10),
        CharCrop(image=np.ones((10, 8), dtype=np.uint8) * 255, x=20, y=4, w=8, h=10),
    ]
    paths = save_pending_chars(proj, src, crops)
    assert len(paths) == 2
    meta = get_meta(proj, paths[0].name)
    assert meta is not None
    assert meta["box"] == [2, 4, 8, 10]
    labeled = move_to_label(proj, paths[0], "3")
    assert get_meta(proj, labeled.name) is not None
    assert paths[0].name not in load_meta(proj) or labeled.name == paths[0].name
