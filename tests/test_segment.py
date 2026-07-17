from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.project import create_project
from game_digit_trainer.segment import segment_binary


def test_segment_three_digits(tmp_path: Path):
    # spaced digits so vertical projection can split
    canvas = np.zeros((40, 160), dtype=np.uint8)
    for i, ch in enumerate("123"):
        cv2.putText(canvas, ch, (10 + i * 45, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    crops = segment_binary(canvas, min_area=5, max_gap=2)
    assert len(crops) >= 2


def test_create_project(tmp_path: Path):
    proj = create_project("demo_game", base=tmp_path)
    assert (proj.root / "config.json").is_file()
    assert (proj.dataset_dir / "0").is_dir()
