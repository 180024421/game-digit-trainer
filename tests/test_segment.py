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


def test_crops_from_full_boxes():
    from game_digit_trainer.project import PreprocessConfig
    from game_digit_trainer.segment import crops_from_full_boxes

    canvas = np.zeros((40, 120, 3), dtype=np.uint8)
    cv2.putText(canvas, "1", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(canvas, "2", (45, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    boxes = [(0, 0, 35, 40), (40, 0, 35, 40)]
    crops = crops_from_full_boxes(canvas, boxes, PreprocessConfig(binarize="none"))
    assert len(crops) == 2


def test_create_project(tmp_path: Path):
    proj = create_project("demo_game", base=tmp_path)
    assert (proj.root / "config.json").is_file()
    assert (proj.dataset_dir / "0").is_dir()


def test_relabel_and_return_to_pending(tmp_path: Path):
    from game_digit_trainer.segment import (
        list_all_labeled,
        move_to_label,
        move_to_pending,
        relabel_dataset_file,
    )

    proj = create_project("relabel_demo", base=tmp_path)
    pending = proj.pending_dir / "char_a.png"
    pending.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    labeled = move_to_label(proj, pending, "1")
    assert labeled.exists()
    assert not pending.exists()
    assert labeled.parent.name == "1"

    labeled2 = relabel_dataset_file(proj, labeled, "2")
    assert labeled2.exists()
    assert labeled2.parent.name == "2"
    assert not labeled.exists()

    # same label should be no-op
    same = relabel_dataset_file(proj, labeled2, "2")
    assert same == labeled2

    items = list_all_labeled(proj)
    assert len(items) == 1
    assert items[0][1] == "2"

    back = move_to_pending(proj, labeled2)
    assert back.exists()
    assert back.parent == proj.pending_dir
    assert list_all_labeled(proj) == []
