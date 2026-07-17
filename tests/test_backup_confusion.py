from __future__ import annotations

from pathlib import Path

from game_digit_trainer.backup import backup_project
from game_digit_trainer.confusion import format_confusion_text
from game_digit_trainer.labels import build_class_list
from game_digit_trainer.project import create_project


def test_backup_and_confusion_text(tmp_path: Path):
    proj = create_project("bakgame", base=tmp_path, classes=build_class_list())
    (proj.dataset_dir / "0").mkdir(parents=True, exist_ok=True)
    (proj.dataset_dir / "0" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    z = backup_project(proj, dest=tmp_path / "out.zip")
    assert z.is_file() and z.stat().st_size > 0
    text = format_confusion_text({"acc": 0.9, "n": 10, "pairs": []})
    assert "准确率" in text
