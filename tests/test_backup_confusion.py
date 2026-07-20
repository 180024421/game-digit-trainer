from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.backup import (
    backup_project,
    export_labels_pack,
    import_labels_pack,
)
from game_digit_trainer.confusion import format_confusion_text
from game_digit_trainer.labels import build_class_list
from game_digit_trainer.line_data import list_line_labeled, lines_labels_path
from game_digit_trainer.project import create_project


def _write_png(path: Path, color: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((16, 24), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())


def test_backup_and_confusion_text(tmp_path: Path):
    proj = create_project("bakgame", base=tmp_path, classes=build_class_list())
    (proj.dataset_dir / "0").mkdir(parents=True, exist_ok=True)
    (proj.dataset_dir / "0" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    z = backup_project(proj, dest=tmp_path / "out.zip")
    assert z.is_file() and z.stat().st_size > 0
    text = format_confusion_text({"acc": 0.9, "n": 10, "pairs": []})
    assert "准确率" in text


def test_export_import_labels_merge(tmp_path: Path):
    src = create_project("lab_src", base=tmp_path, classes=build_class_list(with_units=True))
    _write_png(src.dataset_dir / "1" / "a.png", 50)
    lines = src.root / "lines"
    lines.mkdir(parents=True, exist_ok=True)
    _write_png(lines / "line_0000.png", 60)
    (lines / "labels.jsonl").write_text(
        json.dumps({"image": "line_0000.png", "text": "12万"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pack = export_labels_pack(src, dest=tmp_path / "labels.zip")
    assert pack.is_file()

    dst = create_project("lab_dst", base=tmp_path, classes=build_class_list(with_units=True))
    _write_png(dst.dataset_dir / "1" / "a.png", 70)  # 同名冲突
    _write_png(dst.root / "lines" / "line_0000.png", 80)
    (dst.root / "lines" / "labels.jsonl").write_text(
        json.dumps({"image": "line_0000.png", "text": "9"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = import_labels_pack(dst, pack, mode="merge")
    assert result.dataset_files >= 1
    assert result.line_files >= 1
    assert result.line_labels >= 1
    # 原图保留 + 导入改名
    assert (dst.dataset_dir / "1" / "a.png").is_file()
    assert (dst.dataset_dir / "1" / "a_imp1.png").is_file()
    labeled = {p.name: t for p, t in list_line_labeled(dst)}
    assert "line_0000.png" in labeled
    assert any(n.startswith("line_0000") and n != "line_0000.png" for n in labeled)
    assert "12万" in labeled.values()


def test_import_labels_replace(tmp_path: Path):
    src = create_project("rep_src", base=tmp_path, classes=build_class_list())
    _write_png(src.dataset_dir / "2" / "only.png", 90)
    pack = export_labels_pack(src, dest=tmp_path / "rep.zip")

    dst = create_project("rep_dst", base=tmp_path, classes=build_class_list())
    _write_png(dst.dataset_dir / "0" / "old.png", 20)
    result = import_labels_pack(dst, pack, mode="replace")
    assert result.mode == "replace"
    assert not (dst.dataset_dir / "0" / "old.png").is_file()
    assert (dst.dataset_dir / "2" / "only.png").is_file()
