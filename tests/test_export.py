import json
from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.export_onnx import export_onnx
from game_digit_trainer.project import create_project
from game_digit_trainer.train import train_project


def test_train_and_export_manifest(tmp_path: Path):
    proj = create_project("tiny", base=tmp_path)
    # synthesize a few samples per digit 0 and 1
    for label in ("0", "1"):
        for i in range(6):
            img = np.zeros((32, 32), dtype=np.uint8)
            cv2.putText(img, label, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
            path = proj.dataset_dir / label / f"{i}.png"
            ok, buf = cv2.imencode(".png", img)
            assert ok
            path.write_bytes(buf.tobytes())

    ckpt = train_project(proj, epochs=2, batch_size=4)
    assert ckpt.is_file()
    onnx_path = export_onnx(proj, ckpt)
    assert onnx_path.is_file()
    manifest = json.loads((onnx_path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "game-digit-trainer/v1"
    assert manifest["model"] == "digits.onnx"
    assert (onnx_path.parent / "digits.labels").is_file()
