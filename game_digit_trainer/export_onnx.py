from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from game_digit_trainer import __version__
from game_digit_trainer.model import DigitCNN
from game_digit_trainer.project import GameProject


def export_onnx(
    project: GameProject,
    checkpoint: Path,
    *,
    out_dir: Path | None = None,
) -> Path:
    from game_digit_trainer.predict import check_onnx_dependency

    ok, msg = check_onnx_dependency()
    if not ok:
        raise RuntimeError(msg)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes: list[str] = list(ckpt.get("classes") or project.config.classes)
    w = int(ckpt.get("input_width") or project.config.input_width)
    h = int(ckpt.get("input_height") or project.config.input_height)

    model = DigitCNN(num_classes=len(classes), in_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dest = out_dir or project.exports_dir
    dest.mkdir(parents=True, exist_ok=True)
    onnx_path = dest / "digits.onnx"
    labels_path = dest / "digits.labels"
    manifest_path = dest / "manifest.json"

    dummy = torch.zeros(1, 1, h, w)
    # Prefer legacy exporter when available for broader ORT compatibility
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    labels_path.write_text("\n".join(classes) + "\n", encoding="utf-8")
    prep = project.config.preprocess
    manifest = {
        "format": "game-digit-trainer/v1",
        "game_id": project.config.game_id,
        "model": "digits.onnx",
        "labels": "digits.labels",
        "input": {
            "width": w,
            "height": h,
            "channels": 1,
            "layout": "NCHW",
            "normalize": "0_1",
        },
        "preprocess": {
            "grayscale": prep.grayscale,
            "invert": prep.invert,
            "binarize": prep.binarize,
            "color_filter": prep.color_filter,
        },
        "classes": classes,
        "trainer_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = dest / "README.txt"
    readme.write_text(
        "将本目录 digits.onnx / digits.labels / manifest.json 拷贝到 auto-script 工程 models/\n"
        "运行时接入见 game-digit-trainer 设计规格（v2）。\n",
        encoding="utf-8",
    )
    return onnx_path


def latest_checkpoint(project: GameProject) -> Path | None:
    if not project.runs_dir.is_dir():
        return None
    bests = sorted(project.runs_dir.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return bests[0] if bests else None
