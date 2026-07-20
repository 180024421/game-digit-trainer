"""导出行模型 CRNN 为 ONNX（exports/line/）。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from game_digit_trainer import __version__
from game_digit_trainer.line_data import LINE_HEIGHT, LINE_MAX_WIDTH
from game_digit_trainer.model_crnn import DigitCRNN
from game_digit_trainer.project import GameProject
from game_digit_trainer.train_line import latest_line_checkpoint


def export_line_onnx(
    project: GameProject,
    checkpoint: Path | None = None,
    *,
    out_dir: Path | None = None,
) -> Path:
    from game_digit_trainer.predict import check_onnx_dependency

    ok, msg = check_onnx_dependency()
    if not ok:
        raise RuntimeError(msg)

    ckpt_path = checkpoint or latest_line_checkpoint(project)
    if not ckpt_path or not ckpt_path.is_file():
        raise FileNotFoundError("没有行模型 checkpoint，请先「训练行模型」")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes: list[str] = list(ckpt.get("classes") or project.config.classes)
    h = int(ckpt.get("input_height") or LINE_HEIGHT)
    max_w = int(ckpt.get("input_max_width") or LINE_MAX_WIDTH)

    model = DigitCRNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dest = out_dir or (project.exports_dir / "line")
    dest.mkdir(parents=True, exist_ok=True)
    onnx_path = dest / "digits_line.onnx"
    labels_path = dest / "digits_line.labels"
    manifest_path = dest / "manifest.json"

    dummy = torch.zeros(1, 1, h, max_w)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch", 3: "width"},
                "logits": {0: "time", 1: "batch"},
            },
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
            dynamic_axes={
                "input": {0: "batch", 3: "width"},
                "logits": {0: "time", 1: "batch"},
            },
            opset_version=17,
        )

    labels_path.write_text("\n".join(classes) + "\n", encoding="utf-8")
    prep = project.config.preprocess
    manifest = {
        "format": "game-digit-trainer/line-v1",
        "kind": "line_crnn",
        "game_id": project.config.game_id,
        "model": "digits_line.onnx",
        "labels": "digits_line.labels",
        "blank_index": len(classes),
        "input": {
            "height": h,
            "max_width": max_w,
            "channels": 1,
            "layout": "NCHW",
            "normalize": "0_1",
            "width_dynamic": True,
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
        "source_checkpoint": str(ckpt_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "README.txt").write_text(
        "行模型包：整 ROI 一次识别，无需切字。\n"
        "拷到 Studio models/line/ 或设 model=models/line/digits_line\n"
        "manifest.kind=line_crnn；运行时见 docs/studio-recognize-digits.md\n",
        encoding="utf-8",
    )
    return onnx_path
