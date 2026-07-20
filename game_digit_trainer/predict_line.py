"""行模型推理：ROI → 整串，无切字。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from game_digit_trainer.labels import display_label
from game_digit_trainer.line_data import LINE_MAX_WIDTH, ctc_greedy_decode, prepare_line_tensor
from game_digit_trainer.model_crnn import DigitCRNN
from game_digit_trainer.preprocess import apply_preprocess, load_bgr
from game_digit_trainer.project import GameProject
from game_digit_trainer.segment import crop_bgr


def load_line_checkpoint(path: Path) -> tuple[DigitCRNN, list[str], int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(ckpt.get("classes") or [])
    if not classes:
        raise ValueError(f"行模型缺 classes: {path}")
    model = DigitCRNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    h = int(ckpt.get("input_height") or 32)
    max_w = int(ckpt.get("input_max_width") or LINE_MAX_WIDTH)
    return model, classes, h, max_w


def predict_line_gray(
    model: DigitCRNN,
    classes: list[str],
    gray: np.ndarray,
    *,
    max_w: int = LINE_MAX_WIDTH,
) -> tuple[str, list[tuple[str, float]], float]:
    arr, _valid_w = prepare_line_tensor(gray, max_w=max_w)
    x = torch.from_numpy(arr).unsqueeze(0)  # 1,1,H,W
    with torch.no_grad():
        logits = model(x)  # T,1,C
        log_prob = torch.log_softmax(logits, dim=2)[:, 0, :].cpu().numpy()
        prob = np.exp(log_prob)
    blank = model.blank_index
    idxs = ctc_greedy_decode(log_prob, blank)
    parts: list[tuple[str, float]] = []
    # 粗置信度：各时间步 max 非 blank 平均
    confs = []
    for t in range(prob.shape[0]):
        row = prob[t]
        bi = int(row.argmax())
        if bi != blank:
            confs.append(float(row[bi]))
    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
    for i in idxs:
        if 0 <= i < len(classes):
            parts.append((classes[i], mean_conf))
    text = "".join(display_label(l) for l, _ in parts)
    return text, parts, mean_conf


def predict_line_roi(
    project: GameProject,
    bgr: np.ndarray,
    region: tuple[int, int, int, int] | None,
    checkpoint: Path,
) -> tuple[str, list[tuple[str, float]], float]:
    model, classes, _h, max_w = load_line_checkpoint(checkpoint)
    sliced = crop_bgr(bgr, region)
    # 行模型：灰度即可；可选项目预处理（不强制二值，避免粘连信息丢失）
    gray = apply_preprocess(sliced, project.config.preprocess)
    return predict_line_gray(model, classes, gray, max_w=max_w)


def predict_line_path(
    project: GameProject,
    image_path: Path,
    checkpoint: Path,
    *,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[str, list[tuple[str, float]], float]:
    bgr = load_bgr(image_path)
    return predict_line_roi(project, bgr, region, checkpoint)
