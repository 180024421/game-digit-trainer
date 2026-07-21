"""行模型推理：ROI → 整串，无切字。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from game_digit_trainer.labels import display_label
from game_digit_trainer.line_data import LINE_MAX_WIDTH, prepare_line_tensor
from game_digit_trainer.model_crnn import DigitCRNN
from game_digit_trainer.preprocess import load_bgr
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


def _roi_to_line_gray(bgr: np.ndarray, region: tuple[int, int, int, int] | None) -> np.ndarray:
    """与行样本入库一致：只裁 ROI 转灰度，不套 Otsu/反相等项目预处理。

    训练读的是 lines/*.png 原始灰度；推理若再二值化，分布对不上，会乱读。
    """
    sliced = crop_bgr(bgr, region)
    if sliced.ndim == 3:
        return cv2.cvtColor(sliced, cv2.COLOR_BGR2GRAY)
    return sliced


def trim_line_margins(gray: np.ndarray, *, pad: int = 2) -> np.ndarray:
    """裁掉左右几乎无字的空白/噪点，避免蓝框偏松时多读出开头/结尾的假数字。"""
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.size == 0 or g.shape[1] < 8:
        return g
    work = g.astype(np.float32)
    if float(np.mean(work)) > 127:
        work = 255.0 - work
    # 只认「够亮」的像素，弱背景噪点不计入字宽
    bright = float(np.percentile(work, 85))
    ink_thr = max(60.0, bright * 0.55)
    col = (work >= ink_thr).sum(axis=0).astype(np.float32)
    peak = float(col.max()) if col.size else 0.0
    if peak < 2.0:
        return g
    thr = max(peak * 0.25, 2.0)
    xs = np.where(col >= thr)[0]
    if xs.size == 0:
        return g
    x0 = max(0, int(xs[0]) - pad)
    x1 = min(g.shape[1], int(xs[-1]) + 1 + pad)
    if x1 - x0 < 4:
        return g
    return g[:, x0:x1]


def predict_line_gray(
    model: DigitCRNN,
    classes: list[str],
    gray: np.ndarray,
    *,
    max_w: int = LINE_MAX_WIDTH,
) -> tuple[str, list[tuple[str, float]], float]:
    gray = trim_line_margins(gray)
    arr, valid_w = prepare_line_tensor(gray, max_w=max_w)
    x = torch.from_numpy(arr).unsqueeze(0)  # 1,1,H,W
    with torch.no_grad():
        logits = model(x)  # T,1,C
        # 与训练 CTC input_length 一致：只解码有效宽度对应的时间步（CNN 宽约 /4）
        # 否则右侧 pad 黑边会被解出「万亿2」等尾巴
        t_use = int(max(2, min(logits.size(0), valid_w // 4)))
        log_prob = torch.log_softmax(logits[:t_use], dim=2)[:, 0, :].cpu().numpy()
        prob = np.exp(log_prob)
    blank = model.blank_index
    # 按时间步取 argmax，再合并重复 / blank（标准 CTC greedy）
    pred_steps = log_prob.argmax(axis=1)
    idxs: list[int] = []
    step_confs: list[float] = []
    prev = None
    for t, p in enumerate(pred_steps.tolist()):
        if p == blank:
            prev = p
            continue
        if p != prev:
            idxs.append(int(p))
            step_confs.append(float(prob[t, p]))
        prev = p
    parts: list[tuple[str, float]] = []
    for i, conf in zip(idxs, step_confs):
        if 0 <= i < len(classes):
            parts.append((classes[i], conf))
    mean_conf = float(sum(step_confs) / len(step_confs)) if step_confs else 0.0
    text = "".join(display_label(l) for l, _ in parts)
    return text, parts, mean_conf


def predict_line_roi(
    project: GameProject,
    bgr: np.ndarray,
    region: tuple[int, int, int, int] | None,
    checkpoint: Path,
) -> tuple[str, list[tuple[str, float]], float]:
    del project  # 行推理不用项目预处理，与训练入库一致
    model, classes, _h, max_w = load_line_checkpoint(checkpoint)
    gray = _roi_to_line_gray(bgr, region)
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
