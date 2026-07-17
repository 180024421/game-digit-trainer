from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from game_digit_trainer.model import DigitCNN
from game_digit_trainer.preprocess import apply_preprocess, load_bgr
from game_digit_trainer.project import GameProject
from game_digit_trainer.segment import prepare_tensor_image, segment_binary


def load_checkpoint(path: Path) -> tuple[DigitCNN, list[str], int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(ckpt["classes"])
    w = int(ckpt["input_width"])
    h = int(ckpt["input_height"])
    model = DigitCNN(num_classes=len(classes), in_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes, w, h


def predict_char(
    model: DigitCNN,
    classes: list[str],
    gray: np.ndarray,
    width: int,
    height: int,
) -> tuple[str, float]:
    arr = prepare_tensor_image(gray, width, height)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax().item())
        conf = float(prob[idx].item())
    return classes[idx], conf


def predict_image_string(
    project: GameProject,
    image_path: Path,
    checkpoint: Path,
    *,
    conf_threshold: float = 0.5,
) -> tuple[str, list[tuple[str, float]]]:
    model, classes, w, h = load_checkpoint(checkpoint)
    bgr = load_bgr(image_path)
    binary = apply_preprocess(bgr, project.config.preprocess)
    crops = segment_binary(binary)
    parts: list[tuple[str, float]] = []
    from game_digit_trainer.labels import display_label

    for crop in crops:
        label, conf = predict_char(model, classes, crop.image, w, h)
        parts.append((label, conf))
    if any(c < conf_threshold for _, c in parts):
        text = "".join("?" if c < conf_threshold else display_label(l) for l, c in parts)
    else:
        text = "".join(display_label(l) for l, _ in parts)
    return text, parts


def predict_pending_file(
    checkpoint: Path,
    pending: Path,
    classes: list[str],
    width: int,
    height: int,
) -> tuple[str, float]:
    model, ck_classes, w, h = load_checkpoint(checkpoint)
    use_classes = ck_classes or classes
    raw = np.fromfile(str(pending), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取: {pending}")
    return predict_char(model, use_classes, img, w or width, h or height)
