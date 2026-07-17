from __future__ import annotations

import cv2
import numpy as np

from game_digit_trainer.project import PreprocessConfig


def load_bgr(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def apply_preprocess(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """返回单通道 0-255 uint8 图，便于切字与训练。"""
    if cfg.color_filter:
        out = _color_filter(bgr, cfg.color_filter)
    elif cfg.grayscale:
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if cfg.binarize == "otsu":
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif cfg.binarize == "adaptive":
        out = cv2.adaptiveThreshold(
            out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    # none: keep as-is

    if cfg.invert:
        out = 255 - out

    # Prefer white digits on black for consistency when majority is dark
    if cfg.binarize != "none":
        if float(np.mean(out)) > 127:
            out = 255 - out
    return out


def _color_filter(bgr: np.ndarray, spec: dict) -> np.ndarray:
    """HSV 范围过滤，保留目标色为白。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(spec.get("lower", [0, 0, 0]), dtype=np.uint8)
    upper = np.array(spec.get("upper", [180, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def resize_char(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
