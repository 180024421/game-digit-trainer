from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.preprocess import apply_preprocess, load_bgr, resize_char
from game_digit_trainer.project import GameProject, PreprocessConfig


@dataclass
class CharCrop:
    image: np.ndarray  # gray uint8
    x: int
    y: int
    w: int
    h: int


def segment_binary(
    binary: np.ndarray,
    *,
    min_area: int = 8,
    max_gap: int = 3,
) -> list[CharCrop]:
    """对二值图做连通域/投影切字，返回从左到右的字符块。"""
    if binary.ndim != 2:
        raise ValueError("期望单通道二值图")
    # Ensure digits are white
    work = binary.copy()
    if float(np.mean(work)) > 127:
        work = 255 - work

    # Close small gaps inside strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(work > 0)
    if len(xs) == 0:
        return []

    # Vertical projection split
    col_sum = (work > 0).sum(axis=0)
    gaps: list[tuple[int, int]] = []
    in_gap = False
    start = 0
    for i, v in enumerate(col_sum):
        if v == 0:
            if not in_gap:
                in_gap = True
                start = i
        else:
            if in_gap:
                gaps.append((start, i))
                in_gap = False
    if in_gap:
        gaps.append((start, len(col_sum)))

    # Merge narrow gaps (inside a digit) — keep wide gaps as separators
    cuts = [0]
    for a, b in gaps:
        if b - a >= max_gap:
            mid = (a + b) // 2
            if mid > cuts[-1]:
                cuts.append(mid)
    cuts.append(len(col_sum))

    crops: list[CharCrop] = []
    for i in range(len(cuts) - 1):
        x0, x1 = cuts[i], cuts[i + 1]
        if x1 - x0 < 2:
            continue
        strip = work[:, x0:x1]
        rows = np.where(strip.max(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        patch = strip[y0:y1, :]
        if int(patch.sum() / 255) < min_area:
            continue
        # pad a bit
        pad = 2
        ph, pw = patch.shape
        canvas = np.zeros((ph + pad * 2, pw + pad * 2), dtype=np.uint8)
        canvas[pad : pad + ph, pad : pad + pw] = patch
        crops.append(CharCrop(image=canvas, x=x0, y=y0, w=x1 - x0, h=y1 - y0))
    return crops


def crop_bgr(bgr: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if not roi:
        return bgr
    x, y, w, h = roi
    H, W = bgr.shape[:2]
    x0 = max(0, min(x, W - 1))
    y0 = max(0, min(y, H - 1))
    x1 = max(x0 + 1, min(x + w, W))
    y1 = max(y0 + 1, min(y + h, H))
    return bgr[y0:y1, x0:x1].copy()


def crops_from_full_boxes(
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    preprocess: PreprocessConfig,
) -> list[CharCrop]:
    """按全图坐标的字框，切出单字（已按从左到右排序）。"""
    ordered = sorted(boxes, key=lambda b: (b[0], b[1]))
    out: list[CharCrop] = []
    for x, y, w, h in ordered:
        patch_bgr = crop_bgr(bgr, (x, y, w, h))
        gray = apply_preprocess(patch_bgr, preprocess)
        # ensure white digit on black
        if float(np.mean(gray)) > 127:
            gray = 255 - gray
        pad = 2
        ph, pw = gray.shape[:2]
        canvas = np.zeros((ph + pad * 2, pw + pad * 2), dtype=np.uint8)
        canvas[pad : pad + ph, pad : pad + pw] = gray
        out.append(CharCrop(image=canvas, x=x, y=y, w=w, h=h))
    return out


def segment_image(
    path: Path,
    preprocess: PreprocessConfig,
    *,
    roi: tuple[int, int, int, int] | None = None,
    max_gap: int = 3,
    min_area: int = 8,
) -> tuple[np.ndarray, list[CharCrop], np.ndarray]:
    """返回 (binary全图或ROI, crops, 用于预览的bgr切片)。crops 坐标相对 binary。"""
    bgr = load_bgr(path)
    sliced = crop_bgr(bgr, roi)
    binary = apply_preprocess(sliced, preprocess)
    crops = segment_binary(binary, min_area=min_area, max_gap=max_gap)
    return binary, crops, sliced


def save_pending_chars(
    project: GameProject,
    source: Path,
    crops: list[CharCrop],
) -> list[Path]:
    from game_digit_trainer.sample_meta import record_pending_batch

    project.ensure_dirs()
    stem = source.stem
    out: list[Path] = []
    boxes: list[tuple[int, int, int, int]] = []
    for i, crop in enumerate(crops):
        name = f"{stem}_{i:03d}.png"
        dest = project.pending_dir / name
        # imencode for unicode paths on Windows
        ok, buf = cv2.imencode(".png", crop.image)
        if not ok:
            continue
        dest.write_bytes(buf.tobytes())
        out.append(dest)
        boxes.append((crop.x, crop.y, crop.w, crop.h))
    if out:
        record_pending_batch(project, source, out, boxes)
    return out


def move_to_label(project: GameProject, pending: Path, label: str) -> Path:
    from game_digit_trainer.labels import normalize_label
    from game_digit_trainer.sample_meta import rename_meta_key

    name = normalize_label(label)
    if name not in project.config.classes:
        raise ValueError(f"项目未启用类别: {name}")
    dest_dir = project.dataset_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pending.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{pending.stem}_{n}{pending.suffix}"
        n += 1
    dest.write_bytes(pending.read_bytes())
    pending.unlink(missing_ok=True)
    rename_meta_key(project, pending.name, dest.name)
    return dest


def list_dataset_files(project: GameProject, label: str) -> list[Path]:
    folder = project.dataset_dir / label
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )


def move_to_pending(project: GameProject, labeled_path: Path) -> Path:
    """把已标注样本退回待审核。"""
    from game_digit_trainer.sample_meta import rename_meta_key

    project.ensure_dirs()
    dest = project.pending_dir / labeled_path.name
    n = 1
    while dest.exists():
        dest = project.pending_dir / f"{labeled_path.stem}_{n}{labeled_path.suffix}"
        n += 1
    dest.write_bytes(labeled_path.read_bytes())
    labeled_path.unlink(missing_ok=True)
    rename_meta_key(project, labeled_path.name, dest.name)
    return dest


def list_all_labeled(project: GameProject) -> list[tuple[Path, str]]:
    """返回 [(path, class_name), ...]，按修改时间新到旧。"""
    items: list[tuple[Path, str, float]] = []
    for name in project.config.classes:
        for p in list_dataset_files(project, name):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            items.append((p, name, mtime))
    items.sort(key=lambda t: t[2], reverse=True)
    return [(p, name) for p, name, _ in items]


def relabel_dataset_file(project: GameProject, path: Path, new_label: str) -> Path:
    from game_digit_trainer.labels import normalize_label
    from game_digit_trainer.sample_meta import rename_meta_key

    name = normalize_label(new_label)
    if name not in project.config.classes:
        raise ValueError(f"项目未启用类别: {name}")
    # 同目录同标签：无需移动
    if path.parent.name == name and path.is_file():
        return path
    dest_dir = project.dataset_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{path.stem}_{n}{path.suffix}"
        n += 1
    dest.write_bytes(path.read_bytes())
    path.unlink(missing_ok=True)
    rename_meta_key(project, path.name, dest.name)
    return dest


def prepare_tensor_image(
    gray: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Return float32 HxW in 0..1."""
    img = resize_char(gray, width, height).astype(np.float32) / 255.0
    return img
