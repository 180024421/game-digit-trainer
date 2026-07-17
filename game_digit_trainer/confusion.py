"""混淆矩阵与易错对。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from game_digit_trainer.labels import display_label
from game_digit_trainer.model import DigitCNN
from game_digit_trainer.project import GameProject
from game_digit_trainer.train import CharFolderDataset


def compute_confusion(
    project: GameProject,
    checkpoint: Path,
    *,
    max_pairs: int = 12,
) -> dict[str, Any]:
    """在全量已标注样本上算混淆矩阵，返回易错对。"""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes = list(ckpt.get("classes") or project.config.classes)
    w = int(ckpt.get("input_width") or project.config.input_width)
    h = int(ckpt.get("input_height") or project.config.input_height)
    model = DigitCNN(num_classes=len(classes), in_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = CharFolderDataset(project, augment=False)
    # 临时对齐 classes / size
    ds.classes = classes
    ds.class_to_idx = {c: i for i, c in enumerate(classes)}
    # rebuild items for current classes
    items: list[tuple[Path, int]] = []
    for name in classes:
        folder = project.dataset_dir / name
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and name in ds.class_to_idx:
                items.append((p, ds.class_to_idx[name]))
    ds.items = items
    if not ds.items:
        return {"classes": classes, "matrix": [], "pairs": [], "acc": 0.0, "n": 0}

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    n = len(classes)
    mat = np.zeros((n, n), dtype=np.int64)
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(dim=1)
            for t, p in zip(yb.tolist(), pred.tolist()):
                mat[t, p] += 1
                total += 1
                if t == p:
                    correct += 1

    pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(mat[i, j])
            if c <= 0:
                continue
            pairs.append(
                {
                    "true": classes[i],
                    "pred": classes[j],
                    "count": c,
                    "true_display": display_label(classes[i]),
                    "pred_display": display_label(classes[j]),
                }
            )
    pairs.sort(key=lambda x: -x["count"])
    return {
        "classes": classes,
        "matrix": mat.tolist(),
        "pairs": pairs[:max_pairs],
        "acc": correct / max(total, 1),
        "n": total,
    }


def format_confusion_text(report: dict[str, Any]) -> str:
    lines = [
        f"混淆评估：准确率 {float(report.get('acc', 0)):.1%} · 样本 {report.get('n', 0)}",
        "",
        "最易错对（真→预测）：",
    ]
    pairs = report.get("pairs") or []
    if not pairs:
        lines.append("（无明显混淆）")
    else:
        for p in pairs:
            lines.append(
                f"  {p['true_display']} → {p['pred_display']}  ×{p['count']}"
            )
    return "\n".join(lines)
