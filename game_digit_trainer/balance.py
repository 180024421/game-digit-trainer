"""类别样本均衡检查。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def balance_warnings(counts: dict[str, int], *, min_per_class: int = 5) -> list[str]:
    """返回人类可读警告列表（空表示基本均衡）。"""
    warnings: list[str] = []
    positive = {k: v for k, v in counts.items() if v > 0}
    if not positive:
        return ["尚无已标注样本"]
    total = sum(counts.values())
    zeros = [k for k, v in counts.items() if v == 0]
    digit_zeros = [k for k in zeros if k.isdigit()]
    other_zeros = [k for k in zeros if not k.isdigit()]
    if digit_zeros and total >= 10:
        warnings.append(f"数字类无样本: {', '.join(digit_zeros[:8])}" + ("…" if len(digit_zeros) > 8 else ""))
    if other_zeros and total >= 20:
        warnings.append(f"符号/单位无样本: {', '.join(other_zeros[:6])}")

    vals = list(positive.values())
    lo, hi = min(vals), max(vals)
    if lo > 0 and hi >= min_per_class * 3 and hi / max(lo, 1) >= 5:
        low_classes = [k for k, v in positive.items() if v == lo]
        high_classes = [k for k, v in positive.items() if v == hi]
        warnings.append(
            f"样本不均：最多 {hi}（{high_classes[0]}）/ 最少 {lo}（{low_classes[0]}），建议补少的类"
        )
    thin = [k for k, v in positive.items() if 0 < v < min_per_class]
    if thin and total >= 15:
        warnings.append(f"样本偏少（<{min_per_class}）: {', '.join(thin[:8])}")
    return warnings


def scarce_classes(counts: dict[str, int], *, min_per_class: int = 8, top_n: int = 12) -> list[tuple[str, int]]:
    """返回需要补样的类（数量升序）。"""
    items = [(k, int(v)) for k, v in counts.items() if int(v) < min_per_class]
    items.sort(key=lambda t: (t[1], t[0]))
    return items[:top_n]


def format_balance_text(counts: dict[str, int]) -> str:
    warns = balance_warnings(counts)
    scarce = scarce_classes(counts)
    lines: list[str] = []
    if warns:
        lines.append("均衡提示：\n- " + "\n- ".join(warns))
    else:
        lines.append("类别均衡：看起来还行")
    if scarce:
        tip = "、".join(f"{k}({v})" for k, v in scarce[:8])
        lines.append(f"建议优先刷：{tip}")
    return "\n".join(lines)


def boost_scarce_classes(
    dataset_dir: Path,
    classes: list[str],
    counts: dict[str, int],
    *,
    target: int = 12,
    max_per_class: int = 20,
) -> dict[str, int]:
    """对稀缺类做简单增强拷贝（抖动/对比），返回各类新增数量。"""
    added: dict[str, int] = {}
    for name in classes:
        n = int(counts.get(name, 0))
        if n <= 0 or n >= target:
            continue
        folder = dataset_dir / name
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.png"))
        if not files:
            continue
        need = min(target - n, max_per_class)
        created = 0
        i = 0
        while created < need:
            src = files[i % len(files)]
            i += 1
            raw = np.fromfile(str(src), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            aug = img.copy()
            alpha = 0.85 + 0.3 * ((created % 5) / 4)
            aug = np.clip(aug.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            if created % 2 == 0:
                M = np.float32([[1, 0, (created % 3) - 1], [0, 1, ((created // 2) % 3) - 1]])
                aug = cv2.warpAffine(
                    aug, M, (aug.shape[1], aug.shape[0]), borderMode=cv2.BORDER_REPLICATE
                )
            dest = folder / f"boost_{src.stem}_{created}.png"
            ok, buf = cv2.imencode(".png", aug)
            if not ok:
                continue
            dest.write_bytes(buf.tobytes())
            created += 1
        if created:
            added[name] = created
    return added
