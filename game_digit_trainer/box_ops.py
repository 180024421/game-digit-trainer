"""字框后处理：合并碎框、建议拆粘连。"""
from __future__ import annotations


Box = tuple[int, int, int, int]


def merge_tiny_boxes(
    boxes: list[Box],
    *,
    max_width: int = 6,
    max_gap: int = 4,
) -> list[Box]:
    """把过窄且水平相邻的框合并（过碎切字修复）。"""
    if len(boxes) < 2:
        return list(boxes)
    ordered = sorted(boxes, key=lambda b: (b[0], b[1]))
    out: list[Box] = [ordered[0]]
    for x, y, w, h in ordered[1:]:
        px, py, pw, ph = out[-1]
        gap = x - (px + pw)
        if (w <= max_width or pw <= max_width) and 0 <= gap <= max_gap:
            nx = px
            ny = min(py, y)
            nx2 = max(px + pw, x + w)
            ny2 = max(py + ph, y + h)
            out[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            out.append((x, y, w, h))
    return out


def merge_neighbor_boxes(boxes: list[Box], index: int) -> list[Box] | None:
    """把 index 与右侧相邻框合并；若是最后一个则与左侧合并。"""
    if index < 0 or index >= len(boxes) or len(boxes) < 2:
        return None
    ordered = sorted(enumerate(boxes), key=lambda t: (t[1][0], t[1][1]))
    # map original index → position in ordered
    pos = next(i for i, (oi, _) in enumerate(ordered) if oi == index)
    if pos < len(ordered) - 1:
        a = ordered[pos][1]
        b = ordered[pos + 1][1]
        drop_orig = {ordered[pos][0], ordered[pos + 1][0]}
    else:
        a = ordered[pos - 1][1]
        b = ordered[pos][1]
        drop_orig = {ordered[pos - 1][0], ordered[pos][0]}
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    nx = min(ax, bx)
    ny = min(ay, by)
    nx2 = max(ax + aw, bx + bw)
    ny2 = max(ay + ah, by + bh)
    merged = (nx, ny, nx2 - nx, ny2 - ny)
    result = [boxes[i] for i in range(len(boxes)) if i not in drop_orig]
    result.append(merged)
    result.sort(key=lambda b: (b[0], b[1]))
    return result


def suggest_split_indices(boxes: list[Box], *, width_factor: float = 1.8) -> list[int]:
    """返回偏宽、建议拆粘连的框下标（相对当前 boxes 顺序）。"""
    if not boxes:
        return []
    widths = [b[2] for b in boxes]
    med = sorted(widths)[len(widths) // 2]
    thr = max(12, int(med * width_factor))
    return [i for i, b in enumerate(boxes) if b[2] >= thr]


def auto_fix_boxes(boxes: list[Box]) -> tuple[list[Box], list[int]]:
    """先合并碎框，再给出建议拆分的下标。"""
    fixed = merge_tiny_boxes(boxes)
    tips = suggest_split_indices(fixed)
    return fixed, tips


def is_oversized_box(
    box: Box,
    image_w: int,
    image_h: int,
    *,
    width_ratio: float = 0.72,
    area_ratio: float = 0.40,
) -> bool:
    """框是否大到像「整图/整行误选」（盖住后会挡住继续框选）。"""
    if image_w <= 0 or image_h <= 0:
        return False
    _x, _y, w, h = box
    if w <= 0 or h <= 0:
        return False
    if w >= int(image_w * width_ratio):
        return True
    return (w * h) >= int(image_w * image_h * area_ratio)


def filter_giant_auto_boxes(
    boxes: list[Box],
    image_w: int,
    image_h: int,
) -> list[Box]:
    """自动切字若只产出一个巨框，视为失败，返回空以免盖住画布。"""
    if len(boxes) == 1 and is_oversized_box(boxes[0], image_w, image_h):
        return []
    return [b for b in boxes if not is_oversized_box(b, image_w, image_h)] or list(boxes)
