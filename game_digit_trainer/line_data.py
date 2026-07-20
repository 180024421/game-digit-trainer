"""行样本：从单字目录合成 + 读取真实行标注。"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.gold import tokenize_expected
from game_digit_trainer.labels import display_label
from game_digit_trainer.project import GameProject
from game_digit_trainer.segment import crop_bgr


LINE_HEIGHT = 32
LINE_MAX_WIDTH = 256


def lines_dir(project: GameProject) -> Path:
    d = project.root / "lines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lines_pending_dir(project: GameProject) -> Path:
    d = lines_dir(project) / "pending"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lines_labels_path(project: GameProject) -> Path:
    return lines_dir(project) / "labels.jsonl"


def list_line_pending(project: GameProject) -> list[Path]:
    d = lines_pending_dir(project)
    return sorted(
        p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )


def load_char_pools(project: GameProject) -> dict[str, list[np.ndarray]]:
    pools: dict[str, list[np.ndarray]] = {}
    for name in project.config.classes:
        folder = project.dataset_dir / name
        if not folder.is_dir():
            continue
        imgs: list[np.ndarray] = []
        for p in folder.iterdir():
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            raw = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size:
                imgs.append(img)
        if imgs:
            pools[name] = imgs
    return pools


def _resize_h(img: np.ndarray, height: int = LINE_HEIGHT) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((height, 8), dtype=np.uint8)
    nw = max(2, int(round(w * (height / h))))
    return cv2.resize(img, (nw, height), interpolation=cv2.INTER_AREA)


def synthesize_line(
    pools: dict[str, list[np.ndarray]],
    classes: list[str],
    *,
    min_len: int = 2,
    max_len: int = 6,
    gap: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """合成一行。优先生成游戏 HUD 常见形态：数字 + 可选小数点 + 万/亿。"""
    digit_keys = [c for c in classes if c.isdigit() and c in pools and pools[c]]
    unit_keys = [c for c in ("wan", "yi") if c in pools and pools[c]]
    symbol_keys = [c for c in ("dot", "comma", "colon", "percent", "slash") if c in pools and pools[c]]
    all_keys = [c for c in classes if c in pools and pools[c]]
    if not all_keys:
        raise ValueError("单字样本为空，无法合成行图")

    chosen: list[str] = []
    # 70%：HUD 数值串；30%：随机串（增强泛化）
    if digit_keys and random.random() < 0.7:
        n_digit = random.randint(1, 4)
        chosen.extend(random.choice(digit_keys) for _ in range(n_digit))
        if "dot" in symbol_keys and random.random() < 0.35 and len(chosen) >= 1:
            # 插入小数：如 1.9 / 2.2
            pos = random.randint(1, len(chosen))
            chosen.insert(pos, "dot")
            chosen.insert(pos + 1, random.choice(digit_keys))
        if unit_keys and random.random() < 0.75:
            chosen.append(random.choice(unit_keys))
    else:
        n = random.randint(min_len, max_len)
        chosen = [random.choice(all_keys) for _ in range(n)]

    while len(chosen) < min_len and digit_keys:
        chosen.insert(0, random.choice(digit_keys))
    if len(chosen) < min_len:
        chosen.extend(random.choice(all_keys) for _ in range(min_len - len(chosen)))

    gap_px = gap if gap is not None else random.randint(1, 4)
    patches = [_resize_h(random.choice(pools[c])) for c in chosen]
    normed: list[np.ndarray] = []
    for p in patches:
        g = p
        if float(np.mean(g)) > 127:
            g = 255 - g
        normed.append(g)
    gap_img = np.zeros((LINE_HEIGHT, max(1, gap_px)), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, p in enumerate(normed):
        if i:
            parts.append(gap_img)
        parts.append(p)
    line = np.concatenate(parts, axis=1)
    indices = [classes.index(c) for c in chosen]
    return line, indices


def pad_line(img: np.ndarray, max_w: int = LINE_MAX_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if h != LINE_HEIGHT:
        img = _resize_h(img, LINE_HEIGHT)
        h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, LINE_HEIGHT), interpolation=cv2.INTER_AREA)
        w = max_w
    canvas = np.zeros((LINE_HEIGHT, max_w), dtype=np.uint8)
    canvas[:, :w] = img
    return canvas


def prepare_line_tensor(gray: np.ndarray, max_w: int = LINE_MAX_WIDTH) -> tuple[np.ndarray, int]:
    """返回 (1,H,W) float32 0~1，以及有效宽度（未 pad 前缩放后的宽，至少 8）。"""
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if float(np.mean(g)) > 127:
        g = 255 - g
    g = _resize_h(g, LINE_HEIGHT)
    valid_w = int(g.shape[1])
    padded = pad_line(g, max_w=max_w)
    arr = padded.astype(np.float32) / 255.0
    return arr[None, ...], max(8, min(valid_w, max_w))


def load_real_line_samples(project: GameProject) -> list[tuple[np.ndarray, list[int]]]:
    path = lines_labels_path(project)
    if not path.is_file():
        return []
    classes = list(project.config.classes)
    out: list[tuple[np.ndarray, list[int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(item.get("image") or "")
        text = str(item.get("text") or "").strip()
        if not name or not text:
            continue
        img_path = lines_dir(project) / name
        if not img_path.is_file():
            continue
        raw = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        try:
            tokens = tokenize_expected(text, classes)
            indices = [classes.index(t) for t in tokens]
        except Exception:
            continue
        if not indices:
            continue
        out.append((img, indices))
    return out


def save_line_sample(
    project: GameProject,
    bgr: np.ndarray,
    region: tuple[int, int, int, int] | None,
    text: str,
) -> Path:
    """保存 ROI 行图 + 金标，供行模型训练。"""
    classes = list(project.config.classes)
    tokens = tokenize_expected(text, classes)
    display = "".join(display_label(t) for t in tokens)
    cropped = crop_bgr(bgr, region)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if cropped.ndim == 3 else cropped
    dest_dir = lines_dir(project)
    stem = f"line_{len(list(dest_dir.glob('*.png'))):04d}"
    out = dest_dir / f"{stem}.png"
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("无法编码行图")
    out.write_bytes(buf.tobytes())
    with lines_labels_path(project).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"image": out.name, "text": display}, ensure_ascii=False) + "\n")
    return out


def save_line_pending(
    project: GameProject,
    bgr: np.ndarray,
    region: tuple[int, int, int, int] | None,
    *,
    source_name: str = "roi",
) -> Path:
    """框选区域裁成行图，进入行待审（先不填字）。"""
    if region is None:
        raise ValueError("请先用「整行蓝框」圈住数字行")
    x, y, w, h = region
    if w < 8 or h < 4:
        raise ValueError("蓝框太小，请框紧一行数字")
    cropped = crop_bgr(bgr, region)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if cropped.ndim == 3 else cropped
    dest_dir = lines_pending_dir(project)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(source_name).stem)[:40]
    n = len(list(dest_dir.glob("*.png")))
    out = dest_dir / f"{safe}_{n:04d}.png"
    while out.exists():
        n += 1
        out = dest_dir / f"{safe}_{n:04d}.png"
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("无法编码行图")
    out.write_bytes(buf.tobytes())
    return out


def confirm_line_pending(project: GameProject, pending: Path, text: str) -> Path:
    """行待审填好金标后，移入 lines/ 并写入 labels.jsonl。"""
    classes = list(project.config.classes)
    tokens = tokenize_expected(text, classes)
    display = "".join(display_label(t) for t in tokens)
    if not pending.is_file():
        raise FileNotFoundError(f"待审行图不存在: {pending}")
    dest_dir = lines_dir(project)
    dest = dest_dir / pending.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{pending.stem}_{n}{pending.suffix}"
        n += 1
    dest.write_bytes(pending.read_bytes())
    pending.unlink(missing_ok=True)
    with lines_labels_path(project).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"image": dest.name, "text": display}, ensure_ascii=False) + "\n")
    return dest


# GUI 样本库用的伪类名（与单字目录区分）
LINE_DATASET_KEY = "__line__"


def _read_labels_map(project: GameProject) -> dict[str, str]:
    path = lines_labels_path(project)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(item.get("image") or "")
        text = str(item.get("text") or "").strip()
        if name and text:
            out[name] = text
    return out


def _write_labels_map(project: GameProject, mapping: dict[str, str]) -> None:
    path = lines_labels_path(project)
    lines = [
        json.dumps({"image": name, "text": text}, ensure_ascii=False)
        for name, text in sorted(mapping.items())
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def list_line_labeled(project: GameProject) -> list[tuple[Path, str]]:
    """已确认的行训练样本：(图片路径, 金标文字)。"""
    mapping = _read_labels_map(project)
    root = lines_dir(project)
    out: list[tuple[Path, str]] = []
    for name, text in sorted(mapping.items()):
        p = root / name
        if p.is_file():
            out.append((p, text))
    # 补孤儿图（有文件无 jsonl）
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        if p.name in mapping:
            continue
        out.append((p, ""))
    return out


def count_line_labeled(project: GameProject) -> int:
    return len(list_line_labeled(project))


def update_line_label(project: GameProject, image: Path, text: str) -> str:
    """改已标行样本的金标，返回规范化显示串。"""
    classes = list(project.config.classes)
    tokens = tokenize_expected(text, classes)
    display = "".join(display_label(t) for t in tokens)
    if not display:
        raise ValueError("金标为空或无法识别类别")
    mapping = _read_labels_map(project)
    if image.name not in mapping and not image.is_file():
        raise FileNotFoundError(f"行样本不存在: {image}")
    mapping[image.name] = display
    _write_labels_map(project, mapping)
    return display


def delete_line_sample(project: GameProject, image: Path) -> None:
    mapping = _read_labels_map(project)
    mapping.pop(image.name, None)
    _write_labels_map(project, mapping)
    image.unlink(missing_ok=True)


def clear_line_samples(project: GameProject) -> int:
    """清空全部已标行样本（不动 pending）。返回删除数量。"""
    items = list_line_labeled(project)
    for path, _ in items:
        path.unlink(missing_ok=True)
    path = lines_labels_path(project)
    if path.is_file():
        path.write_text("", encoding="utf-8")
    return len(items)


def ctc_greedy_decode(logits_t_n_c: np.ndarray, blank_index: int) -> list[int]:
    """logits: T×C → class indices（已去 blank / 重复）。"""
    pred = logits_t_n_c.argmax(axis=1)
    out: list[int] = []
    prev = None
    for p in pred.tolist():
        if p == blank_index:
            prev = p
            continue
        if p != prev:
            out.append(int(p))
        prev = p
    return out
