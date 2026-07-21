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


def load_char_pools(
    project: GameProject,
    *,
    include_bootstrap: bool = False,
) -> dict[str, list[np.ndarray]]:
    """加载单字图池。默认排除 from_line_* 粗切（易脏，勿用于合成行）。

    include_bootstrap=True 时才混入粗切图（仅「自动粗切合成」开启时）。
    """
    pools: dict[str, list[np.ndarray]] = {}
    for name in project.config.classes:
        folder = project.dataset_dir / name
        if not folder.is_dir():
            continue
        imgs: list[np.ndarray] = []
        for p in folder.iterdir():
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            if not include_bootstrap and p.name.startswith("from_line_"):
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
    pred: str = "",
    conf: float = 0.0,
    hint: str = "",
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
    if pred or hint or conf:
        set_line_pending_hint(project, out.name, pred=pred, conf=conf, hint=hint)
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
    clear_line_pending_hint(project, pending.name)
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


def _split_widths_by_ink(gray: np.ndarray, n: int) -> list[tuple[int, int]]:
    """按墨迹投影粗分 n 段，失败则等宽。返回 [(x0,x1), ...]。"""
    h, w = gray.shape[:2]
    if n <= 0 or w < n * 2:
        return []
    g = gray
    if float(np.mean(g)) > 127:
        g = 255 - g
    col = (g > 30).sum(axis=0).astype(np.float32)
    if float(col.sum()) < 1:
        # 等宽
        step = w / n
        return [(int(i * step), int((i + 1) * step)) for i in range(n)]
    # 累积墨迹分位数切分
    cdf = np.cumsum(col)
    total = float(cdf[-1])
    cuts = [0]
    for i in range(1, n):
        target = total * i / n
        x = int(np.searchsorted(cdf, target))
        cuts.append(max(cuts[-1] + 1, min(w - (n - i), x)))
    cuts.append(w)
    return [(cuts[i], cuts[i + 1]) for i in range(n)]


def bootstrap_chars_from_line_samples(
    project: GameProject,
    *,
    max_per_class: int = 80,
) -> tuple[dict[str, int], list[tuple[str, Path, str]]]:
    """
    从已标行样本按金标粗切单字写入 dataset/，便于合成更多数字组合。

    返回 (各类新增数量, 预览列表[(类名, 图片路径, 来源行金标)])。
    """
    classes = list(project.config.classes)
    added: dict[str, int] = {c: 0 for c in classes}
    previews: list[tuple[str, Path, str]] = []
    project.ensure_dirs()
    for path, text in list_line_labeled(project):
        if not text.strip():
            continue
        try:
            tokens = tokenize_expected(text, classes)
        except Exception:
            continue
        if not tokens:
            continue
        raw = np.fromfile(str(path), dtype=np.uint8)
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            continue
        spans = _split_widths_by_ink(gray, len(tokens))
        if len(spans) != len(tokens):
            continue
        _h, w = gray.shape[:2]
        for tok, (x0, x1) in zip(tokens, spans):
            if added.get(tok, 0) >= max_per_class:
                continue
            x0 = max(0, min(w - 1, x0))
            x1 = max(x0 + 1, min(w, x1))
            pad = max(1, (x1 - x0) // 10)
            xa = max(0, x0 - pad)
            xb = min(w, x1 + pad)
            crop = gray[:, xa:xb]
            if crop.shape[1] < 2:
                continue
            dest_dir = project.dataset_dir / tok
            dest_dir.mkdir(parents=True, exist_ok=True)
            out = dest_dir / f"from_line_{path.stem}_{added[tok]:03d}.png"
            while out.exists():
                added[tok] += 1
                if added[tok] >= max_per_class:
                    break
                out = dest_dir / f"from_line_{path.stem}_{added[tok]:03d}.png"
            if added[tok] >= max_per_class:
                continue
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                continue
            out.write_bytes(buf.tobytes())
            added[tok] = added.get(tok, 0) + 1
            if len(previews) < 48:
                previews.append((tok, out, text))
    return {k: v for k, v in added.items() if v > 0}, previews


def clear_bootstrap_chars(project: GameProject) -> int:
    """删除 dataset/ 下 from_line_* 粗切单字，返回删除文件数。"""
    n = 0
    root = project.dataset_dir
    if not root.is_dir():
        return 0
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        for p in folder.glob("from_line_*.png"):
            p.unlink(missing_ok=True)
            n += 1
        for p in folder.glob("from_line_*.jpg"):
            p.unlink(missing_ok=True)
            n += 1
    return n


def line_pending_hints_path(project: GameProject) -> Path:
    return lines_dir(project) / "pending_hints.json"


def load_line_pending_hints(project: GameProject) -> dict[str, dict]:
    path = line_pending_hints_path(project)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_line_pending_hints(project: GameProject, hints: dict[str, dict]) -> None:
    path = line_pending_hints_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hints, ensure_ascii=False, indent=2), encoding="utf-8")


def set_line_pending_hint(
    project: GameProject,
    filename: str,
    *,
    pred: str = "",
    conf: float = 0.0,
    hint: str = "",
) -> None:
    hints = load_line_pending_hints(project)
    hints[filename] = {
        "pred": pred or "",
        "conf": float(conf),
        "hint": hint or "",
    }
    save_line_pending_hints(project, hints)


def get_line_pending_hint(project: GameProject, filename: str) -> dict:
    return dict(load_line_pending_hints(project).get(filename) or {})


def clear_line_pending_hint(project: GameProject, filename: str) -> None:
    hints = load_line_pending_hints(project)
    if filename in hints:
        del hints[filename]
        save_line_pending_hints(project, hints)


def line_coverage_report(project: GameProject) -> dict[str, object]:
    """行样本形态覆盖：长度 / 小数 / 单位 / 纯数字。"""
    items = list_line_labeled(project)
    classes = list(project.config.classes)
    lengths: dict[int, int] = {}
    with_dot = 0
    with_unit = 0
    plain = 0
    for _path, text in items:
        if not text.strip():
            continue
        try:
            tokens = tokenize_expected(text, classes)
        except Exception:
            continue
        n = len(tokens)
        lengths[n] = lengths.get(n, 0) + 1
        if "dot" in tokens:
            with_dot += 1
        if any(t in {"wan", "yi"} for t in tokens):
            with_unit += 1
        if all(t.isdigit() for t in tokens):
            plain += 1
    return {
        "total": len(items),
        "lengths": dict(sorted(lengths.items())),
        "with_dot": with_dot,
        "with_unit": with_unit,
        "plain": plain,
    }


def coverage_fill_suggestions(
    project: GameProject,
    *,
    target_total: int = 30,
    min_dot: int = 3,
    min_unit: int = 5,
    min_plain: int = 5,
) -> list[str]:
    """根据覆盖度给出可执行补样建议。"""
    cov = line_coverage_report(project)
    total = int(cov.get("total") or 0)
    tips: list[str] = []
    if total < target_total:
        tips.append(f"还差约 {target_total - total} 条整行样本（目标 ≥{target_total}）")
    if int(cov.get("with_dot") or 0) < min_dot:
        tips.append(f"补 {min_dot - int(cov.get('with_dot') or 0)} 条带小数的（如 1.9 / 12.5万）")
    if int(cov.get("with_unit") or 0) < min_unit:
        tips.append(f"补 {min_unit - int(cov.get('with_unit') or 0)} 条带「万/亿」的")
    if int(cov.get("plain") or 0) < min_plain:
        tips.append(f"补 {min_plain - int(cov.get('plain') or 0)} 条纯数字（如 6370）")
    lengths = cov.get("lengths") or {}
    if total >= 8 and not any(int(k) >= 4 for k in lengths):
        tips.append("补几条 4 位以上长数字（防短串过拟合）")
    if not tips and total > 0:
        tips.append("形态覆盖尚可，可继续用同 HUD 刷几条再续训")
    return tips


def suggest_line_max_width(samples: list[tuple[np.ndarray, list[int]]]) -> int:
    """按样本实际行宽建议 pad 宽度，短数字少算空白。"""
    if not samples:
        return LINE_MAX_WIDTH
    widths: list[int] = []
    for img, _ in samples:
        g = img
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        if float(np.mean(g)) > 127:
            g = 255 - g
        g = _resize_h(g, LINE_HEIGHT)
        widths.append(int(g.shape[1]))
    need = max(widths) + 16
    # 对齐到 8，便于 CNN 下采样
    need = int((need + 7) // 8 * 8)
    return int(min(LINE_MAX_WIDTH, max(64, need)))


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
