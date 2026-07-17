"""单字样本与截图源框的对照元数据（供审核页原图高亮）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from game_digit_trainer.project import GameProject


def meta_path(project: GameProject) -> Path:
    return project.root / "sample_meta.json"


def load_meta(project: GameProject) -> dict[str, Any]:
    path = meta_path(project)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_meta(project: GameProject, data: dict[str, Any]) -> None:
    meta_path(project).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record_pending_batch(
    project: GameProject,
    source: Path,
    pending_paths: list[Path],
    boxes: list[tuple[int, int, int, int]],
) -> None:
    """切字后写入：每个 pending 文件对应全图字框与整行框。"""
    if not pending_paths or not boxes:
        return
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2 = [b[0] + b[2] for b in boxes]
    y2 = [b[1] + b[3] for b in boxes]
    line = (min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys))
    try:
        src_rel = str(source.resolve().relative_to(project.root.resolve()))
    except ValueError:
        src_rel = str(source.resolve())
    data = load_meta(project)
    for path, box in zip(pending_paths, boxes, strict=False):
        data[path.name] = {
            "source": src_rel,
            "box": list(box),
            "line_box": list(line),
        }
    save_meta(project, data)


def rename_meta_key(project: GameProject, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    data = load_meta(project)
    if old_name not in data:
        return
    data[new_name] = data.pop(old_name)
    save_meta(project, data)


def get_meta(project: GameProject, filename: str) -> dict[str, Any] | None:
    item = load_meta(project).get(filename)
    return item if isinstance(item, dict) else None


def resolve_source(project: GameProject, meta: dict[str, Any]) -> Path | None:
    raw = meta.get("source")
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = project.root / p
    return p if p.is_file() else None
