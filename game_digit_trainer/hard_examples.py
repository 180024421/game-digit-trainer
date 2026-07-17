"""难例队列：低置信确认、人工改标、金标对比失败。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game_digit_trainer.project import GameProject


def hard_dir(project: GameProject) -> Path:
    d = project.root / "hard"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hard_index_path(project: GameProject) -> Path:
    return hard_dir(project) / "index.json"


def load_hard_index(project: GameProject) -> list[dict[str, Any]]:
    path = hard_index_path(project)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_hard_index(project: GameProject, items: list[dict[str, Any]]) -> None:
    hard_index_path(project).write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_hard_example(
    project: GameProject,
    src: Path,
    *,
    reason: str,
    pred: str | None = None,
    conf: float | None = None,
    expected: str | None = None,
) -> Path | None:
    if not src.is_file():
        return None
    dest_dir = hard_dir(project)
    dest = dest_dir / src.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
        n += 1
    shutil.copy2(src, dest)
    items = load_hard_index(project)
    items.insert(
        0,
        {
            "file": dest.name,
            "reason": reason,
            "pred": pred,
            "conf": conf,
            "expected": expected,
            "at": datetime.now(timezone.utc).isoformat(),
        },
    )
    # keep last 500
    save_hard_index(project, items[:500])
    return dest


def list_hard_files(project: GameProject) -> list[Path]:
    d = hard_dir(project)
    return sorted(
        [p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def remove_hard_file(project: GameProject, path: Path) -> None:
    name = path.name
    path.unlink(missing_ok=True)
    items = [x for x in load_hard_index(project) if x.get("file") != name]
    save_hard_index(project, items)
