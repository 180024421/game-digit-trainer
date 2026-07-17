"""项目备份 / 迁移 zip。"""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

from game_digit_trainer.project import GameProject

INCLUDE_DIRS = ("dataset", "config.json", "exports", "regression", "hard", "pending")


def backup_project(project: GameProject, dest: Path | None = None) -> Path:
    """打包 dataset/config/exports/regression/hard/pending 为 zip。"""
    root = project.root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dest or (root / "backups" / f"{project.config.game_id}_{stamp}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        cfg = root / "config.json"
        if cfg.is_file():
            zf.write(cfg, "config.json")
        for name in ("dataset", "exports", "regression", "hard", "pending"):
            folder = root / name
            if not folder.exists():
                continue
            if folder.is_file():
                zf.write(folder, name)
                continue
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())
    return out
