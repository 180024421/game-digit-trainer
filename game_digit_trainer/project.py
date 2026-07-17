from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game_digit_trainer.labels import DIGIT_CLASSES, build_class_list, UNIT_CLASS_NAMES


@dataclass
class PreprocessConfig:
    grayscale: bool = True
    invert: bool = False
    binarize: str = "otsu"  # otsu | none | adaptive
    color_filter: dict[str, Any] | None = None


@dataclass
class ProjectConfig:
    game_id: str
    classes: list[str] = field(default_factory=lambda: list(DIGIT_CLASSES))
    input_width: int = 32
    input_height: int = 32
    channels: int = 1
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    created_at: str = ""

    def validate(self) -> ProjectConfig:
        if not self.game_id.strip():
            raise ValueError("game_id 不能为空")
        if self.input_width < 8 or self.input_height < 8:
            raise ValueError("input size 过小")
        if not self.classes:
            raise ValueError("classes 不能为空")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        return self


class GameProject:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.config_path = self.root / "config.json"
        self.raw_dir = self.root / "images" / "raw"
        self.roi_dir = self.root / "images" / "roi"
        self.dataset_dir = self.root / "dataset"
        self.pending_dir = self.root / "pending"
        self.runs_dir = self.root / "runs"
        self.exports_dir = self.root / "exports"
        self._config: ProjectConfig | None = None

    @property
    def config(self) -> ProjectConfig:
        if self._config is None:
            self._config = load_config(self.config_path)
        return self._config

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.roi_dir,
            self.dataset_dir,
            self.pending_dir,
            self.runs_dir,
            self.exports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        for name in self.config.classes:
            (self.dataset_dir / name).mkdir(parents=True, exist_ok=True)

    def reload(self) -> ProjectConfig:
        self._config = load_config(self.config_path)
        return self.config

    def save_config(self) -> None:
        assert self._config is not None
        save_config(self.config_path, self._config)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.config.classes:
            folder = self.dataset_dir / name
            if not folder.is_dir():
                counts[name] = 0
                continue
            counts[name] = sum(
                1 for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
        return counts

    def pending_files(self) -> list[Path]:
        if not self.pending_dir.is_dir():
            return []
        return sorted(
            p
            for p in self.pending_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )


def projects_root(base: Path | None = None) -> Path:
    root = (base or Path.cwd()) / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_project(
    game_id: str,
    base: Path | None = None,
    *,
    with_symbols: bool = False,
    with_units: bool = False,
) -> GameProject:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in game_id.strip())
    if not safe:
        raise ValueError("无效 game_id")
    root = projects_root(base) / safe
    if root.exists() and (root / "config.json").exists():
        raise FileExistsError(f"项目已存在: {root}")
    classes = build_class_list(with_symbols=with_symbols, with_units=with_units)
    cfg = ProjectConfig(game_id=safe, classes=classes).validate()
    root.mkdir(parents=True, exist_ok=True)
    save_config(root / "config.json", cfg)
    proj = GameProject(root)
    proj._config = cfg
    proj.ensure_dirs()
    return proj


def ensure_unit_classes(project: GameProject) -> list[str]:
    """为已有项目追加 万/亿 类别（若尚未包含）。返回新加入的类名。"""
    added: list[str] = []
    cfg = project.config
    for name in UNIT_CLASS_NAMES:
        if name not in cfg.classes:
            cfg.classes.append(name)
            added.append(name)
    if added:
        project.save_config()
        project.ensure_dirs()
    return added


def open_project(path: Path) -> GameProject:
    root = path.resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"不是有效项目: {root}")
    proj = GameProject(root)
    proj.ensure_dirs()
    return proj


def load_config(path: Path) -> ProjectConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    prep_raw = data.get("preprocess") or {}
    prep = PreprocessConfig(
        grayscale=bool(prep_raw.get("grayscale", True)),
        invert=bool(prep_raw.get("invert", False)),
        binarize=str(prep_raw.get("binarize", "otsu")),
        color_filter=prep_raw.get("color_filter"),
    )
    return ProjectConfig(
        game_id=str(data["game_id"]),
        classes=list(data.get("classes") or DIGIT_CLASSES),
        input_width=int(data.get("input_width", 32)),
        input_height=int(data.get("input_height", 32)),
        channels=int(data.get("channels", 1)),
        preprocess=prep,
        created_at=str(data.get("created_at") or ""),
    ).validate()


def save_config(path: Path, cfg: ProjectConfig) -> None:
    payload = {
        "game_id": cfg.game_id,
        "classes": cfg.classes,
        "input_width": cfg.input_width,
        "input_height": cfg.input_height,
        "channels": cfg.channels,
        "preprocess": asdict(cfg.preprocess),
        "created_at": cfg.created_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
