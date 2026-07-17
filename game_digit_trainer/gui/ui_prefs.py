"""本机 UI 偏好（窗口、模式、引导等），与项目配置分离。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prefs_path() -> Path:
    root = Path.home() / ".game_digit_trainer"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ui_prefs.json"


def load_prefs() -> dict[str, Any]:
    path = prefs_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_prefs(data: dict[str, Any]) -> None:
    path = prefs_path()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_prefs(**kwargs: Any) -> dict[str, Any]:
    data = load_prefs()
    data.update({k: v for k, v in kwargs.items() if v is not None})
    save_prefs(data)
    return data
