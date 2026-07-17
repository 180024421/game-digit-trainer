"""把导出包拷到 auto-script-studio 工程 models/。"""
from __future__ import annotations

import shutil
from pathlib import Path

EXPORT_FILES = ("digits.onnx", "digits.labels", "manifest.json", "README.txt")

LUA_STUB = '''-- recognize_digits.lua — game-digit-trainer 生成
-- 推荐直接使用运行时 API：bot.recognizeDigits

local M = {}

function M.recognize(roi, model)
  local r = bot.recognizeDigits({
    roi = roi,
    model = model or "models/digits",
    min_confidence = 0.85,
    max_gap = 3,
  })
  return r.text, r.chars, r.confidence
end

return M
'''


def copy_exports_to_studio(exports_dir: Path, studio_models_dir: Path) -> list[str]:
    """复制导出文件到目标 models 目录，返回已拷文件名。"""
    exports_dir = Path(exports_dir)
    studio_models_dir = Path(studio_models_dir)
    studio_models_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in EXPORT_FILES:
        src = exports_dir / name
        if not src.is_file():
            continue
        shutil.copy2(src, studio_models_dir / name)
        copied.append(name)
    stub = studio_models_dir / "recognize_digits.lua"
    stub.write_text(LUA_STUB, encoding="utf-8")
    copied.append("recognize_digits.lua")
    if not copied or copied == ["recognize_digits.lua"]:
        raise FileNotFoundError(f"导出目录缺少 ONNX 包: {exports_dir}")
    return copied
