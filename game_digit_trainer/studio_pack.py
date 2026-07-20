"""把导出包拷到 auto-script-studio 工程 models/。"""
from __future__ import annotations

import shutil
from pathlib import Path

EXPORT_FILES = ("digits.onnx", "digits.labels", "manifest.json", "README.txt")
LINE_EXPORT_FILES = (
    "digits_line.onnx",
    "digits_line.labels",
    "manifest.json",
    "README.txt",
)

LUA_STUB = '''-- recognize_digits.lua — game-digit-trainer 生成
-- 单字包：切字+分类；行包：整 ROI 一次识别（manifest.kind=line_crnn）

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

-- 行模型示例：model = "models/line/digits_line"
function M.recognize_line(roi, model)
  return M.recognize(roi, model or "models/line/digits_line")
end

return M
'''


def copy_exports_to_studio(exports_dir: Path, studio_models_dir: Path) -> list[str]:
    """复制单字导出文件到目标 models 目录，返回已拷文件名。"""
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


def copy_line_exports_to_studio(line_exports_dir: Path, studio_models_dir: Path) -> list[str]:
    """复制行模型包到 models/line/。"""
    line_exports_dir = Path(line_exports_dir)
    dest = Path(studio_models_dir) / "line"
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in LINE_EXPORT_FILES:
        src = line_exports_dir / name
        if not src.is_file():
            continue
        shutil.copy2(src, dest / name)
        copied.append(f"line/{name}")
    stub = Path(studio_models_dir) / "recognize_digits.lua"
    stub.write_text(LUA_STUB, encoding="utf-8")
    if "recognize_digits.lua" not in copied:
        copied.append("recognize_digits.lua")
    if not any(n.endswith("digits_line.onnx") for n in copied):
        raise FileNotFoundError(f"行导出目录缺少 digits_line.onnx: {line_exports_dir}")
    return copied
