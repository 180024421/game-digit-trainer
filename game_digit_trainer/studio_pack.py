"""把导出包拷到 auto-script-studio 工程 models/。"""
from __future__ import annotations

import shutil
from pathlib import Path

EXPORT_FILES = ("digits.onnx", "digits.labels", "manifest.json", "README.txt")

LUA_STUB = '''-- recognize_digits.lua — 由 game-digit-trainer 生成的接入草稿
-- 放到脚本工程后，按 Studio 二期 API 微调函数名即可

local M = {}

local function load_manifest(dir)
  -- TODO: 读 models/manifest.json
  return { width = 32, height = 32 }
end

function M.recognizeDigits(roi, modelsDir)
  modelsDir = modelsDir or "models"
  local manifest = load_manifest(modelsDir)
  local labels = loadLabels(modelsDir .. "/digits.labels")
  local boxes = segmentChars(roi)
  local parts = {}
  for i, box in ipairs(boxes) do
    local gray = preprocessGray(cropRoi(roi, box), manifest.preprocess)
    local tensor = resizeNorm(gray, manifest.width, manifest.height)
    local logits = onnxInfer(modelsDir .. "/digits.onnx", tensor)
    local idx, conf = argmaxSoftmax(logits)
    parts[#parts + 1] = { label = labels[idx], conf = conf }
  end
  local text = ""
  for _, p in ipairs(parts) do
    text = text .. tostring(p.label)
  end
  return text, parts
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
