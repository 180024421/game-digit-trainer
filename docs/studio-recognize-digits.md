# Studio 接入草稿：recognizeDigits

> 供 [auto-script-studio](https://github.com/180024421/auto-script-studio) 二期接入参考。  
> 本仓库 v1 **不修改** Studio 运行时；导出包拷到工程 `models/` 即可。

## 导出包

```text
exports/
  digits.onnx
  digits.labels
  manifest.json
  README.txt
```

把整个 `exports/`（或其中三个文件）拷到脚本工程例如：

```text
your-script/models/digits.onnx
your-script/models/digits.labels
your-script/models/manifest.json
```

## 约定

- 输入：`NCHW` float32，单通道，像素 `/255` → `[0,1]`
- 尺寸：读 `manifest.json` 的 `input.width/height`（默认 32×32）
- 输出：logits，对类别维做 softmax，取 argmax → `digits.labels` 第 N 行

预处理应与训练站一致（灰度 / 反色 / 二值化），见 `manifest.preprocess`。

## Lua 伪代码（草稿）

```lua
-- 假设运行时提供：cropRoi / preprocessGray / onnxInfer / loadLabels
-- 实际 API 名以 Studio 二期为准

local function recognizeDigits(roi)
  local labels = loadLabels("models/digits.labels")
  local boxes = segmentChars(roi)  -- 或脚本侧已切好的字框列表
  local parts = {}
  for i, box in ipairs(boxes) do
    local gray = preprocessGray(cropRoi(roi, box), manifest.preprocess)
    local tensor = resizeNorm(gray, manifest.input.width, manifest.input.height)
    local logits = onnxInfer("models/digits.onnx", tensor)
    local idx, conf = argmaxSoftmax(logits)
    parts[#parts+1] = { label = labels[idx], conf = conf }
  end
  return joinDisplay(parts), parts
end

-- 例：读金币行
local text, parts = recognizeDigits(coinRoi)
log("金币=" .. text)
```

## 热更新

若 Studio 支持运行时按路径加载 ONNX：只需替换 `models/digits.onnx`（及 labels/manifest），**无需重打包 APK**。  
若不支持，则需把模型打进资源包后重装。

## 自检

训练站导出后会尝试 `onnxruntime` 跑一次 dummy（可选安装）。  
脚本侧建议在接入时对已知截图做一次对比测试。
