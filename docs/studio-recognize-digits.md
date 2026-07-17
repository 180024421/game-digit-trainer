# Studio 接入草稿：recognizeDigits

> 供 [auto-script-studio](https://github.com/180024421/auto-script-studio) 二期接入参考。  
> 本仓库 v1 **不修改** Studio 运行时；导出包拷到工程 `models/` 即可。

## 一键拷贝

训练站「③ 训练导出」→ **拷到 Studio models/**：选择脚本工程根目录，会写入：

```text
your-script/models/
  digits.onnx
  digits.labels
  manifest.json
  README.txt
  recognize_digits.lua   # 接入草稿
```

也可手动把 `projects/<game>/exports/` 整目录拷过去。

## 导出包

```text
exports/
  digits.onnx
  digits.labels
  manifest.json
  README.txt
```

## 约定

- 输入：`NCHW` float32，单通道，像素 `/255` → `[0,1]`
- 尺寸：读 `manifest.json` 的 `input.width/height`（默认 32×32）
- 输出：logits，对类别维做 softmax，取 argmax → `digits.labels` 第 N 行

预处理应与训练站一致（灰度 / 反色 / 二值化），见 `manifest.preprocess`。

## 回归自检

1. 切字页填金标 → **加入回归集**（存到 `projects/<game>/regression/`）
2. 导出/训练后点 **跑回归集**，对照期望字符串

## Lua

见拷贝后的 `models/recognize_digits.lua`。核心：

```lua
local text, parts = recognizeDigits(coinRoi, "models")
log("金币=" .. text)
```

（`segmentChars` / `onnxInfer` 等 API 以 Studio 二期为准。）

## 热更新

若 Studio 支持运行时按路径加载 ONNX：只需替换 `models/digits.onnx`（及 labels/manifest），**无需重打包 APK**。

## 切字说明

训练站侧以投影切字 + **修碎框 / 拆粘连 / 合并框** 为主；未引入 YOLO（保持包体小）。粘连严重时优先调间距/切字预设与人工拆合。
