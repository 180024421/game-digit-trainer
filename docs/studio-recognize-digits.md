# Studio 接入：recognizeDigits

> [auto-script-studio](https://github.com/180024421/auto-script-studio) 已支持 `bot.recognizeDigits`（APK + PC 联调）。

## 一键拷贝

训练站「③ 训练导出」→ **拷到 Studio models/**：

```text
your-script/models/
  digits.onnx
  digits.labels
  manifest.json
  README.txt
  recognize_digits.lua
```

`project.json` 可设：

```json
"runtime": {
  "default_digit_model": "models/digits"
}
```

## Lua

```lua
local r = bot.recognizeDigits({
  roi = {100, 200, 180, 40},
  model = "models/digits",
  min_confidence = 0.85,
  max_gap = 3,
})
bot.log("值=" .. r.text)
-- r.confidence, r.chars[i]
```

## 约定

- 输入：`NCHW` float32，单通道 `/255`
- 尺寸：`manifest.json` → `input.width/height`（默认 32×32）
- 切字：投影 + `max_gap`（与训练站间距同语义）

## 回归自检

切字页金标 → **加入回归集** → 训练页 **跑回归集**。

## 热更新

替换 `models/digits.onnx`（及 labels/manifest）即可，无需改脚本逻辑。
