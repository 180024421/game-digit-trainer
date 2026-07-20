# Studio 接入：recognizeDigits

> [auto-script-studio](https://github.com/180024421/auto-script-studio) 已支持 `bot.recognizeDigits`（APK + PC 联调）。

## 两种模型包

| 类型 | 目录 | 识别方式 | 推荐场景 |
|------|------|----------|----------|
| **行模型（推荐）** | `models/line/` | 整 ROI 一次 CRNN，不切字 | 游戏 HUD 数字串（如 `3920万`） |
| 单字模型 | `models/` | 投影切字 + 逐字分类 | 兼容旧脚本 / 细粒度纠错 |

## 一键拷贝

训练站 ③：

- **导出行 ONNX** → **拷行模型到 Studio** → `your-script/models/line/`
- **导出单字 ONNX** → **拷单字到 Studio** → `your-script/models/`

行包文件：

```text
your-script/models/line/
  digits_line.onnx
  digits_line.labels
  manifest.json          # kind=line_crnn
  README.txt
```

`project.json` 示例：

```json
"runtime": {
  "default_digit_model": "models/line/digits_line"
}
```

## Lua

```lua
-- 行模型（推荐）
local r = bot.recognizeDigits({
  roi = {100, 200, 180, 40},
  model = "models/line/digits_line",
  min_confidence = 0.85,
})
bot.log("值=" .. r.text)

-- 单字模型（兼容）
local r2 = bot.recognizeDigits({
  roi = {100, 200, 180, 40},
  model = "models/digits",
  min_confidence = 0.85,
  max_gap = 3,
})
```

## 约定

### 行模型 `line_crnn`

- 输入：`NCHW` float32，高固定（默认 32），宽 pad 到 `max_width`（默认 256），`/255`
- 输出：时间步 logits `(T, N, C)`，C = 类别数 + blank；CTC 贪心解码
- `manifest.kind` = `line_crnn`，含 `blank_index`

### 单字模型

- 输入：`NCHW` float32，单通道 `/255`，尺寸见 `manifest.input`
- 切字：投影 + `max_gap`（与训练站间距同语义）

## 回归自检

切字页金标 → **加入回归集**（会记下蓝框 ROI）→ 训练页 **跑回归集**（优先行模型）。

## 热更新

替换对应 `*.onnx`（及 labels/manifest）即可，无需改脚本逻辑。
