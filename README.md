# game-digit-trainer

游戏 HUD **多字体数字** 训练站：采标 → 训练 → 导出 ONNX，供 [auto-script-studio](https://github.com/180024421/auto-script-studio) 脚本识别。

**主路径（推荐）**：整行蓝框 → 行待审 → **行模型 CRNN** → 蓝框一次出整串。  
**兼容路径**：逐字切字 → 单字 CNN → `digits.onnx`（旧 Studio 包）。

> 仅供学习与研究。请遵守游戏服务条款与当地法律法规。

## 一键启动（Windows）

```powershell
cd E:\xiangmu\game-digit-trainer
.\一键启动.cmd
```

首次会自动创建 `.venv` 并安装依赖（含 PyTorch，可能较慢）。也可先运行 `.\安装依赖.cmd`。

## 推荐操作（行模型）

1. 顶部 **新建/打开** 项目  
2. **①**：框选截屏 F2 → **整行蓝框** 圈数字 → **加入行待审**（约 30～50 条更好）  
3. **②「行待审」**：填整串金标确认  
4. **③**：**训练行模型** → **导出行 ONNX** → **拷行模型到 Studio**  
5. Studio：`bot.recognizeDigits({ roi=..., model="models/line/digits_line" })`  

换机/备份标注：③ 页 **导出标注包** / **导入标注包**（单字 + 行样本/待审 + 难例；不含模型）。也可「备份项目」打整包。

单字路径仍保留（确认切字 / 训练单字 / 导出单字 ONNX），见 `docs/studio-recognize-digits.md`。

快捷键：`F2` · `Enter` · `Ctrl+Z` / `Ctrl+Y` · 空格拖图/确认  

## CLI

```powershell
.\.venv\Scripts\python -m game_digit_trainer create mygame --symbols
.\.venv\Scripts\python -m game_digit_trainer segment --project projects\mygame --image hud.png
.\.venv\Scripts\python -m game_digit_trainer train --project projects\mygame
.\.venv\Scripts\python -m game_digit_trainer export --project projects\mygame
```

## 文档

- Studio 接入：`docs/studio-recognize-digits.md`
- 行模型设计：`docs/superpowers/specs/2026-07-20-line-crnn-design.md`
- 主设计：`docs/superpowers/specs/2026-07-17-game-digit-trainer-design.md`
- Agent：`AGENTS.md`

## 许可

MIT
