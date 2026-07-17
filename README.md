# game-digit-trainer

游戏 HUD **多字体数字** 训练站：切字 → 审核修正 → 训练小 CNN → 导出 ONNX，供 [auto-script-studio](https://github.com/180024421/auto-script-studio) 后续接入。

> 仅供学习与研究。请遵守游戏服务条款与当地法律法规。

## 一键启动（Windows）

```powershell
cd E:\xiangmu\game-digit-trainer
.\一键启动.cmd
```

首次会自动创建 `.venv` 并安装依赖（含 PyTorch，可能较慢）。也可先运行 `.\安装依赖.cmd`。

## 推荐操作

1. 顶部选**模板**后 **新建/打开** 项目  
2. **① 截图切字**：主路径截图→切字→预览→Enter；进阶在「更多」（修碎框/定时刷样/颜色过滤/切字预设）  
3. **② 审核**：预标排序 + 同类批量  
4. **③ 训练导出**：补齐稀缺类 → 训练（自动写混淆矩阵）→ 导出 → **拷到 Studio models/**  
5. Studio：`bot.recognizeDigits({ roi=... })` 读 HUD 数字（见 `docs/studio-recognize-digits.md`）  

快捷键：`F2` · `Enter` · `Ctrl+Z` / `Ctrl+Y` · 空格拖图/确认  

## CLI

```powershell
.\.venv\Scripts\python -m game_digit_trainer create mygame --symbols
.\.venv\Scripts\python -m game_digit_trainer segment --project projects\mygame --image hud.png
.\.venv\Scripts\python -m game_digit_trainer train --project projects\mygame
.\.venv\Scripts\python -m game_digit_trainer export --project projects\mygame
```

## 文档

- 设计：`docs/superpowers/specs/2026-07-17-game-digit-trainer-design.md`
- Agent：`AGENTS.md`

## 许可

MIT
