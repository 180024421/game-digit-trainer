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

1. 顶部 **新建/打开** 项目（默认勾选万/亿）  
2. **① 截图切字**：`F2` 框选截屏 → 在图上再框数字 → **切字并去审核**  
3. **② 审核标注**：大数字键 / `W`万 `Y`亿 / 空格确认预测  
4. **③ 训练导出**：样本够了就训练并导出 ONNX  

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
