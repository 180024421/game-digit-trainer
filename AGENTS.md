# AGENTS.md — game-digit-trainer

游戏 HUD 多字体数字训练站（切字 / 修正 / 训练 / ONNX 导出）。

## 启动

```powershell
.\一键启动.cmd
# 或
.\安装依赖.cmd
python -m game_digit_trainer gui
```

## 范围

- v1：训练站 + 标准导出包（`digits.onnx` / `digits.labels` / `manifest.json`）
- 不改 auto-script-studio 运行时（见设计规格）

## 文档

- 设计：`docs/superpowers/specs/2026-07-17-game-digit-trainer-design.md`
