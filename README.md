# game-digit-trainer

游戏 HUD **多字体数字** 训练站：切字 → 审核修正 → 训练小 CNN → 导出 ONNX，供 [auto-script-studio](https://github.com/180024421/auto-script-studio) 后续接入。

> 仅供学习与研究。请遵守游戏服务条款与当地法律法规。

## 一键启动（Windows）

```powershell
cd E:\xiangmu\game-digit-trainer
.\一键启动.cmd
```

首次会自动创建 `.venv` 并安装依赖（含 PyTorch，可能较慢）。也可先运行 `.\安装依赖.cmd`。

## 工作流

1. **项目**：新建游戏 ID（可勾选启用 `,` `/` `%` `:`）
2. **导入切字**：选 HUD 截图 → 切字加入待审核
3. **审核修正**：看大图，按 `0-9` 或点按钮改标签（多字体混在同一类）
4. **训练导出**：训练 → 导出 `exports/digits.onnx` + `digits.labels` + `manifest.json`

把导出文件拷到 Studio 工程 `models/` 即可（运行时 API 为 v2，见设计规格）。

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
