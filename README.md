# game-digit-trainer

游戏 HUD **多字体数字** 训练站：切字 → 审核修正 → 训练小 CNN → 导出 ONNX，供 [auto-script-studio](https://github.com/180024421/auto-script-studio) 后续接入。

> 仅供学习与研究。请遵守游戏服务条款与当地法律法规。

## 一键启动（Windows）

```powershell
cd E:\xiangmu\game-digit-trainer
.\一键启动.cmd
```

首次会自动创建 `.venv` 并安装依赖（含 PyTorch，可能较慢）。也可先运行 `.\安装依赖.cmd`。

## 推荐操作（好上手）

1. **项目**：新建游戏 ID（勾选「单位 万/亿」；已有项目可点「加万/亿」）  
2. **导入切字**：
   - **框选截屏**（推荐）/ **ADB截图**（雷电）/ Win+Shift+S 后 **粘贴**
   - 在图上再框选金币数字区域 → 调二值化直到绿框对准 → 切字  
3. **审核修正**：`0-9`；`W`=万、`Y`=亿；空格确认预测  
4. **数据集**：改错类、删脏样本  
5. **训练导出**：训练 → 导出 `exports/` 下的 onnx 包

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
