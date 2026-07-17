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
2. **① 截图切字**：`F2` / ADB / **雷电窗口** → 拖绿框 → **预览识别**（大号结果+框上叠字）  
3. 可填**金标**点「对比回流」；错字进难例；**多ROI刷样**批量切字  
4. **② 审核**：待审 / 已标注 / **难例**；空格批量确认  
5. **③ 训练导出**：看训练曲线 → 质量门禁 → 导出（ONNX 自检）→ 见 `docs/studio-recognize-digits.md`  

快捷键：`F2` 截屏 · `Enter` 切字 · `Ctrl+Z` 按页撤销 · 数字键标注  

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
