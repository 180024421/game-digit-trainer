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
2. **① 截图切字**：`F2` / ADB / **雷电窗口** → 蓝框/绿框 → **预览识别**  
   - **画框 / 拖图** 切换；空格临时拖图；方向键/`[` `]` 微调选中框  
   - **修碎框 / 拆粘连 / 合并框**；切字预设保存在「更多」  
   - 金标对比回流；**加入回归集**  
3. **② 审核**：**全部预标排序**、按置信度先标难例；空格批量 / **同类批量**  
4. **③ 训练导出**：补齐稀缺类 → 训练 → 导出 → **拷到 Studio models/** → **跑回归集**  

快捷键：`F2` 截屏 · `Enter` 切字 · `Ctrl+Z` 撤销 · 空格拖图/确认 · 数字键标注  

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
