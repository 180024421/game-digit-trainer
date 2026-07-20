# 行优先工作流 + 易用性全量落地 — 实现计划

日期：2026-07-20

## 目标

收敛产品主路径为「行模型」，补齐导出/Studio 契约，并完成易用性改造。

## 文件

| 文件 | 职责 |
|------|------|
| `game_digit_trainer/export_line_onnx.py` | 行 CRNN → `exports/line/` |
| `game_digit_trainer/studio_pack.py` | 支持拷贝行包 |
| `game_digit_trainer/gui/app.py` | 默认行路径、UI 精简、停止训练、回归、提示 |
| `game_digit_trainer/train_line.py` | 可取消训练 |
| `docs/studio-recognize-digits.md` / README | 双模型说明 |
| `auto-script-studio/.../vision_pc.py` + DigitRecognizer | manifest.kind=line_crnn 整行推理 |

## 任务

1. 默认整行蓝框；文案统一；引导改行优先
2. export_line_onnx + GUI 导出/拷贝
3. Studio PC + Android 识别行 ONNX
4. 验模型默认折叠；预览识别合并入口文案
5. 信息类改 statusBar；训练停止；行回归；样本数建议；低置信进行待审提示

不拆 app.py（风险大），行为改完即可。
