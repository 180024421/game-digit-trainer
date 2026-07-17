# game-digit-trainer 设计规格

日期：2026-07-17  
状态：待用户确认  
方案：方案 1 — PyQt 桌面 GUI + PyTorch 小 CNN → ONNX 导出包  
确认项：独立仓库；桌面 GUI；v1 仅训练站 + 标准导出（不改 auto-script-studio 运行时）

## 1. 目标

为**单个联网/单机 Unity 等游戏**建立「多字体数字」专用小分类器：从截图切字、人工修正误识、训练、导出可被 auto-script-studio 后续接入的标准模型包。

解决通用 OCR 对游戏 HUD 美术字易错的问题；**只读屏幕显示值**，不涉及改内存或改数值。

## 2. 成功标准（v1）

1. 可新建/打开游戏项目：`projects/<game_id>/`。
2. 可导入整图或 ROI；可配置预处理（灰度/颜色阈值/二值化）并**自动切成单字符图**。
3. **审核修正 GUI**：展示单字大图；支持键盘 `0-9` 与按钮改标签；支持 `,` `/` `%` `:` 等可选符号类；改完立即写入对应 `dataset/<label>/`。
4. 可训练小 CNN（CPU 可跑）；显示 loss/acc 与各类样本数。
5. 可导出标准包：
   - `digits.onnx`
   - `digits.labels`（每行一类名）
   - `manifest.json`（输入尺寸、预处理约定、类别、游戏名、版本）
6. CLI 可训练/导出（便于无 GUI 复现）；单测覆盖切字、标签映射、manifest 结构（不依赖 GPU）。
7. 仓库推送到 GitHub 远程；README 说明如何把导出包拷到 Studio 工程 `models/`。

## 3. 非目标（v1 明确不做）

- 修改 `auto-script-studio` 的 Kotlin/Lua 运行时（`recognizeDigits` 等留二期）。
- 通用场景 OCR、检测框 YOLO、改游戏数值、绕过反作弊。
- 自动从 ADB 截图（可后续加；v1 以本地导入图片为主）。
- 保证任意粘连字/特效字 100% 切对（切坏的可在审核台删除或重标）。

## 4. 用户工作流

```text
新建游戏项目
  → 导入 HUD 截图 / 已裁 ROI
  → 调预处理 + 自动切字 → 进入审核台
  → 修正错误标签（多字体样本混入同一 label 目录）
  → 训练小 CNN
  → 导出 digits.onnx + labels + manifest
  →（人工）拷贝到 auto-script 工程 models/ 供二期运行时使用
```

误识回流：

```text
脚本/试推理发现错字 → 把该单字图或 ROI 再导入
  → 审核台改正 → 增量样本 → 再训练 → 再导出
```

## 5. 仓库与技术栈

| 项 | 选择 |
|----|------|
| 路径 | `E:\xiangmu\game-digit-trainer` |
| 远程 | GitHub `180024421/game-digit-trainer`（与 gg-base-analyzer 同账号） |
| 语言 | Python ≥ 3.11 |
| GUI | PyQt6 |
| 训练 | PyTorch（默认 CPU） |
| 图像 | OpenCV + Pillow |
| 导出 | ONNX（opset 与 Studio 常见 ONNX Runtime 兼容，优先 17 或文档写明） |

包名：`game_digit_trainer`  
入口：`python -m game_digit_trainer` / `一键启动.cmd`

## 6. 项目目录（每个游戏）

```text
projects/<game_id>/
  config.json           # 标签集、默认预处理、input_size、ROI 预设
  images/raw/           # 原始截图
  images/roi/           # 可选：已裁区域
  dataset/
    0/ 1/ … 9/
    comma/  slash/  percent/  colon/   # 按 config 启用
  runs/<timestamp>/     # checkpoint、metrics
  exports/
    digits.onnx
    digits.labels
    manifest.json
```

**多字体策略**：不按字体分子目录；同一标签下混入该游戏所有外观。一个游戏一个模型。

## 7. 模型与导出契约

### 7.1 模型

- 输入：单通道或 3 通道，固定 `H×W`（默认 32×32），与训练预处理一致。
- 输出：`N` 类 logits / softmax。
- 结构：小型 CNN（数层 conv + pool + FC），参数量小，便于日后进 APK。

### 7.2 `digits.labels`

每行一个类名，顺序与输出维度一致，例如：

```text
0
1
…
9
comma
slash
```

### 7.3 `manifest.json`（契约，供 Studio 二期读取）

```json
{
  "format": "game-digit-trainer/v1",
  "game_id": "mygame",
  "model": "digits.onnx",
  "labels": "digits.labels",
  "input": {
    "width": 32,
    "height": 32,
    "channels": 1,
    "layout": "NCHW",
    "normalize": "0_1"
  },
  "preprocess": {
    "grayscale": true,
    "invert": false,
    "binarize": "otsu",
    "color_filter": null
  },
  "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  "trainer_version": "0.1.0",
  "created_at": "ISO-8601"
}
```

推理侧约定（文档声明，v1 本仓库实现「试推理」即可）：

1. ROI → 与 `preprocess` 一致 → 切字 → 每字 resize → ONNX → argmax  
2. 置信度低于阈值则整串标记 uncertain  
3. 建议多帧投票（由调用方实现）

## 8. GUI 页面

1. **项目**：新建/打开、样本统计、打开文件夹  
2. **导入与切字**：选图、预览二值化/切框、批量写入「待审核」或直接按预测入桶  
3. **审核修正**（核心）：当前字大图、预测与置信度、标签按钮、上一张/下一张、删除、快捷键  
4. **训练**：epoch、batch、lr、开始/停止、曲线或日志  
5. **导出**：选 run、写出 `exports/`、复制路径提示  

风格：实用清晰，参考 gg-base-analyzer 的简洁桌面工具，不追求营销风落地页。

## 9. CLI（与 GUI 共用核心库）

```text
python -m game_digit_trainer segment --project projects/mygame --image x.png
python -m game_digit_trainer train --project projects/mygame
python -m game_digit_trainer export --project projects/mygame --run runs/...
python -m game_digit_trainer predict --project projects/mygame --image roi.png
```

## 10. 测试

- 合成条状数字图：切字数量与顺序  
- 标签名 ↔ 索引映射  
- `manifest.json` 字段完整性  
- 极小随机数据集上 train 1 step + export 文件存在（可 mock 或 CPU 秒级）

## 11. 分期

| 期 | 内容 |
|----|------|
| **v1（本期）** | 仓库 + GUI + 切字/修正/训练/ONNX 导出 + 文档 + 推远程 |
| **v2** | auto-script-studio：`recognizeDigits` / PC 联调读 manifest |
| **v3（可选）** | ADB 截图导入、低置信度自动收样 |

## 12. 许可与免责

MIT。仅供学习与辅助读屏；请遵守游戏服务条款与当地法律；禁止用于破坏公平竞技等用途。
