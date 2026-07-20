# 切字页：加载导出模型验识别 — 设计规格

日期：2026-07-20  
状态：待用户审阅规格  
确认项：

- 场景：训练站 GUI「验模型」（方案 A）
- 入口：① 截图切字页内嵌（方案 B），复用画布与圈选
- 模型：支持 ONNX 导出包 **与** `.pt` checkpoint；可切换、可点选；可浏览外部 ONNX 包加入列表

## 1. 目标

在「① 截图切字」页用**用户选定的模型**对当前截图上圈出的区域做数字串识别，便于导出后自测、对比不同导出包，而不必只依赖「最新 checkpoint + 预览识别」。

## 2. 成功标准

1. 切字页「更多」区有「验模型」折叠块（默认展开或与更多联动即可）。
2. **模型下拉**可列出并点选：
   - 当前工程 `exports/` 下发现的 `digits.onnx`（含子目录）
   - 当前工程 `checkpoints/` 下 `.pt`（按修改时间取最近若干，如 10 个）
   - 用户「浏览…」加入的外部 ONNX 包路径（写入 UI prefs，重启仍在）
3. 下拉项文案区分类型，例如 `[ONNX] exports/digits.onnx`、`[PT] epoch_12.pt`。
4. 点「用所选模型识别」：
   - 输入：当前打开的截图；优先使用画布**绿字框**；若无字框则对**蓝 ROI** 内自动切字；若无蓝框则整图切字（与现有预览逻辑一致，可复用）。
   - 输出：下方「识别预览」大号字符串 + 明细置信度；状态栏注明所用模型短名。
5. ONNX 路径走 `onnxruntime`；未安装时明确提示，且仍可选 `.pt`。
6. 现有「预览识别」行为不变（仍默认用 `latest_checkpoint`），避免打断训练中试跑习惯。

## 3. 非目标

- 不改 auto-script-studio 运行时。
- 不做批量文件夹验模、不做与金标自动对比（金标/回归集已有入口）。
- 不在本迭代做模型管理删除 UI（可从 prefs 清最近列表即可，可选）。

## 4. UI 布局（切字页 · 更多）

```text
验模型
  模型: [ 下拉：可选 ONNX / PT 列表 ▼ ]  [刷新] [浏览 ONNX…]
  [用所选模型识别]
  提示：圈蓝框/绿字框后识别；结果见上方预览区
```

- 「刷新」：重新扫描工程 exports/checkpoints。
- 「浏览 ONNX…」：选 `digits.onnx` 或含该文件的目录；校验同目录尽量有 `digits.labels`（缺失则报错）；可选读 `manifest.json` 取宽高。

## 5. 数据与推理

### 5.1 模型描述符

```text
kind: "onnx" | "pt"
path: Path          # onnx 文件或 .pt 文件
labels_path: Path?  # onnx 旁 digits.labels
manifest_path: Path?
display: str
```

### 5.2 扫描规则

| 来源 | 规则 |
|------|------|
| exports | `project.exports_dir.rglob("digits.onnx")` |
| checkpoints | `project.checkpoints_dir.glob("*.pt")`，按 mtime 降序，最多 10 |
| recent | `ui_prefs["recent_onnx_models"]`：绝对路径字符串列表（上限 8） |

当前无工程时：下拉仅显示 recent；识别时提示先打开项目（或仅当图已打开且模型为外部 ONNX 时允许——**v1 要求已打开项目**，预处理用项目 config）。

### 5.3 推理 API（新建/扩展 `predict.py`）

- `list_project_models(project) -> list[ModelRef]`
- `predict_boxes_string_onnx(project, bgr, boxes, onnx_path, *, conf_threshold) -> (text, parts)`
  - 读 labels + manifest 宽高；`prepare_tensor_image` 后 ORT 跑 logits → softmax
- 现有 `predict_boxes_string(..., checkpoint)` 继续服务 `.pt` 与「预览识别」

ROI/字框：GUI 侧与 `_preview_recognize` 相同取框逻辑，再调用上述 API。

## 6. Prefs

`ui_prefs`（或现有 prefs 文件）增加：

```json
"recent_onnx_models": ["D:/.../digits.onnx"],
"last_verify_model": "onnx:D:/.../digits.onnx"
```

`last_verify_model` 用于下次打开下拉默认选中。

## 7. 错误与依赖

| 情况 | 行为 |
|------|------|
| 无模型可选 | 下拉占位「（无模型，请先导出或浏览）」 |
| 无截图 | 提示先截图/打开图 |
| 无框且自动切字失败 | 提示框选蓝框或调间距 |
| 缺 onnxruntime | MessageBox 提示 `pip install onnxruntime` |
| 缺 labels | 拒绝加载该 ONNX |

## 8. 测试

- 单测：假 ONNX 目录结构解析、`list_project_models` 扫描（tmp_path fixture）。
- ORT 推理：若环境无 onnxruntime 则 skip；有则对 dummy 1×1×H×W 不崩溃。
- 不依赖真机 GUI。

## 9. 实现落点

| 文件 | 变更 |
|------|------|
| `game_digit_trainer/predict.py` | ModelRef、list、ONNX 按框推理 |
| `game_digit_trainer/gui/app.py` | 更多区 UI + 槽函数 |
| `game_digit_trainer/gui/ui_prefs.py` | recent / last_verify |
| `tests/test_verify_model.py` | 新建 |
| `README.md` | 一句：切字页可选用导出 ONNX 验识别 |

## 10. 验收手测

1. 训练并导出 → 切字页刷新 → 下拉出现 `[ONNX] ...`  
2. 截图 → 蓝框 → 「用所选模型识别」→ 预览区有读数  
3. 浏览另一导出目录 → 下拉可切换 → 再识别  
4. 选 `[PT] ...` → 同样能识别  
5. 「预览识别」仍走最新 checkpoint  
