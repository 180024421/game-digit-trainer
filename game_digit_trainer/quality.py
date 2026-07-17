"""导出前质量门禁与 ONNX 自检。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from game_digit_trainer.project import GameProject


def min_samples_gate(counts: dict[str, int], *, min_per_digit: int = 5) -> list[str]:
    errs: list[str] = []
    digits = [k for k in counts if k.isdigit()]
    thin = [k for k in digits if counts.get(k, 0) < min_per_digit]
    if thin:
        errs.append(f"数字类样本不足（<{min_per_digit}）: {', '.join(thin)}")
    total = sum(counts.values())
    if total < 20:
        errs.append(f"总样本过少（{total} < 20）")
    return errs


def checkpoint_acc_gate(checkpoint: Path, *, min_val_acc: float = 0.7) -> list[str]:
    errs: list[str] = []
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        return [f"无法读取 checkpoint: {exc}"]
    acc = ckpt.get("val_acc")
    if acc is None:
        return []
    if float(acc) < min_val_acc:
        errs.append(f"验证准确率偏低：{float(acc):.1%} < {min_val_acc:.0%}（建议多标难例再训）")
    return errs


def export_quality_report(
    project: GameProject,
    checkpoint: Path,
    *,
    min_per_digit: int = 5,
    min_val_acc: float = 0.7,
) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。errors 非空时应阻止导出（可强制）。"""
    counts = project.class_counts()
    errors = min_samples_gate(counts, min_per_digit=min_per_digit)
    warnings = checkpoint_acc_gate(checkpoint, min_val_acc=min_val_acc)
    # 有样本为 0 的启用类 → warning
    zeros = [k for k, v in counts.items() if v == 0]
    if zeros and sum(counts.values()) >= 15:
        warnings.append(f"以下类别尚无样本: {', '.join(zeros[:10])}")
    return errors, warnings


def verify_onnx_runtime(onnx_path: Path, *, width: int, height: int) -> str:
    """用 onnxruntime（若已装）跑一次 dummy；否则仅检查文件存在。"""
    if not onnx_path.is_file():
        raise RuntimeError(f"找不到 {onnx_path}")
    try:
        import onnxruntime as ort
    except ImportError:
        return "已导出文件；未安装 onnxruntime，跳过运行时自检（可选: pip install onnxruntime）"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    name = inp.name
    x = np.zeros((1, 1, height, width), dtype=np.float32)
    out = sess.run(None, {name: x})
    if not out:
        raise RuntimeError("ONNX Runtime 推理无输出")
    return f"ONNX Runtime 自检通过（输出 shape={getattr(out[0], 'shape', '?')}）"
