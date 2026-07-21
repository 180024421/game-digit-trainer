"""行样本全集评估：对照金标跑行模型。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.line_data import list_line_labeled, tokenize_expected
from game_digit_trainer.predict_line import load_line_checkpoint, predict_line_gray
from game_digit_trainer.project import GameProject


def evaluate_line_samples(
    project: GameProject,
    checkpoint: Path,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """对已标行样本逐条推理，返回准确率与错例。"""
    model, classes, _h, max_w = load_line_checkpoint(checkpoint)
    items = list_line_labeled(project)
    if limit is not None:
        items = items[: max(0, int(limit))]

    total = 0
    ok = 0
    mismatches: list[dict[str, object]] = []
    by_shape = {
        "plain": {"ok": 0, "n": 0},
        "with_dot": {"ok": 0, "n": 0},
        "with_unit": {"ok": 0, "n": 0},
    }
    confs: list[float] = []

    for path, expected in items:
        if not expected.strip():
            continue
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        try:
            tokens = tokenize_expected(expected, classes)
        except Exception:
            continue
        pred, _parts, mean_conf = predict_line_gray(model, classes, img, max_w=max_w)
        total += 1
        confs.append(float(mean_conf))
        hit = pred == expected
        if hit:
            ok += 1

        shapes: list[str] = []
        if all(t.isdigit() for t in tokens):
            shapes.append("plain")
        if "dot" in tokens:
            shapes.append("with_dot")
        if any(t in {"wan", "yi"} for t in tokens):
            shapes.append("with_unit")
        for key in shapes:
            by_shape[key]["n"] = int(by_shape[key]["n"]) + 1
            if hit:
                by_shape[key]["ok"] = int(by_shape[key]["ok"]) + 1

        if not hit:
            mismatches.append(
                {
                    "image": path.name,
                    "expected": expected,
                    "pred": pred,
                    "conf": round(float(mean_conf), 4),
                }
            )

    acc = (ok / total) if total else 0.0
    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return {
        "total": total,
        "ok": ok,
        "acc": acc,
        "mean_conf": mean_conf,
        "mismatches": mismatches,
        "by_shape": by_shape,
        "checkpoint": str(checkpoint),
    }


def format_line_eval_report(report: dict[str, object], *, max_mismatch: int = 20) -> str:
    total = int(report.get("total") or 0)
    ok = int(report.get("ok") or 0)
    acc = float(report.get("acc") or 0.0)
    mean_conf = float(report.get("mean_conf") or 0.0)
    lines = [
        f"行样本评估：{ok}/{total} = {acc:.1%} · 平均置信 {mean_conf:.0%}",
        f"checkpoint: {report.get('checkpoint')}",
        "",
    ]
    by_shape = report.get("by_shape") or {}
    for key, title in (
        ("plain", "纯数字"),
        ("with_dot", "带小数"),
        ("with_unit", "带万/亿"),
    ):
        bucket = by_shape.get(key) or {}
        n = int(bucket.get("n") or 0)
        if n <= 0:
            lines.append(f"  {title}: 无样本")
            continue
        hit = int(bucket.get("ok") or 0)
        lines.append(f"  {title}: {hit}/{n} = {hit / n:.1%}")
    mismatches = list(report.get("mismatches") or [])
    if mismatches:
        lines.append("")
        lines.append(f"错例（最多 {max_mismatch} 条）：")
        for m in mismatches[:max_mismatch]:
            lines.append(
                f"  · {m.get('image')}: 金标「{m.get('expected')}」→「{m.get('pred')}」"
                f"（{float(m.get('conf') or 0):.0%}）"
            )
        if len(mismatches) > max_mismatch:
            lines.append(f"  … 另有 {len(mismatches) - max_mismatch} 条")
    else:
        lines.append("")
        lines.append("全部命中（或无有效金标样本）。")
    return "\n".join(lines)
