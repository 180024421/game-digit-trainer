"""金标字符串与预测结果对比。"""
from __future__ import annotations

from game_digit_trainer.labels import display_label, normalize_label


def tokenize_expected(raw: str, classes: list[str]) -> list[str]:
    """把用户输入的金标拆成类别名列表。支持 万/亿 与符号。"""
    s = raw.strip().replace(" ", "")
    out: list[str] = []
    for i, ch in enumerate(s):
        try:
            name = normalize_label(ch)
        except ValueError as exc:
            raise ValueError(f"无法识别字符: {ch!r}（位置 {i + 1}）") from exc
        if name not in classes:
            raise ValueError(f"项目未启用类别「{display_label(name)}」")
        out.append(name)
    return out


def compare_preds(expected: list[str], preds: list[tuple[str, float]]) -> list[dict]:
    """返回不匹配项 {index, expected, pred, conf}。"""
    mismatches: list[dict] = []
    n = max(len(expected), len(preds))
    for i in range(n):
        exp = expected[i] if i < len(expected) else None
        pred, conf = (preds[i] if i < len(preds) else (None, 0.0))
        if exp is None or pred is None or exp != pred:
            mismatches.append(
                {
                    "index": i,
                    "expected": exp,
                    "pred": pred,
                    "conf": conf,
                }
            )
    return mismatches
