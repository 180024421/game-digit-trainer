"""类别样本均衡检查。"""
from __future__ import annotations


def balance_warnings(counts: dict[str, int], *, min_per_class: int = 5) -> list[str]:
    """返回人类可读警告列表（空表示基本均衡）。"""
    warnings: list[str] = []
    positive = {k: v for k, v in counts.items() if v > 0}
    if not positive:
        return ["尚无已标注样本"]
    total = sum(counts.values())
    zeros = [k for k, v in counts.items() if v == 0]
    # 只提示「启用了但完全没样本」里，数字类优先
    digit_zeros = [k for k in zeros if k.isdigit()]
    other_zeros = [k for k in zeros if not k.isdigit()]
    if digit_zeros and total >= 10:
        warnings.append(f"数字类无样本: {', '.join(digit_zeros[:8])}" + ("…" if len(digit_zeros) > 8 else ""))
    if other_zeros and total >= 20:
        warnings.append(f"符号/单位无样本: {', '.join(other_zeros[:6])}")

    vals = list(positive.values())
    lo, hi = min(vals), max(vals)
    if lo > 0 and hi >= min_per_class * 3 and hi / max(lo, 1) >= 5:
        low_classes = [k for k, v in positive.items() if v == lo]
        high_classes = [k for k, v in positive.items() if v == hi]
        warnings.append(
            f"样本不均：最多 {hi}（{high_classes[0]}）/ 最少 {lo}（{low_classes[0]}），建议补少的类"
        )
    thin = [k for k, v in positive.items() if 0 < v < min_per_class]
    if thin and total >= 15:
        warnings.append(f"样本偏少（<{min_per_class}）: {', '.join(thin[:8])}")
    return warnings


def format_balance_text(counts: dict[str, int]) -> str:
    warns = balance_warnings(counts)
    if not warns:
        return "类别均衡：看起来还行"
    return "均衡提示：\n- " + "\n- ".join(warns)
