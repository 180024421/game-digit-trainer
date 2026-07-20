"""新建项目时的类别模板。"""
from __future__ import annotations

TEMPLATES: dict[str, dict] = {
    "digits": {
        "label": "仅数字 0-9",
        "symbols": False,
        "units": False,
    },
    "coins": {
        "label": "金币（数字+小数点+万/亿）",
        "symbols": False,
        "units": True,
        "force_classes": [str(i) for i in range(10)] + ["dot", "wan", "yi"],
    },
    "hp_percent": {
        "label": "血量百分比（数字+%）",
        "symbols": True,  # includes more than % — user can ignore
        "units": False,
        "force_classes": [str(i) for i in range(10)] + ["percent"],
    },
    "timer": {
        "label": "倒计时（数字+:）",
        "symbols": False,
        "units": False,
        "force_classes": [str(i) for i in range(10)] + ["colon"],
    },
    "full": {
        "label": "全开（数字+符号+万亿）",
        "symbols": True,
        "units": True,
    },
}


def template_choices() -> list[tuple[str, str]]:
    return [(k, v["label"]) for k, v in TEMPLATES.items()]


def resolve_template(key: str) -> tuple[bool, bool, list[str] | None]:
    """返回 (with_symbols, with_units, force_classes|None)。"""
    t = TEMPLATES.get(key) or TEMPLATES["coins"]
    return bool(t["symbols"]), bool(t["units"]), t.get("force_classes")
