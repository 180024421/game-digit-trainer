from __future__ import annotations

# Digits always included
DIGIT_CLASSES: list[str] = [str(i) for i in range(10)]

# Optional punctuation
SYMBOL_CLASS_NAMES: list[str] = ["dot", "comma", "slash", "percent", "colon"]

# Optional Chinese magnitude units (game HUD: 1.2万 / 3亿)
UNIT_CLASS_NAMES: list[str] = ["wan", "yi"]

# Full catalog (digits + symbols + units)
DEFAULT_CLASSES: list[str] = DIGIT_CLASSES + SYMBOL_CLASS_NAMES + UNIT_CLASS_NAMES

# Raw / display char -> folder class name
SYMBOL_CLASSES: dict[str, str] = {
    ".": "dot",
    "·": "dot",  # middle dot sometimes used in HUDs
    ",": "comma",
    "/": "slash",
    "%": "percent",
    ":": "colon",
    "万": "wan",
    "億": "yi",  # traditional
    "亿": "yi",
    "w": "wan",
    "W": "wan",
    "y": "yi",
    "Y": "yi",
}

DISPLAY_FOR_CLASS: dict[str, str] = {
    "dot": ".",
    "comma": ",",
    "slash": "/",
    "percent": "%",
    "colon": ":",
    "wan": "万",
    "yi": "亿",
}


def build_class_list(*, with_symbols: bool = False, with_units: bool = False) -> list[str]:
    classes = list(DIGIT_CLASSES)
    if with_symbols:
        classes.extend(SYMBOL_CLASS_NAMES)
    if with_units:
        classes.extend(UNIT_CLASS_NAMES)
    return classes


def normalize_label(raw: str) -> str:
    s = raw.strip()
    if s in SYMBOL_CLASSES:
        return SYMBOL_CLASSES[s]
    if s in DISPLAY_FOR_CLASS:
        return s
    if len(s) == 1 and s.isdigit():
        return s
    if s in DEFAULT_CLASSES:
        return s
    raise ValueError(f"未知标签: {raw!r}")


def display_label(class_name: str) -> str:
    if class_name in DISPLAY_FOR_CLASS:
        return DISPLAY_FOR_CLASS[class_name]
    return class_name


def index_to_label(classes: list[str], index: int) -> str:
    return classes[index]


def label_to_index(classes: list[str], label: str) -> int:
    name = normalize_label(label)
    return classes.index(name)
