from __future__ import annotations

# Display label -> folder / class name
DEFAULT_CLASSES: list[str] = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "comma",
    "slash",
    "percent",
    "colon",
]

# Optional extras beyond digits; enabled by project config
SYMBOL_CLASSES: dict[str, str] = {
    ",": "comma",
    "/": "slash",
    "%": "percent",
    ":": "colon",
}

DISPLAY_FOR_CLASS: dict[str, str] = {
    "comma": ",",
    "slash": "/",
    "percent": "%",
    "colon": ":",
}


def normalize_label(raw: str) -> str:
    s = raw.strip()
    if s in SYMBOL_CLASSES:
        return SYMBOL_CLASSES[s]
    if s in DISPLAY_FOR_CLASS or (len(s) == 1 and s.isdigit()) or s in DEFAULT_CLASSES:
        if s.isdigit():
            return s
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
