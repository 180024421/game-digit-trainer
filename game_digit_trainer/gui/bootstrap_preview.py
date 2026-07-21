"""粗切单字预览对话框：让用户看到「行→单字」切得怎么样。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from game_digit_trainer.gui.widgets import numpy_to_pixmap
from game_digit_trainer.labels import display_label


def show_bootstrap_preview(
    parent,
    added: dict[str, int],
    previews: list[tuple[str, Path, str]],
    *,
    title: str = "行→单字粗切预览",
) -> None:
    total = sum(added.values())
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(720, 480)
    root = QVBoxLayout(dlg)
    summary = "、".join(f"{display_label(k)}×{v}" for k, v in sorted(added.items())[:16])
    tip = QLabel(
        f"共新增 {total} 张单字（粗切，供合成行增强，不替代整行训练）。\n"
        f"{summary}\n"
        "下面是部分切出效果：类名 · 来源行金标。切歪的可到 ③ 样本库删掉。"
    )
    tip.setWordWrap(True)
    tip.setObjectName("hintLabel")
    root.addWidget(tip)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    body = QWidget()
    grid = QGridLayout(body)
    grid.setSpacing(8)
    cols = 6
    for i, (tok, path, src_text) in enumerate(previews):
        cell = QVBoxLayout()
        img_lab = QLabel()
        img_lab.setFixedSize(88, 56)
        img_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lab.setStyleSheet("background:#111827; border-radius:6px;")
        raw = np.fromfile(str(path), dtype=np.uint8)
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            show = gray
            if float(np.mean(show)) < 90:
                show = cv2.normalize(show, None, 40, 255, cv2.NORM_MINMAX)
            pix = numpy_to_pixmap(show).scaled(
                84,
                52,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            img_lab.setPixmap(pix)
        else:
            img_lab.setText("?")
        name = QLabel(f"{display_label(tok)}\n←{src_text[:10]}")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet("font-size:11px; color:#94a3b8;")
        cell.addWidget(img_lab)
        cell.addWidget(name)
        wrap = QWidget()
        wrap.setLayout(cell)
        grid.addWidget(wrap, i // cols, i % cols)
    if not previews:
        grid.addWidget(QLabel("（无预览图）"), 0, 0)
    scroll.setWidget(body)
    root.addWidget(scroll, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    buttons.accepted.connect(dlg.accept)
    root.addWidget(buttons)
    dlg.exec()
