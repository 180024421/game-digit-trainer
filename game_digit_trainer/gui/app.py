from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from game_digit_trainer.export_onnx import export_onnx, latest_checkpoint
from game_digit_trainer.labels import display_label
from game_digit_trainer.preprocess import apply_preprocess, load_bgr
from game_digit_trainer.project import GameProject, create_project, open_project, projects_root
from game_digit_trainer.segment import move_to_label, save_pending_chars, segment_image
from game_digit_trainer.train import train_project


def _numpy_to_pixmap(gray_or_bgr) -> QPixmap:
    import numpy as np

    img = np.ascontiguousarray(gray_or_bgr)
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class TrainWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, project: GameProject, epochs: int) -> None:
        super().__init__()
        self.project = project
        self.epochs = epochs

    def run(self) -> None:
        try:
            path = train_project(
                self.project,
                epochs=self.epochs,
                log=lambda m: self.log.emit(m),
            )
            self.done.emit(str(path))
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("游戏数字训练站 · game-digit-trainer")
        self.resize(1100, 720)
        self.project: GameProject | None = None
        self._pending: list[Path] = []
        self._idx = 0
        self._worker: TrainWorker | None = None

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        self.tab_project = QWidget()
        self.tab_import = QWidget()
        self.tab_review = QWidget()
        self.tab_train = QWidget()
        tabs.addTab(self.tab_project, "项目")
        tabs.addTab(self.tab_import, "导入切字")
        tabs.addTab(self.tab_review, "审核修正")
        tabs.addTab(self.tab_train, "训练导出")

        self._build_project()
        self._build_import()
        self._build_review()
        self._build_train()
        self._status = self.statusBar()
        self._status.showMessage("就绪：先新建或打开游戏项目")

    def _build_project(self) -> None:
        layout = QVBoxLayout(self.tab_project)
        row = QHBoxLayout()
        self.game_id_edit = QLineEdit()
        self.game_id_edit.setPlaceholderText("游戏 ID，如 mygame")
        self.chk_symbols = QCheckBox("启用符号 , / % :")
        btn_new = QPushButton("新建项目")
        btn_open = QPushButton("打开项目…")
        btn_new.clicked.connect(self._new_project)
        btn_open.clicked.connect(self._open_project_dialog)
        row.addWidget(self.game_id_edit)
        row.addWidget(self.chk_symbols)
        row.addWidget(btn_new)
        row.addWidget(btn_open)
        layout.addLayout(row)

        self.project_info = QTextEdit()
        self.project_info.setReadOnly(True)
        layout.addWidget(self.project_info)
        layout.addWidget(QLabel(f"默认项目目录: {projects_root()}"))

    def _build_import(self) -> None:
        layout = QVBoxLayout(self.tab_import)
        row = QHBoxLayout()
        btn_pick = QPushButton("选择截图…")
        btn_seg = QPushButton("切字并加入待审核")
        self.chk_invert = QCheckBox("反色")
        btn_pick.clicked.connect(self._pick_images)
        btn_seg.clicked.connect(self._segment_selected)
        row.addWidget(btn_pick)
        row.addWidget(btn_seg)
        row.addWidget(self.chk_invert)
        row.addStretch()
        layout.addLayout(row)

        self.import_list = QListWidget()
        layout.addWidget(self.import_list)
        self.preview_label = QLabel("预处理预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setStyleSheet("background:#1e1e1e;color:#aaa;")
        layout.addWidget(self.preview_label)
        self.import_list.currentRowChanged.connect(self._preview_import)

    def _build_review(self) -> None:
        layout = QVBoxLayout(self.tab_review)
        top = QHBoxLayout()
        btn_reload = QPushButton("刷新待审核")
        btn_reload.clicked.connect(self._reload_pending)
        self.review_meta = QLabel("无待审核")
        top.addWidget(btn_reload)
        top.addWidget(self.review_meta)
        top.addStretch()
        layout.addLayout(top)

        self.char_view = QLabel()
        self.char_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.char_view.setMinimumHeight(220)
        self.char_view.setStyleSheet("background:#111; border:1px solid #333;")
        layout.addWidget(self.char_view)

        grid = QGridLayout()
        self._label_buttons: list[QPushButton] = []
        for i in range(10):
            b = QPushButton(str(i))
            b.setFixedHeight(40)
            b.clicked.connect(lambda _=False, d=str(i): self._assign(d))
            grid.addWidget(b, 0, i)
            self._label_buttons.append(b)
        layout.addLayout(grid)

        sym_row = QHBoxLayout()
        for text, lab in [("逗号 ,", ","), ("斜杠 /", "/"), ("百分号 %", "%"), ("冒号 :", ":")]:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, d=lab: self._assign(d))
            sym_row.addWidget(b)
        btn_del = QPushButton("删除当前")
        btn_skip = QPushButton("跳过")
        btn_del.clicked.connect(self._delete_current)
        btn_skip.clicked.connect(self._next_pending)
        sym_row.addWidget(btn_del)
        sym_row.addWidget(btn_skip)
        layout.addLayout(sym_row)

        nav = QHBoxLayout()
        btn_prev = QPushButton("上一张")
        btn_next = QPushButton("下一张")
        btn_prev.clicked.connect(self._prev_pending)
        btn_next.clicked.connect(self._next_pending)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        layout.addLayout(nav)

        for d in "0123456789":
            QShortcut(QKeySequence(d), self.tab_review, activated=lambda d=d: self._assign(d))

    def _build_train(self) -> None:
        layout = QVBoxLayout(self.tab_train)
        box = QGroupBox("训练")
        form = QHBoxLayout(box)
        form.addWidget(QLabel("Epochs"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(15)
        form.addWidget(self.epochs_spin)
        btn_train = QPushButton("开始训练")
        btn_train.clicked.connect(self._start_train)
        form.addWidget(btn_train)
        form.addStretch()
        layout.addWidget(box)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log)

        exp = QHBoxLayout()
        btn_export = QPushButton("导出 ONNX 包")
        btn_export.clicked.connect(self._export)
        exp.addWidget(btn_export)
        exp.addStretch()
        layout.addLayout(exp)

    # ---- project ----
    def _new_project(self) -> None:
        gid = self.game_id_edit.text().strip()
        if not gid:
            QMessageBox.warning(self, "提示", "请填写游戏 ID")
            return
        try:
            self.project = create_project(gid, with_symbols=self.chk_symbols.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "新建失败", str(exc))
            return
        self._refresh_project_info()
        self._status.showMessage(f"已创建项目 {self.project.root}")

    def _open_project_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录", str(projects_root()))
        if not path:
            return
        try:
            self.project = open_project(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        self.game_id_edit.setText(self.project.config.game_id)
        self._refresh_project_info()
        self._reload_pending()
        self._status.showMessage(f"已打开 {self.project.root}")

    def _refresh_project_info(self) -> None:
        if not self.project:
            self.project_info.setPlainText("")
            return
        counts = self.project.class_counts()
        lines = [
            f"路径: {self.project.root}",
            f"游戏: {self.project.config.game_id}",
            f"类别: {', '.join(self.project.config.classes)}",
            f"输入尺寸: {self.project.config.input_width}x{self.project.config.input_height}",
            "",
            "样本数:",
        ]
        for k, v in counts.items():
            lines.append(f"  {display_label(k)} ({k}): {v}")
        lines.append(f"待审核: {len(self.project.pending_files())}")
        self.project_info.setPlainText("\n".join(lines))

    def _require_project(self) -> GameProject | None:
        if not self.project:
            QMessageBox.warning(self, "提示", "请先新建或打开项目")
            return None
        return self.project

    # ---- import ----
    def _pick_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择截图",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        for f in files:
            self.import_list.addItem(f)

    def _preview_import(self, row: int) -> None:
        if row < 0 or not self.project:
            return
        path = Path(self.import_list.item(row).text())
        try:
            bgr = load_bgr(path)
            cfg = self.project.config.preprocess
            cfg.invert = self.chk_invert.isChecked()
            binary = apply_preprocess(bgr, cfg)
            pix = _numpy_to_pixmap(binary)
            self.preview_label.setPixmap(
                pix.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as exc:
            self.preview_label.setText(str(exc))

    def _segment_selected(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if self.import_list.count() == 0:
            QMessageBox.warning(self, "提示", "请先选择截图")
            return
        proj.config.preprocess.invert = self.chk_invert.isChecked()
        total = 0
        for i in range(self.import_list.count()):
            src = Path(self.import_list.item(i).text())
            dest_raw = proj.raw_dir / src.name
            if not dest_raw.exists():
                shutil.copy2(src, dest_raw)
            binary, crops = segment_image(src, proj.config.preprocess)
            del binary
            paths = save_pending_chars(proj, src, crops)
            total += len(paths)
        self._reload_pending()
        self._refresh_project_info()
        QMessageBox.information(self, "完成", f"已切出 {total} 个字符到待审核")

    # ---- review ----
    def _reload_pending(self) -> None:
        if not self.project:
            return
        self._pending = self.project.pending_files()
        self._idx = 0
        self._show_current()

    def _show_current(self) -> None:
        if not self._pending:
            self.char_view.setText("无待审核")
            self.review_meta.setText("无待审核")
            return
        self._idx = max(0, min(self._idx, len(self._pending) - 1))
        path = self._pending[self._idx]
        raw = path.read_bytes()
        import numpy as np
        import cv2

        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.char_view.setText("无法读取")
            return
        pix = _numpy_to_pixmap(img)
        self.char_view.setPixmap(
            pix.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        )
        self.review_meta.setText(f"{self._idx + 1}/{len(self._pending)}  {path.name}")

    def _assign(self, label: str) -> None:
        proj = self._require_project()
        if not proj or not self._pending:
            return
        path = self._pending[self._idx]
        try:
            move_to_label(proj, path, label)
        except Exception as exc:
            QMessageBox.warning(self, "标注失败", str(exc))
            return
        self._pending.pop(self._idx)
        if self._idx >= len(self._pending) and self._pending:
            self._idx = len(self._pending) - 1
        self._show_current()
        self._refresh_project_info()
        self._status.showMessage(f"已标为 {label}")

    def _delete_current(self) -> None:
        if not self._pending:
            return
        path = self._pending[self._idx]
        path.unlink(missing_ok=True)
        self._pending.pop(self._idx)
        if self._idx >= len(self._pending) and self._pending:
            self._idx = len(self._pending) - 1
        self._show_current()
        self._refresh_project_info()

    def _next_pending(self) -> None:
        if self._pending:
            self._idx = min(self._idx + 1, len(self._pending) - 1)
            self._show_current()

    def _prev_pending(self) -> None:
        if self._pending:
            self._idx = max(self._idx - 1, 0)
            self._show_current()

    # ---- train / export ----
    def _start_train(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "提示", "训练进行中")
            return
        self.train_log.clear()
        self._worker = TrainWorker(proj, self.epochs_spin.value())
        self._worker.log.connect(lambda m: self.train_log.append(m))
        self._worker.done.connect(self._train_done)
        self._worker.failed.connect(lambda e: QMessageBox.critical(self, "训练失败", e))
        self._worker.start()

    def _train_done(self, path: str) -> None:
        self.train_log.append(f"完成: {path}")
        self._status.showMessage("训练完成，可导出 ONNX")
        QMessageBox.information(self, "完成", f"最佳模型:\n{path}")

    def _export(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            QMessageBox.warning(self, "提示", "没有 checkpoint，请先训练")
            return
        try:
            out = export_onnx(proj, ckpt)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"{exc}\n{traceback.format_exc()}")
            return
        QMessageBox.information(self, "已导出", f"导出目录:\n{out.parent}")


def run_gui() -> int:
    import sys

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()
