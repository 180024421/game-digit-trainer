from __future__ import annotations

import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from game_digit_trainer.capture import (
    capture_adb,
    capture_clipboard_bgr,
    list_adb_devices,
    qimage_to_bgr,
    save_bgr,
)
from game_digit_trainer.export_onnx import export_onnx, latest_checkpoint
from game_digit_trainer.gui.region_capture import RegionCaptureOverlay
from game_digit_trainer.gui.widgets import ImageCanvas, numpy_to_pixmap
from game_digit_trainer.labels import display_label, normalize_label
from game_digit_trainer.predict import predict_pending_file
from game_digit_trainer.preprocess import apply_preprocess, load_bgr
from game_digit_trainer.project import (
    GameProject,
    create_project,
    ensure_unit_classes,
    open_project,
    projects_root,
)
from game_digit_trainer.segment import (
    crop_bgr,
    list_dataset_files,
    move_to_label,
    relabel_dataset_file,
    save_pending_chars,
    segment_binary,
    segment_image,
)
from game_digit_trainer.train import train_project


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
        self.resize(1280, 820)
        self.project: GameProject | None = None
        self._pending: list[Path] = []
        self._idx = 0
        self._worker: TrainWorker | None = None
        self._pred_label: str | None = None
        self._pred_conf: float = 0.0
        self._current_import: Path | None = None

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        self.tab_project = QWidget()
        self.tab_import = QWidget()
        self.tab_review = QWidget()
        self.tab_dataset = QWidget()
        self.tab_train = QWidget()
        tabs.addTab(self.tab_project, "1. 项目")
        tabs.addTab(self.tab_import, "2. 导入切字")
        tabs.addTab(self.tab_review, "3. 审核修正")
        tabs.addTab(self.tab_dataset, "4. 数据集")
        tabs.addTab(self.tab_train, "5. 训练导出")
        self.tabs = tabs
        tabs.currentChanged.connect(self._on_tab_changed)

        self._build_project()
        self._build_import()
        self._build_review()
        self._build_dataset()
        self._build_train()
        self._status = self.statusBar()
        self._status.showMessage("就绪：先新建或打开游戏项目 · 导入页可拖拽框选 ROI")

    # ---------- build tabs ----------
    def _build_project(self) -> None:
        layout = QVBoxLayout(self.tab_project)
        tip = QLabel(
            "流程：新建项目 → 框选截屏/ADB截图 → 框选数字区域切字 → 审核（空格确认）→ 训练导出"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#555; padding:4px;")
        layout.addWidget(tip)

        row = QHBoxLayout()
        self.game_id_edit = QLineEdit()
        self.game_id_edit.setPlaceholderText("游戏 ID，如 mygame")
        self.chk_symbols = QCheckBox("符号 ,/%:")
        self.chk_units = QCheckBox("单位 万/亿")
        self.chk_units.setChecked(True)
        btn_new = QPushButton("新建项目")
        btn_open = QPushButton("打开项目…")
        btn_folder = QPushButton("打开文件夹")
        btn_add_units = QPushButton("已有项目加万/亿")
        btn_new.clicked.connect(self._new_project)
        btn_open.clicked.connect(self._open_project_dialog)
        btn_folder.clicked.connect(self._open_project_folder)
        btn_add_units.clicked.connect(self._add_units_to_project)
        row.addWidget(self.game_id_edit, 2)
        row.addWidget(self.chk_symbols)
        row.addWidget(self.chk_units)
        row.addWidget(btn_new)
        row.addWidget(btn_open)
        row.addWidget(btn_folder)
        row.addWidget(btn_add_units)
        layout.addLayout(row)

        self.project_info = QTextEdit()
        self.project_info.setReadOnly(True)
        layout.addWidget(self.project_info)
        layout.addWidget(QLabel(f"默认项目目录: {projects_root()}"))

    def _build_import(self) -> None:
        root = QHBoxLayout(self.tab_import)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        btn_row = QHBoxLayout()
        btn_region = QPushButton("框选截屏")
        btn_region.setStyleSheet("font-weight:600;")
        btn_adb = QPushButton("ADB截图")
        btn_paste = QPushButton("粘贴剪贴板")
        btn_pick = QPushButton("添加文件…")
        btn_clear = QPushButton("清空列表")
        btn_region.clicked.connect(self._capture_region)
        btn_adb.clicked.connect(self._capture_adb)
        btn_paste.clicked.connect(self._capture_clipboard)
        btn_pick.clicked.connect(self._pick_images)
        btn_clear.clicked.connect(self._clear_import_list)
        btn_row.addWidget(btn_region)
        btn_row.addWidget(btn_adb)
        btn_row.addWidget(btn_paste)
        btn_row.addWidget(btn_pick)
        btn_row.addWidget(btn_clear)
        left_l.addLayout(btn_row)
        self.import_list = QListWidget()
        self.import_list.currentRowChanged.connect(self._on_import_row)
        left_l.addWidget(self.import_list)
        splitter.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        tools = QHBoxLayout()
        self.chk_invert = QCheckBox("反色")
        self.chk_invert.stateChanged.connect(lambda _: self._refresh_import_preview())
        self.binarize_combo = QComboBox()
        self.binarize_combo.addItems(["otsu", "adaptive", "none"])
        self.binarize_combo.currentTextChanged.connect(lambda _: self._refresh_import_preview())
        self.chk_show_binary = QCheckBox("看二值图")
        self.chk_show_binary.setChecked(True)
        self.chk_show_binary.stateChanged.connect(lambda _: self._refresh_import_preview())
        tools.addWidget(QLabel("二值化"))
        tools.addWidget(self.binarize_combo)
        tools.addWidget(self.chk_invert)
        tools.addWidget(self.chk_show_binary)
        tools.addWidget(QLabel("字间距阈值"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(1, 20)
        self.gap_spin.setValue(3)
        self.gap_spin.setToolTip("越大越容易把粘连字拆开；太小会把一个字切碎")
        self.gap_spin.valueChanged.connect(lambda _: self._refresh_import_preview())
        tools.addWidget(self.gap_spin)
        tools.addStretch()
        right_l.addLayout(tools)

        self.import_canvas = ImageCanvas()
        self.import_canvas.roi_changed.connect(lambda _: self._refresh_import_preview(keep_roi=True))
        right_l.addWidget(self.import_canvas, 1)

        actions = QHBoxLayout()
        btn_clear_roi = QPushButton("清除框选（用整图）")
        btn_preview = QPushButton("刷新切框预览")
        btn_seg = QPushButton("切字 → 待审核")
        btn_seg.setStyleSheet("font-weight:600; padding:8px 16px;")
        btn_clear_roi.clicked.connect(self._clear_roi)
        btn_preview.clicked.connect(lambda: self._refresh_import_preview())
        btn_seg.clicked.connect(self._segment_current)
        actions.addWidget(btn_clear_roi)
        actions.addWidget(btn_preview)
        actions.addStretch()
        actions.addWidget(btn_seg)
        right_l.addLayout(actions)

        hint = QLabel(
            "截图：框选截屏 / ADB（雷电）/ Win+Shift+S 后粘贴。然后在图上再框选数字区域，绿框对准后切字。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;")
        right_l.addWidget(hint)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def _build_review(self) -> None:
        layout = QVBoxLayout(self.tab_review)
        top = QHBoxLayout()
        btn_reload = QPushButton("刷新")
        btn_reload.clicked.connect(self._reload_pending)
        self.review_meta = QLabel("无待审核")
        self.pred_label_ui = QLabel("预测：—")
        self.pred_label_ui.setStyleSheet("font-size:18px; font-weight:600; color:#0a7;")
        top.addWidget(btn_reload)
        top.addWidget(self.review_meta, 1)
        top.addWidget(self.pred_label_ui)
        layout.addLayout(top)

        self.char_view = QLabel()
        self.char_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.char_view.setMinimumHeight(260)
        self.char_view.setStyleSheet("background:#111; border:1px solid #333;")
        layout.addWidget(self.char_view)

        confirm_row = QHBoxLayout()
        self.btn_confirm = QPushButton("空格：确认预测")
        self.btn_confirm.setStyleSheet(
            "background:#1a7f37; color:white; font-size:16px; padding:10px; font-weight:600;"
        )
        self.btn_confirm.clicked.connect(self._confirm_prediction)
        confirm_row.addWidget(self.btn_confirm)
        layout.addLayout(confirm_row)

        grid = QGridLayout()
        for i in range(10):
            b = QPushButton(str(i))
            b.setFixedHeight(44)
            b.setShortcut(QKeySequence(str(i)))
            b.clicked.connect(lambda _=False, d=str(i): self._assign(d))
            grid.addWidget(b, 0, i)
        layout.addLayout(grid)

        sym_row = QHBoxLayout()
        for text, lab in [
            ("逗号 ,", ","),
            ("斜杠 /", "/"),
            ("百分号 %", "%"),
            ("冒号 :", ":"),
            ("万 (W)", "万"),
            ("亿 (Y)", "亿"),
        ]:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, d=lab: self._assign(d))
            sym_row.addWidget(b)
        btn_del = QPushButton("删除 Del")
        btn_skip = QPushButton("跳过 →")
        btn_del.clicked.connect(self._delete_current)
        btn_skip.clicked.connect(self._next_pending)
        sym_row.addWidget(btn_del)
        sym_row.addWidget(btn_skip)
        layout.addLayout(sym_row)

        nav = QHBoxLayout()
        btn_prev = QPushButton("← 上一张")
        btn_next = QPushButton("下一张 →")
        btn_prev.clicked.connect(self._prev_pending)
        btn_next.clicked.connect(self._next_pending)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        layout.addLayout(nav)

        tip = QLabel(
            "快捷键：0-9 标注 · W=万 · Y=亿 · 空格确认预测 · ←/→ 翻页 · Delete 删除"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#666;")
        layout.addWidget(tip)

        # Window-level shortcuts that work when review tab focused
        QShortcut(QKeySequence(Qt.Key.Key_Space), self.tab_review, activated=self._confirm_prediction)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self.tab_review, activated=self._delete_current)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.tab_review, activated=self._prev_pending)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self.tab_review, activated=self._next_pending)
        QShortcut(QKeySequence("W"), self.tab_review, activated=lambda: self._assign("万"))
        QShortcut(QKeySequence("Y"), self.tab_review, activated=lambda: self._assign("亿"))
        for d in "0123456789":
            QShortcut(QKeySequence(d), self.tab_review, activated=lambda d=d: self._assign(d))

    def _build_dataset(self) -> None:
        layout = QHBoxLayout(self.tab_dataset)
        splitter = QSplitter()
        layout.addWidget(splitter)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("类别"))
        self.ds_class_list = QListWidget()
        self.ds_class_list.currentTextChanged.connect(self._on_ds_class)
        ll.addWidget(self.ds_class_list)
        btn_refresh_ds = QPushButton("刷新统计")
        btn_refresh_ds.clicked.connect(self._reload_dataset_browser)
        ll.addWidget(btn_refresh_ds)
        splitter.addWidget(left)

        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.addWidget(QLabel("样本文件"))
        self.ds_file_list = QListWidget()
        self.ds_file_list.currentRowChanged.connect(self._on_ds_file)
        ml.addWidget(self.ds_file_list)
        splitter.addWidget(mid)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.ds_preview = QLabel("选择样本预览")
        self.ds_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ds_preview.setMinimumSize(240, 240)
        self.ds_preview.setStyleSheet("background:#111; border:1px solid #333;")
        rl.addWidget(self.ds_preview)
        move_row = QHBoxLayout()
        move_row.addWidget(QLabel("改到"))
        self.ds_move_combo = QComboBox()
        btn_move = QPushButton("移动")
        btn_del = QPushButton("删除样本")
        btn_move.clicked.connect(self._ds_move)
        btn_del.clicked.connect(self._ds_delete)
        move_row.addWidget(self.ds_move_combo, 1)
        move_row.addWidget(btn_move)
        move_row.addWidget(btn_del)
        rl.addLayout(move_row)
        rl.addWidget(QLabel("发现标错的样本可在这里改类或删除，比翻文件夹快。"))
        rl.addStretch()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

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
        btn_open_export = QPushButton("打开导出目录")
        btn_open_export.clicked.connect(self._open_export_dir)
        exp.addWidget(btn_export)
        exp.addWidget(btn_open_export)
        exp.addStretch()
        layout.addLayout(exp)

    # ---------- helpers ----------
    def _on_tab_changed(self, index: int) -> None:
        if index == 2:
            self._reload_pending()
        elif index == 3:
            self._reload_dataset_browser()

    def _require_project(self) -> GameProject | None:
        if not self.project:
            QMessageBox.warning(self, "提示", "请先新建或打开项目")
            return None
        return self.project

    def _apply_preprocess_ui(self) -> None:
        if not self.project:
            return
        cfg = self.project.config.preprocess
        cfg.invert = self.chk_invert.isChecked()
        cfg.binarize = self.binarize_combo.currentText()

    def _persist_preprocess(self) -> None:
        if not self.project:
            return
        self._apply_preprocess_ui()
        self.project.save_config()

    # ---------- project ----------
    def _new_project(self) -> None:
        gid = self.game_id_edit.text().strip()
        if not gid:
            QMessageBox.warning(self, "提示", "请填写游戏 ID")
            return
        try:
            self.project = create_project(
                gid,
                with_symbols=self.chk_symbols.isChecked(),
                with_units=self.chk_units.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "新建失败", str(exc))
            return
        self._sync_preprocess_ui_from_project()
        self._refresh_project_info()
        self.tabs.setCurrentIndex(1)
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
        self._sync_preprocess_ui_from_project()
        self._refresh_project_info()
        self._reload_pending()
        self._reload_dataset_browser()
        self._status.showMessage(f"已打开 {self.project.root}")

    def _sync_preprocess_ui_from_project(self) -> None:
        if not self.project:
            return
        cfg = self.project.config.preprocess
        self.chk_invert.setChecked(cfg.invert)
        idx = self.binarize_combo.findText(cfg.binarize)
        if idx >= 0:
            self.binarize_combo.setCurrentIndex(idx)

    def _open_project_folder(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        import os

        os.startfile(proj.root)  # noqa: S606

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
        ckpt = latest_checkpoint(self.project)
        lines.append(f"最新模型: {ckpt.name if ckpt else '无'}")
        self.project_info.setPlainText("\n".join(lines))

    # ---------- import ----------
    def _add_units_to_project(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        added = ensure_unit_classes(proj)
        self._refresh_project_info()
        self._reload_dataset_browser()
        if added:
            QMessageBox.information(
                self,
                "已添加",
                f"已加入类别：{', '.join(display_label(x) for x in added)}\n"
                "审核台可用「万/亿」按钮或 W/Y 键标注。",
            )
        else:
            QMessageBox.information(self, "提示", "本项目已包含万/亿类别")

    def _capture_dest_dir(self) -> Path | None:
        proj = self._require_project()
        if not proj:
            return None
        proj.raw_dir.mkdir(parents=True, exist_ok=True)
        return proj.raw_dir

    def _add_captured_path(self, path: Path) -> None:
        self.import_list.addItem(str(path))
        self.import_list.setCurrentRow(self.import_list.count() - 1)
        self._status.showMessage(f"已截取: {path.name}")
        self.tabs.setCurrentIndex(1)

    def _capture_region(self) -> None:
        dest = self._capture_dest_dir()
        if not dest:
            return
        # Hide main window so it is not in the shot
        self.showMinimized()
        QApplication.processEvents()

        overlay = RegionCaptureOverlay()
        self._region_overlay = overlay  # keep ref

        def on_captured(qimg) -> None:
            self.showNormal()
            self.raise_()
            try:
                bgr = qimage_to_bgr(qimg)
                from game_digit_trainer.capture import _timestamp_name

                path = save_bgr(dest / _timestamp_name("region"), bgr)
                self._add_captured_path(path)
            except Exception as exc:
                QMessageBox.critical(self, "截屏失败", str(exc))

        def on_cancelled() -> None:
            self.showNormal()
            self.raise_()

        overlay.captured.connect(on_captured)
        overlay.cancelled.connect(on_cancelled)
        overlay.show()

    def _capture_adb(self) -> None:
        dest = self._capture_dest_dir()
        if not dest:
            return
        try:
            devices = list_adb_devices()
            path = capture_adb(dest)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "ADB 截图失败",
                f"{exc}\n\n提示：雷电设置里打开 ADB，或执行 adb connect 127.0.0.1:5555",
            )
            return
        self._add_captured_path(path)
        if devices:
            self._status.showMessage(f"ADB 截图成功（{devices[0]}）")

    def _capture_clipboard(self) -> None:
        dest = self._capture_dest_dir()
        if not dest:
            return
        try:
            bgr = capture_clipboard_bgr()
            from game_digit_trainer.capture import _timestamp_name

            path = save_bgr(dest / _timestamp_name("clip"), bgr)
        except Exception as exc:
            QMessageBox.warning(self, "粘贴失败", str(exc))
            return
        self._add_captured_path(path)

    def _clear_import_list(self) -> None:
        self.import_list.clear()
        self._current_import = None
        self.import_canvas.clear_image()

    def _pick_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择截图",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        for f in files:
            self.import_list.addItem(f)
        if files and self.import_list.count():
            self.import_list.setCurrentRow(self.import_list.count() - len(files))

    def _on_import_row(self, row: int) -> None:
        if row < 0:
            return
        self._current_import = Path(self.import_list.item(row).text())
        self.import_canvas.clear_roi()
        self._refresh_import_preview()

    def _clear_roi(self) -> None:
        self.import_canvas.clear_roi()
        self._refresh_import_preview()

    def _refresh_import_preview(self, keep_roi: bool = False) -> None:
        del keep_roi
        if not self.project or not self._current_import:
            return
        path = self._current_import
        try:
            self._apply_preprocess_ui()
            bgr = load_bgr(path)
            roi = self.import_canvas.roi()
            sliced = crop_bgr(bgr, roi)
            binary = apply_preprocess(sliced, self.project.config.preprocess)
            crops = segment_binary(binary, max_gap=self.gap_spin.value())
            ox = roi[0] if roi else 0
            oy = roi[1] if roi else 0
            boxes = [(c.x + ox, c.y + oy, c.w, c.h) for c in crops]
            if self.chk_show_binary.isChecked():
                # paint binary into full-size canvas for alignment with ROI
                canvas = np.zeros(bgr.shape[:2], dtype=np.uint8)
                if roi:
                    x, y, w, h = roi
                    bh, bw = binary.shape[:2]
                    canvas[y : y + bh, x : x + bw] = binary
                else:
                    canvas = binary
                show = canvas
            else:
                show = bgr
            self.import_canvas.set_image_bgr_or_gray(show, boxes, draw_stored_roi=True)
            self._status.showMessage(f"预览切出 {len(crops)} 个字符 · ROI={roi or '整图'}")
        except Exception as exc:
            self.import_canvas.setText(str(exc))

    def _segment_current(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if not self._current_import:
            QMessageBox.warning(self, "提示", "请先在左侧选择一张截图")
            return
        self._persist_preprocess()
        src = self._current_import
        dest_raw = proj.raw_dir / src.name
        if not dest_raw.exists():
            shutil.copy2(src, dest_raw)
        roi = self.import_canvas.roi()
        try:
            binary, crops, sliced = segment_image(
                src,
                proj.config.preprocess,
                roi=roi,
                max_gap=self.gap_spin.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "切字失败", str(exc))
            return
        if roi:
            # save roi crop for reference
            ok, buf = cv2.imencode(".png", sliced)
            if ok:
                (proj.roi_dir / f"{src.stem}_roi.png").write_bytes(buf.tobytes())
        paths = save_pending_chars(proj, src, crops)
        self.import_canvas.set_image_bgr_or_gray(
            binary if self.chk_show_binary.isChecked() else sliced,
            [(c.x, c.y, c.w, c.h) for c in crops],
            draw_stored_roi=False,
        )
        self._reload_pending()
        self._refresh_project_info()
        go = QMessageBox.question(
            self,
            "切字完成",
            f"已切出 {len(paths)} 个字符到待审核。\n是否立刻去审核？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if go == QMessageBox.StandardButton.Yes:
            self.tabs.setCurrentIndex(2)

    # ---------- review ----------
    def _reload_pending(self) -> None:
        if not self.project:
            return
        self._pending = self.project.pending_files()
        self._idx = 0
        self._show_current()

    def _show_current(self) -> None:
        self._pred_label = None
        self._pred_conf = 0.0
        if not self._pending:
            self.char_view.setText("无待审核 — 去「导入切字」添加")
            self.review_meta.setText("无待审核")
            self.pred_label_ui.setText("预测：—")
            self.btn_confirm.setEnabled(False)
            return
        self._idx = max(0, min(self._idx, len(self._pending) - 1))
        path = self._pending[self._idx]
        raw = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.char_view.setText("无法读取")
            return
        pix = numpy_to_pixmap(img)
        self.char_view.setPixmap(
            pix.scaled(
                240,
                240,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )
        self.review_meta.setText(f"{self._idx + 1}/{len(self._pending)}  {path.name}")

        # prediction
        if self.project:
            ckpt = latest_checkpoint(self.project)
            if ckpt:
                try:
                    lab, conf = predict_pending_file(
                        ckpt,
                        path,
                        self.project.config.classes,
                        self.project.config.input_width,
                        self.project.config.input_height,
                    )
                    self._pred_label = lab
                    self._pred_conf = conf
                    shown = display_label(lab)
                    self.pred_label_ui.setText(f"预测：{shown}  ({conf:.0%})")
                    color = "#1a7f37" if conf >= 0.8 else "#b36b00"
                    self.pred_label_ui.setStyleSheet(
                        f"font-size:18px; font-weight:600; color:{color};"
                    )
                    self.btn_confirm.setEnabled(True)
                    self.btn_confirm.setText(f"空格：确认「{shown}」")
                except Exception:
                    self.pred_label_ui.setText("预测：失败")
                    self.btn_confirm.setEnabled(False)
            else:
                self.pred_label_ui.setText("预测：无模型（先标一些再训练）")
                self.btn_confirm.setEnabled(False)
                self.btn_confirm.setText("空格：确认预测（需先训练）")

    def _confirm_prediction(self) -> None:
        if self._pred_label:
            self._assign(self._pred_label)

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
        self._status.showMessage(f"已标为 {display_label(normalize_label(label))}")

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

    # ---------- dataset ----------
    def _reload_dataset_browser(self) -> None:
        self.ds_class_list.clear()
        self.ds_file_list.clear()
        self.ds_move_combo.clear()
        self.ds_preview.setText("选择样本预览")
        if not self.project:
            return
        counts = self.project.class_counts()
        for name in self.project.config.classes:
            item = QListWidgetItem(f"{display_label(name)}  ({counts.get(name, 0)})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.ds_class_list.addItem(item)
            self.ds_move_combo.addItem(display_label(name), name)
        if self.ds_class_list.count():
            self.ds_class_list.setCurrentRow(0)

    def _on_ds_class(self, _text: str) -> None:
        self.ds_file_list.clear()
        if not self.project:
            return
        item = self.ds_class_list.currentItem()
        if not item:
            return
        label = item.data(Qt.ItemDataRole.UserRole)
        for p in list_dataset_files(self.project, label):
            it = QListWidgetItem(p.name)
            it.setData(Qt.ItemDataRole.UserRole, str(p))
            self.ds_file_list.addItem(it)

    def _on_ds_file(self, row: int) -> None:
        if row < 0:
            return
        item = self.ds_file_list.item(row)
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.ds_preview.setText("无法读取")
            return
        self.ds_preview.setPixmap(
            numpy_to_pixmap(img).scaled(
                220,
                220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def _current_ds_path(self) -> Path | None:
        item = self.ds_file_list.currentItem()
        if not item:
            return None
        return Path(item.data(Qt.ItemDataRole.UserRole))

    def _ds_move(self) -> None:
        proj = self._require_project()
        path = self._current_ds_path()
        if not proj or not path:
            return
        new_label = self.ds_move_combo.currentData()
        try:
            relabel_dataset_file(proj, path, new_label)
        except Exception as exc:
            QMessageBox.warning(self, "移动失败", str(exc))
            return
        self._reload_dataset_browser()
        self._refresh_project_info()

    def _ds_delete(self) -> None:
        path = self._current_ds_path()
        if not path:
            return
        path.unlink(missing_ok=True)
        self._reload_dataset_browser()
        self._refresh_project_info()

    # ---------- train / export ----------
    def _start_train(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "提示", "训练进行中")
            return
        counts = proj.class_counts()
        total = sum(counts.values())
        if total < 10:
            QMessageBox.warning(self, "提示", f"样本太少（{total}），建议每类至少几十张再训")
            return
        empty = [k for k, v in counts.items() if v == 0]
        if empty:
            QMessageBox.warning(
                self,
                "提示",
                f"这些类还没有样本：{', '.join(display_label(x) for x in empty)}\n"
                "可以先去掉未用符号类，或继续标注。仍将训练已有类。",
            )
        self.train_log.clear()
        self._worker = TrainWorker(proj, self.epochs_spin.value())
        self._worker.log.connect(lambda m: self.train_log.append(m))
        self._worker.done.connect(self._train_done)
        self._worker.failed.connect(lambda e: QMessageBox.critical(self, "训练失败", e))
        self._worker.start()

    def _train_done(self, path: str) -> None:
        self.train_log.append(f"完成: {path}")
        self._status.showMessage("训练完成 — 审核台可用空格快速确认预测")
        self._refresh_project_info()
        QMessageBox.information(self, "完成", f"最佳模型:\n{path}\n\n回到审核台可用空格确认预测。")

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

    def _open_export_dir(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        import os

        proj.exports_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(proj.exports_dir)  # noqa: S606


def run_gui() -> int:
    import sys

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()
