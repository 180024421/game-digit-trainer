from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
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
from game_digit_trainer.gui.theme import APP_QSS
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
    crops_from_full_boxes,
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
    """三步工作台：截图切字 → 审核标注 → 训练导出。"""

    TAB_WORK = 0
    TAB_REVIEW = 1
    TAB_TRAIN = 2

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("游戏数字训练站")
        self.resize(1280, 860)
        self.project: GameProject | None = None
        self._pending: list[Path] = []
        self._idx = 0
        self._worker: TrainWorker | None = None
        self._pred_label: str | None = None
        self._pred_conf: float = 0.0
        self._current_import: Path | None = None
        self._region_overlay = None

        root = QWidget()
        self.setCentralWidget(root)
        root_l = QVBoxLayout(root)
        root_l.setContentsMargins(12, 12, 12, 8)
        root_l.setSpacing(10)

        root_l.addWidget(self._build_top_bar())

        self.tabs = QTabWidget()
        self.tab_work = QWidget()
        self.tab_review = QWidget()
        self.tab_train = QWidget()
        self.tabs.addTab(self.tab_work, "① 截图切字")
        self.tabs.addTab(self.tab_review, "② 审核标注")
        self.tabs.addTab(self.tab_train, "③ 训练导出")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root_l.addWidget(self.tabs, 1)

        self._build_work()
        self._build_review()
        self._build_train()
        self._install_global_shortcuts()

        self._status = self.statusBar()
        self._status.showMessage("先打开或新建项目，再用「框选截屏」抓取游戏数字区域")

        QTimer.singleShot(0, self._try_autoload_last_project)

    # ---------- top bar / project ----------
    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 10, 12, 10)

        title = QLabel("数字训练站")
        title.setObjectName("titleLabel")
        lay.addWidget(title)

        self.project_title = QLabel("未打开项目")
        self.project_title.setStyleSheet("color:#6b7280; margin-left:8px;")
        lay.addWidget(self.project_title, 1)

        self.badge_pending = QLabel("待审 0")
        self.badge_pending.setObjectName("badge")
        self.badge_pending.setVisible(False)
        lay.addWidget(self.badge_pending)

        self.game_id_edit = QLineEdit()
        self.game_id_edit.setPlaceholderText("新项目名")
        self.game_id_edit.setMaximumWidth(140)
        lay.addWidget(self.game_id_edit)

        self.chk_units = QCheckBox("万/亿")
        self.chk_units.setChecked(True)
        self.chk_symbols = QCheckBox(",/%:")
        lay.addWidget(self.chk_units)
        lay.addWidget(self.chk_symbols)

        btn_new = QPushButton("新建")
        btn_open = QPushButton("打开…")
        btn_folder = QPushButton("文件夹")
        btn_units = QPushButton("+万/亿")
        btn_new.setObjectName("primaryBtn")
        btn_new.clicked.connect(self._new_project)
        btn_open.clicked.connect(self._open_project_dialog)
        btn_folder.clicked.connect(self._open_project_folder)
        btn_units.clicked.connect(self._add_units_to_project)
        lay.addWidget(btn_new)
        lay.addWidget(btn_open)
        lay.addWidget(btn_folder)
        lay.addWidget(btn_units)
        return bar

    def _build_work(self) -> None:
        layout = QVBoxLayout(self.tab_work)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Step 1 capture
        cap = QHBoxLayout()
        step1 = QLabel("第一步：截取画面")
        step1.setObjectName("stepLabel")
        cap.addWidget(step1)
        btn_region = QPushButton("框选截屏  F2")
        btn_region.setObjectName("primaryBtn")
        btn_adb = QPushButton("ADB 截图")
        btn_paste = QPushButton("粘贴")
        btn_pick = QPushButton("打开文件…")
        btn_region.clicked.connect(self._capture_region)
        btn_adb.clicked.connect(self._capture_adb)
        btn_paste.clicked.connect(self._capture_clipboard)
        btn_pick.clicked.connect(self._pick_images)
        cap.addWidget(btn_region)
        cap.addWidget(btn_adb)
        cap.addWidget(btn_paste)
        cap.addWidget(btn_pick)
        cap.addStretch()
        layout.addLayout(cap)

        splitter = QSplitter()
        layout.addWidget(splitter, 1)

        # Main canvas
        mid = QWidget()
        mid_l = QVBoxLayout(mid)
        mid_l.setContentsMargins(0, 0, 0, 0)

        mode_row = QHBoxLayout()
        step2 = QLabel("第二步：切字")
        step2.setObjectName("stepLabel")
        mode_row.addWidget(step2)
        self.btn_mode_roi = QPushButton("① 框整行数字")
        self.btn_mode_char = QPushButton("② 手动逐字框（推荐）")
        self.btn_mode_char.setObjectName("primaryBtn")
        self.btn_mode_roi.clicked.connect(lambda: self._set_cut_mode("roi"))
        self.btn_mode_char.clicked.connect(lambda: self._set_cut_mode("char"))
        mode_row.addWidget(self.btn_mode_roi)
        mode_row.addWidget(self.btn_mode_char)
        mode_row.addStretch()
        mid_l.addLayout(mode_row)

        self.work_hint = QLabel(
            "推荐：点「手动逐字框」→ 每个数字拖一个绿框（如 2 : 0 3 共 4 个）→ 再点绿色切字"
        )
        self.work_hint.setObjectName("hintLabel")
        self.work_hint.setWordWrap(True)
        mid_l.addWidget(self.work_hint)

        self.import_canvas = ImageCanvas()
        self.import_canvas.setMinimumHeight(420)
        self.import_canvas.roi_changed.connect(self._on_roi_changed)
        self.import_canvas.boxes_changed.connect(self._on_boxes_changed)
        mid_l.addWidget(self.import_canvas, 1)

        tools = QHBoxLayout()
        self.chk_show_binary = QCheckBox("看二值图")
        self.chk_show_binary.setChecked(False)
        self.chk_show_binary.stateChanged.connect(lambda _: self._refresh_import_preview())
        self.chk_invert = QCheckBox("反色")
        self.chk_invert.stateChanged.connect(lambda _: self._refresh_import_preview())
        self.binarize_combo = QComboBox()
        self.binarize_combo.addItems(["otsu", "adaptive", "none"])
        self.binarize_combo.currentTextChanged.connect(lambda _: self._refresh_import_preview())
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(1, 20)
        self.gap_spin.setValue(3)
        self.gap_spin.setToolTip("仅「框整行+自动切」时用：粘连调大，切碎调小")
        self.gap_spin.valueChanged.connect(lambda _: self._refresh_import_preview())
        tools.addWidget(self.chk_show_binary)
        tools.addWidget(self.chk_invert)
        tools.addWidget(QLabel("二值化"))
        tools.addWidget(self.binarize_combo)
        tools.addWidget(QLabel("自动间距"))
        tools.addWidget(self.gap_spin)

        btn_auto = QPushButton("自动预览切字")
        btn_auto.setToolTip("在蓝框区域内自动生成绿框（可再改成手动微调）")
        btn_auto.clicked.connect(self._auto_preview_boxes)
        btn_undo = QPushButton("撤销一字")
        btn_undo.clicked.connect(self._undo_char_box)
        btn_clear_boxes = QPushButton("清空字框")
        btn_clear_boxes.clicked.connect(self._clear_char_boxes)
        btn_clear_roi = QPushButton("取消蓝框")
        btn_clear_roi.clicked.connect(self._clear_roi)
        self.crop_count_label = QLabel("字框 0")
        self.crop_count_label.setObjectName("hintLabel")
        btn_seg = QPushButton("确认切字 → 审核")
        btn_seg.setObjectName("successBtn")
        btn_seg.setToolTip("把当前绿框切成单字样本，进入审核标注")
        btn_seg.clicked.connect(self._segment_current)
        tools.addWidget(btn_auto)
        tools.addWidget(btn_undo)
        tools.addWidget(btn_clear_boxes)
        tools.addWidget(btn_clear_roi)
        tools.addStretch()
        tools.addWidget(self.crop_count_label)
        tools.addWidget(btn_seg)
        mid_l.addLayout(tools)
        splitter.addWidget(mid)

        # default to manual char cutting — more reliable for game fonts
        self._set_cut_mode("char")

        # History
        right = QWidget()
        right.setMaximumWidth(220)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("最近截图"))
        self.import_list = QListWidget()
        self.import_list.currentRowChanged.connect(self._on_import_row)
        rl.addWidget(self.import_list, 1)
        btn_clear = QPushButton("清空列表")
        btn_clear.clicked.connect(self._clear_import_list)
        rl.addWidget(btn_clear)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)

    def _build_review(self) -> None:
        layout = QHBoxLayout(self.tab_review)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        left = QVBoxLayout()
        head = QHBoxLayout()
        self.review_meta = QLabel("无待审核")
        self.review_meta.setObjectName("titleLabel")
        btn_reload = QPushButton("刷新")
        btn_reload.clicked.connect(self._reload_pending)
        head.addWidget(self.review_meta, 1)
        head.addWidget(btn_reload)
        left.addLayout(head)

        self.char_view = QLabel("先去「截图切字」")
        self.char_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.char_view.setMinimumSize(320, 320)
        self.char_view.setStyleSheet(
            "background:#111827; color:#9ca3af; border-radius:12px; font-size:16px;"
        )
        left.addWidget(self.char_view, 1)

        nav = QHBoxLayout()
        btn_prev = QPushButton("← 上一张")
        btn_next = QPushButton("下一张 →")
        btn_prev.clicked.connect(self._prev_pending)
        btn_next.clicked.connect(self._next_pending)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        left.addLayout(nav)
        layout.addLayout(left, 3)

        right = QVBoxLayout()
        self.pred_label_ui = QLabel("预测：—")
        self.pred_label_ui.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_label_ui.setStyleSheet("font-size:22px; font-weight:700; color:#059669;")
        right.addWidget(self.pred_label_ui)

        self.btn_confirm = QPushButton("空格：确认预测")
        self.btn_confirm.setObjectName("successBtn")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._confirm_prediction)
        right.addWidget(self.btn_confirm)

        right.addWidget(QLabel("点数字键标注（或键盘 0-9）"))
        grid = QGridLayout()
        grid.setSpacing(8)
        for i in range(10):
            b = QPushButton(str(i))
            b.setObjectName("digitBtn")
            b.clicked.connect(lambda _=False, d=str(i): self._assign(d))
            grid.addWidget(b, i // 5, i % 5)
        right.addLayout(grid)

        right.addWidget(QLabel("单位 / 符号"))
        unit_row = QHBoxLayout()
        for text, lab in [("万 W", "万"), ("亿 Y", "亿"), (",", ","), ("/", "/"), ("%", "%"), (":", ":")]:
            b = QPushButton(text)
            b.setObjectName("unitBtn")
            b.clicked.connect(lambda _=False, d=lab: self._assign(d))
            unit_row.addWidget(b)
        right.addLayout(unit_row)

        act = QHBoxLayout()
        btn_del = QPushButton("删除 Del")
        btn_del.setObjectName("dangerBtn")
        btn_skip = QPushButton("跳过")
        btn_del.clicked.connect(self._delete_current)
        btn_skip.clicked.connect(self._next_pending)
        act.addWidget(btn_del)
        act.addWidget(btn_skip)
        right.addLayout(act)

        tip = QLabel("快捷键：0-9 · W万 · Y亿 · 空格确认 · ←→翻页 · Delete删除")
        tip.setObjectName("hintLabel")
        tip.setWordWrap(True)
        right.addWidget(tip)
        right.addStretch()
        layout.addLayout(right, 2)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self.tab_review, activated=self._confirm_prediction)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self.tab_review, activated=self._delete_current)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.tab_review, activated=self._prev_pending)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self.tab_review, activated=self._next_pending)
        QShortcut(QKeySequence("W"), self.tab_review, activated=lambda: self._assign("万"))
        QShortcut(QKeySequence("Y"), self.tab_review, activated=lambda: self._assign("亿"))
        for d in "0123456789":
            QShortcut(QKeySequence(d), self.tab_review, activated=lambda d=d: self._assign(d))

    def _build_train(self) -> None:
        layout = QHBoxLayout(self.tab_train)
        layout.setContentsMargins(12, 12, 12, 12)

        left = QVBoxLayout()
        self.project_info = QTextEdit()
        self.project_info.setReadOnly(True)
        self.project_info.setMaximumHeight(180)
        left.addWidget(QLabel("项目概况"))
        left.addWidget(self.project_info)

        box = QGroupBox("训练")
        form = QHBoxLayout(box)
        form.addWidget(QLabel("轮数"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(15)
        form.addWidget(self.epochs_spin)
        btn_train = QPushButton("开始训练")
        btn_train.setObjectName("primaryBtn")
        btn_train.clicked.connect(self._start_train)
        form.addWidget(btn_train)
        form.addStretch()
        left.addWidget(box)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        left.addWidget(self.train_log, 1)

        exp = QHBoxLayout()
        btn_export = QPushButton("导出 ONNX")
        btn_export.setObjectName("successBtn")
        btn_export.clicked.connect(self._export)
        btn_open_export = QPushButton("打开导出目录")
        btn_open_export.clicked.connect(self._open_export_dir)
        exp.addWidget(btn_export)
        exp.addWidget(btn_open_export)
        exp.addStretch()
        left.addLayout(exp)
        layout.addLayout(left, 2)

        # dataset browser embedded
        right = QGroupBox("样本库（改错/删除）")
        rl = QVBoxLayout(right)
        row = QHBoxLayout()
        self.ds_class_list = QListWidget()
        self.ds_class_list.setMaximumWidth(120)
        self.ds_class_list.currentTextChanged.connect(self._on_ds_class)
        self.ds_file_list = QListWidget()
        self.ds_file_list.currentRowChanged.connect(self._on_ds_file)
        row.addWidget(self.ds_class_list)
        row.addWidget(self.ds_file_list, 1)
        rl.addLayout(row, 1)

        self.ds_preview = QLabel("选样本")
        self.ds_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ds_preview.setFixedHeight(140)
        self.ds_preview.setStyleSheet("background:#111827; color:#9ca3af; border-radius:8px;")
        rl.addWidget(self.ds_preview)

        move_row = QHBoxLayout()
        self.ds_move_combo = QComboBox()
        btn_move = QPushButton("改到此类")
        btn_del = QPushButton("删除")
        btn_del.setObjectName("dangerBtn")
        btn_move.clicked.connect(self._ds_move)
        btn_del.clicked.connect(self._ds_delete)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._reload_dataset_browser)
        move_row.addWidget(self.ds_move_combo, 1)
        move_row.addWidget(btn_move)
        move_row.addWidget(btn_del)
        move_row.addWidget(btn_refresh)
        rl.addLayout(move_row)
        layout.addWidget(right, 2)

    def _install_global_shortcuts(self) -> None:
        QShortcut(QKeySequence("F2"), self, activated=self._capture_region)

    # ---------- helpers ----------
    def _on_tab_changed(self, index: int) -> None:
        if index == self.TAB_REVIEW:
            self._reload_pending()
        elif index == self.TAB_TRAIN:
            self._refresh_project_info()
            self._reload_dataset_browser()

    def _require_project(self) -> GameProject | None:
        if not self.project:
            QMessageBox.warning(self, "提示", "请先在顶部新建或打开项目")
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

    def _update_header(self) -> None:
        if not self.project:
            self.project_title.setText("未打开项目")
            self.badge_pending.setVisible(False)
            return
        n = len(self.project.pending_files())
        self.project_title.setText(f"项目：{self.project.config.game_id}")
        self.badge_pending.setText(f"待审 {n}")
        self.badge_pending.setVisible(n > 0)
        # tab title badge-ish
        self.tabs.setTabText(self.TAB_REVIEW, f"② 审核标注（{n}）" if n else "② 审核标注")

    def _try_autoload_last_project(self) -> None:
        root = projects_root()
        candidates = sorted(
            [p for p in root.iterdir() if p.is_dir() and (p / "config.json").is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return
        try:
            self.project = open_project(candidates[0])
            self.game_id_edit.setText(self.project.config.game_id)
            self._sync_preprocess_ui_from_project()
            self._refresh_project_info()
            self._update_header()
            self._status.showMessage(f"已自动打开最近项目：{self.project.config.game_id}")
        except Exception:
            pass

    # ---------- project ----------
    def _new_project(self) -> None:
        gid = self.game_id_edit.text().strip()
        if not gid:
            QMessageBox.warning(self, "提示", "请填写项目名")
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
        self._update_header()
        self.tabs.setCurrentIndex(self.TAB_WORK)
        self._status.showMessage(f"已创建，直接点「框选截屏」开始")

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
        self._update_header()
        self.tabs.setCurrentIndex(self.TAB_WORK)
        self._status.showMessage(f"已打开 {self.project.config.game_id}")

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
        os.startfile(proj.root)  # noqa: S606

    def _refresh_project_info(self) -> None:
        if not self.project:
            self.project_info.setPlainText("")
            return
        counts = self.project.class_counts()
        total = sum(counts.values())
        lines = [
            f"路径: {self.project.root}",
            f"类别: {' '.join(display_label(c) for c in self.project.config.classes)}",
            f"已标注样本: {total} · 待审核: {len(self.project.pending_files())}",
            "",
        ]
        for k, v in counts.items():
            if v:
                lines.append(f"  {display_label(k)}: {v}")
        ckpt = latest_checkpoint(self.project)
        lines.append(f"模型: {ckpt.parent.name if ckpt else '尚未训练'}")
        self.project_info.setPlainText("\n".join(lines))
        self._update_header()

    def _add_units_to_project(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        added = ensure_unit_classes(proj)
        self._refresh_project_info()
        self._reload_dataset_browser()
        if added:
            QMessageBox.information(self, "已添加", "已加入「万 / 亿」，审核时用 W / Y 标注")
        else:
            QMessageBox.information(self, "提示", "本项目已有万/亿")

    # ---------- capture / import ----------
    def _capture_dest_dir(self) -> Path | None:
        proj = self._require_project()
        if not proj:
            return None
        proj.raw_dir.mkdir(parents=True, exist_ok=True)
        return proj.raw_dir

    def _add_captured_path(self, path: Path) -> None:
        self.import_list.addItem(str(path))
        self.import_list.setCurrentRow(self.import_list.count() - 1)
        self.tabs.setCurrentIndex(self.TAB_WORK)
        self._status.showMessage(f"已截取 {path.name} — 请在图上框选数字")

    def _capture_region(self) -> None:
        dest = self._capture_dest_dir()
        if not dest:
            return
        self.showMinimized()
        QApplication.processEvents()

        def start_overlay() -> None:
            overlay = RegionCaptureOverlay()
            self._region_overlay = overlay

            def on_captured(qimg) -> None:
                self.showNormal()
                self.raise_()
                self.activateWindow()
                try:
                    from game_digit_trainer.capture import _timestamp_name

                    bgr = qimage_to_bgr(qimg)
                    path = save_bgr(dest / _timestamp_name("region"), bgr)
                    self._add_captured_path(path)
                except Exception as exc:
                    QMessageBox.critical(self, "截屏失败", str(exc))

            def on_cancelled() -> None:
                self.showNormal()
                self.raise_()
                self.activateWindow()

            overlay.captured.connect(on_captured)
            overlay.cancelled.connect(on_cancelled)
            overlay.show()

        # brief delay so minimize finishes before grab
        QTimer.singleShot(180, start_overlay)

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
                f"{exc}\n\n雷电：设置 → 打开 ADB，或 adb connect 127.0.0.1:5555",
            )
            return
        self._add_captured_path(path)
        if devices:
            self._status.showMessage(f"ADB 截图成功（{devices[0]}）— 请框选数字")

    def _capture_clipboard(self) -> None:
        dest = self._capture_dest_dir()
        if not dest:
            return
        try:
            from game_digit_trainer.capture import _timestamp_name

            bgr = capture_clipboard_bgr()
            path = save_bgr(dest / _timestamp_name("clip"), bgr)
        except Exception as exc:
            QMessageBox.warning(self, "粘贴失败", str(exc))
            return
        self._add_captured_path(path)

    def _clear_import_list(self) -> None:
        self.import_list.clear()
        self._current_import = None
        self.import_canvas.clear_image()
        self.crop_count_label.setText("字框 0")

    def _pick_images(self) -> None:
        if not self._require_project():
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择截图", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        for f in files:
            self.import_list.addItem(f)
        if files:
            self.import_list.setCurrentRow(self.import_list.count() - len(files))

    def _on_import_row(self, row: int) -> None:
        if row < 0:
            return
        self._current_import = Path(self.import_list.item(row).text())
        self.import_canvas.clear_roi()
        self.import_canvas.clear_boxes()
        self._refresh_import_preview()

    def _set_cut_mode(self, mode: str) -> None:
        self.import_canvas.set_mode(mode)
        if mode == "char":
            self.work_hint.setText(
                "手动切字：每个字拖一个绿框（例如 2、:、0、3 共 4 个框）→ 点「确认切字 → 审核」"
            )
            self.btn_mode_char.setObjectName("primaryBtn")
            self.btn_mode_roi.setObjectName("")
        else:
            self.work_hint.setText(
                "区域模式：先拖蓝框框住整行数字 → 点「自动预览切字」看绿框 → 再确认切字"
            )
            self.btn_mode_roi.setObjectName("primaryBtn")
            self.btn_mode_char.setObjectName("")
        # refresh button styles
        self.btn_mode_char.style().unpolish(self.btn_mode_char)
        self.btn_mode_char.style().polish(self.btn_mode_char)
        self.btn_mode_roi.style().unpolish(self.btn_mode_roi)
        self.btn_mode_roi.style().polish(self.btn_mode_roi)
        self._update_box_count()

    def _on_roi_changed(self, _roi) -> None:
        # region mode: auto preview after drawing blue box
        if self.import_canvas.mode() == ImageCanvas.MODE_ROI:
            self._auto_preview_boxes()
        else:
            self._refresh_import_preview()

    def _on_boxes_changed(self, boxes: list) -> None:
        self.crop_count_label.setText(f"字框 {len(boxes)}")
        self._status.showMessage(f"已手动框 {len(boxes)} 个字 — 框完后点「确认切字 → 审核」")

    def _update_box_count(self) -> None:
        n = len(self.import_canvas.boxes())
        self.crop_count_label.setText(f"字框 {n}")

    def _undo_char_box(self) -> None:
        self.import_canvas.undo_box()
        self._update_box_count()

    def _clear_char_boxes(self) -> None:
        self.import_canvas.clear_boxes()
        self._update_box_count()
        self._refresh_import_preview()

    def _clear_roi(self) -> None:
        self.import_canvas.clear_roi()
        if self.import_canvas.mode() == ImageCanvas.MODE_ROI:
            self.import_canvas.clear_boxes()
        self._refresh_import_preview()

    def _auto_preview_boxes(self) -> None:
        """在蓝框（或整图）内自动生成绿字框。"""
        if not self.project or not self._current_import:
            return
        try:
            self._apply_preprocess_ui()
            bgr = load_bgr(self._current_import)
            roi = self.import_canvas.roi()
            sliced = crop_bgr(bgr, roi)
            binary = apply_preprocess(sliced, self.project.config.preprocess)
            crops = segment_binary(binary, max_gap=self.gap_spin.value())
            ox = roi[0] if roi else 0
            oy = roi[1] if roi else 0
            boxes = [(c.x + ox, c.y + oy, c.w, c.h) for c in crops]
            self.import_canvas.set_boxes(boxes)
            self._refresh_import_preview(keep_manual_boxes=True)
            self._status.showMessage(f"自动预览 {len(boxes)} 个字框 — 不对就改「手动逐字框」")
        except Exception as exc:
            QMessageBox.warning(self, "自动切字失败", str(exc))

    def _refresh_import_preview(self, keep_manual_boxes: bool = False) -> None:
        del keep_manual_boxes
        if not self.project or not self._current_import:
            return
        path = self._current_import
        try:
            self._apply_preprocess_ui()
            bgr = load_bgr(path)
            roi = self.import_canvas.roi()
            if self.chk_show_binary.isChecked():
                if roi:
                    sliced = crop_bgr(bgr, roi)
                    binary = apply_preprocess(sliced, self.project.config.preprocess)
                    canvas = np.zeros(bgr.shape[:2], dtype=np.uint8)
                    x, y, w, h = roi
                    bh, bw = binary.shape[:2]
                    canvas[y : y + bh, x : x + bw] = binary
                    show = canvas
                else:
                    show = apply_preprocess(bgr, self.project.config.preprocess)
            else:
                show = bgr
            # 只刷新底图，保留已有绿字框
            self.import_canvas.set_image_bgr_or_gray(
                show, boxes=None, draw_stored_roi=True, keep_boxes=True
            )
            self._update_box_count()
        except Exception as exc:
            self.import_canvas.setText(str(exc))

    def _segment_current(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if not self._current_import:
            QMessageBox.warning(self, "提示", "请先截图或打开一张图")
            return
        self._persist_preprocess()
        src = self._current_import
        dest_raw = proj.raw_dir / src.name
        if not dest_raw.exists():
            shutil.copy2(src, dest_raw)

        boxes = self.import_canvas.boxes()
        # If no manual/auto boxes yet, try auto once
        if not boxes:
            self._auto_preview_boxes()
            boxes = self.import_canvas.boxes()
        if not boxes:
            QMessageBox.warning(
                self,
                "还没有字框",
                "请先切换到「手动逐字框」，把每个数字各框一下；\n"
                "或先「① 框整行数字」再点「自动预览切字」。",
            )
            return

        try:
            bgr = load_bgr(src)
            crops = crops_from_full_boxes(bgr, boxes, proj.config.preprocess)
        except Exception as exc:
            QMessageBox.critical(self, "切字失败", str(exc))
            return
        if not crops:
            QMessageBox.warning(self, "未切出字符", "字框无效，请重新框选")
            return

        roi = self.import_canvas.roi()
        if roi:
            sliced = crop_bgr(bgr, roi)
            ok, buf = cv2.imencode(".png", sliced)
            if ok:
                (proj.roi_dir / f"{src.stem}_roi.png").write_bytes(buf.tobytes())

        paths = save_pending_chars(proj, src, crops)
        self._reload_pending()
        self._refresh_project_info()
        self._update_header()
        self.tabs.setCurrentIndex(self.TAB_REVIEW)
        self._status.showMessage(f"已主动切出 {len(paths)} 个字，开始标注")

    # ---------- review ----------
    def _reload_pending(self) -> None:
        if not self.project:
            return
        self._pending = self.project.pending_files()
        self._idx = 0
        self._show_current()
        self._update_header()

    def _show_current(self) -> None:
        self._pred_label = None
        self._pred_conf = 0.0
        if not self._pending:
            self.char_view.setText("没有待审核\n去「截图切字」继续")
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
            pix.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        )
        self.review_meta.setText(f"{self._idx + 1} / {len(self._pending)}")

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
                    self.pred_label_ui.setText(f"预测 {shown}  ·  {conf:.0%}")
                    color = "#059669" if conf >= 0.8 else "#d97706"
                    self.pred_label_ui.setStyleSheet(
                        f"font-size:22px; font-weight:700; color:{color};"
                    )
                    self.btn_confirm.setEnabled(True)
                    self.btn_confirm.setText(f"空格：确认「{shown}」")
                except Exception:
                    self.pred_label_ui.setText("预测失败")
                    self.btn_confirm.setEnabled(False)
            else:
                self.pred_label_ui.setText("尚无模型 · 先手动标一些再训练")
                self.btn_confirm.setEnabled(False)
                self.btn_confirm.setText("空格确认（需先训练）")

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
        self._update_header()

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
        self.ds_preview.setText("选样本")
        if not self.project:
            return
        counts = self.project.class_counts()
        for name in self.project.config.classes:
            item = QListWidgetItem(f"{display_label(name)} ({counts.get(name, 0)})")
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
                120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
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
            QMessageBox.warning(self, "提示", f"样本太少（{total}），建议先多标一些")
            return
        self.train_log.clear()
        self._worker = TrainWorker(proj, self.epochs_spin.value())
        self._worker.log.connect(lambda m: self.train_log.append(m))
        self._worker.done.connect(self._train_done)
        self._worker.failed.connect(lambda e: QMessageBox.critical(self, "训练失败", e))
        self._worker.start()

    def _train_done(self, path: str) -> None:
        self.train_log.append(f"完成: {path}")
        self._refresh_project_info()
        QMessageBox.information(self, "完成", "训练完成。回审核页可用空格快速确认。")
        self.tabs.setCurrentIndex(self.TAB_REVIEW)

    def _export(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            QMessageBox.warning(self, "提示", "请先训练")
            return
        try:
            out = export_onnx(proj, ckpt)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"{exc}\n{traceback.format_exc()}")
            return
        QMessageBox.information(self, "已导出", f"{out.parent}")

    def _open_export_dir(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        proj.exports_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(proj.exports_dir)  # noqa: S606


def run_gui() -> int:
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    return app.exec()
