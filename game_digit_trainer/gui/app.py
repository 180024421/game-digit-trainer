from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QByteArray, QThread, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from game_digit_trainer.backup import backup_project, export_labels_pack, import_labels_pack
from game_digit_trainer.balance import balance_warnings, boost_scarce_classes, format_balance_text, scarce_classes
from game_digit_trainer.box_ops import auto_fix_boxes
from game_digit_trainer.confusion import compute_confusion, format_confusion_text
from game_digit_trainer.gold import compare_preds, tokenize_expected
from game_digit_trainer.hard_examples import add_hard_example, list_hard_files, load_hard_index, remove_hard_file
from game_digit_trainer.quality import export_quality_report, verify_onnx_runtime
from game_digit_trainer.regression import add_regression_case, load_cases, run_line_regression, run_regression
from game_digit_trainer.studio_pack import copy_exports_to_studio, copy_line_exports_to_studio
from game_digit_trainer.export_line_onnx import export_line_onnx
from game_digit_trainer.templates import resolve_template, template_choices
from game_digit_trainer.window_capture import capture_window_by_title

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
from game_digit_trainer.gui.ui_prefs import load_prefs, update_prefs
from game_digit_trainer.gui.widgets import ImageCanvas, numpy_to_pixmap
from game_digit_trainer.gui.train_worker import TrainWorker
from game_digit_trainer.gui.work_tab import WorkTabMixin
from game_digit_trainer.gui.review_tab import ReviewTabMixin
from game_digit_trainer.gui.train_tab import TrainTabMixin
from game_digit_trainer.labels import display_label, normalize_label
from game_digit_trainer.predict import (
    ModelRef,
    check_onnx_dependency,
    check_onnxruntime_dependency,
    list_project_models,
    predict_boxes_string,
    predict_boxes_with_model,
    predict_pending_file,
    resolve_onnx_pack,
    score_pending_files,
)
from game_digit_trainer.preprocess import apply_preprocess, load_bgr
from game_digit_trainer.project import (
    GameProject,
    RoiPreset,
    SegmentPreset,
    create_project,
    ensure_dot_class,
    ensure_unit_classes,
    open_project,
    projects_root,
)
from game_digit_trainer.sample_meta import get_meta, resolve_source
from game_digit_trainer.segment import (
    crop_bgr,
    crops_from_full_boxes,
    list_all_labeled,
    list_dataset_files,
    move_to_label,
    move_to_pending,
    relabel_dataset_file,
    resolve_recognize_boxes,
    save_pending_chars,
    segment_binary,
    segment_image,
)
from game_digit_trainer.train import train_project
from game_digit_trainer.train_line import latest_line_checkpoint, train_line_project
from game_digit_trainer.predict_line import predict_line_roi
from game_digit_trainer.line_data import (
    LINE_DATASET_KEY,
    clear_line_samples,
    confirm_line_pending,
    count_line_labeled,
    delete_line_sample,
    list_line_labeled,
    list_line_pending,
    save_line_pending,
    save_line_sample,
    update_line_label,
)


class MainWindow(WorkTabMixin, ReviewTabMixin, TrainTabMixin, QMainWindow):
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
        self._review_mode = "pending"  # pending | labeled | hard | line
        self._labeled: list[tuple[Path, str]] = []
        self._labeled_idx = 0
        self._line_pending: list[Path] = []
        self._line_idx = 0
        self._undo_stack: list[dict] = []
        self._last_labeled_path: Path | None = None
        self._batch_confirming = False
        self._last_roi: tuple[int, int, int, int] | None = None
        self._pending_apply_roi = False
        self._hard: list[Path] = []
        self._hard_idx = 0
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._preview_recognize_silent)
        self._digit_btns: dict[str, QPushButton] = {}
        self._unit_btns: dict[str, QPushButton] = {}
        self._ui_prefs = load_prefs()
        self._guide_step = 0
        self._pending_scores: dict[str, tuple[str, float]] = {}
        self._sort_pending_by_conf = True
        self._chk_batch_same = None  # set in _build_review

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

        QTimer.singleShot(0, self._bootstrap_ui_prefs)

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

        self.badge_pending = QPushButton("待审 0")
        self.badge_pending.setObjectName("badgeBtn")
        self.badge_pending.setCursor(Qt.CursorShape.PointingHandCursor)
        self.badge_pending.setVisible(False)
        self.badge_pending.clicked.connect(self._goto_review)
        lay.addWidget(self.badge_pending)

        self.game_id_edit = QLineEdit()
        self.game_id_edit.setPlaceholderText("新项目名")
        self.game_id_edit.setMaximumWidth(140)
        lay.addWidget(self.game_id_edit)

        self.template_combo = QComboBox()
        self.template_combo.setToolTip("新建项目时的类别模板")
        for key, lab in template_choices():
            self.template_combo.addItem(lab, key)
        idx = self.template_combo.findData("coins")
        if idx >= 0:
            self.template_combo.setCurrentIndex(idx)
        lay.addWidget(self.template_combo)
        self.chk_units = QCheckBox("万/亿")
        self.chk_units.setChecked(True)
        self.chk_symbols = QCheckBox(".,/%:")
        self.chk_symbols.setToolTip("新建时启用小数点、逗号、斜杠、百分号、冒号")
        lay.addWidget(self.chk_units)
        lay.addWidget(self.chk_symbols)

        btn_new = QPushButton("新建")
        btn_open = QPushButton("打开…")
        btn_folder = QPushButton("文件夹")
        btn_units = QPushButton("+万/亿")
        btn_dot = QPushButton("+小数点")
        btn_dot.setToolTip("给当前项目追加小数点「.」类别（已有工程缺小数点时点这个）")
        btn_new.setObjectName("primaryBtn")
        btn_new.clicked.connect(self._new_project)
        btn_open.clicked.connect(self._open_project_dialog)
        btn_folder.clicked.connect(self._open_project_folder)
        btn_units.clicked.connect(self._add_units_to_project)
        btn_dot.clicked.connect(self._add_dot_to_project)
        lay.addWidget(btn_new)
        lay.addWidget(btn_open)
        lay.addWidget(btn_folder)
        lay.addWidget(btn_units)
        lay.addWidget(btn_dot)
        return bar

    def _install_global_shortcuts(self) -> None:
        QShortcut(QKeySequence("F2"), self, activated=self._capture_region)

    # ---------- helpers ----------
    def _on_tab_changed(self, index: int) -> None:
        if index == self.TAB_REVIEW:
            self._reload_review_lists()
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
        if hasattr(self, "chk_color_filter") and self.chk_color_filter.isChecked():
            cfg.color_filter = {
                "lower": [int(self.color_h_min.value()), int(self.color_s_min.value()), 0],
                "upper": [int(self.color_h_max.value()), 255, 255],
            }
        else:
            cfg.color_filter = None

    def _on_color_filter_changed(self) -> None:
        self._apply_preprocess_ui()
        self._refresh_import_preview()
        if self.import_canvas.roi() or self.import_canvas.boxes():
            self._auto_preview_boxes()

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
        line_n = len(list_line_pending(self.project))
        labeled_n = sum(self.project.class_counts().values())
        self.project_title.setText(f"项目：{self.project.config.game_id}")
        if line_n > 0:
            self.badge_pending.setText(f"行待审 {line_n}（点此标注）")
            self.badge_pending.setVisible(True)
        elif n > 0:
            self.badge_pending.setText(f"待审 {n}（点此标注）")
            self.badge_pending.setVisible(True)
        elif labeled_n > 0:
            self.badge_pending.setText(f"已标 {labeled_n}（可点此改标）")
            self.badge_pending.setVisible(True)
        else:
            self.badge_pending.setVisible(False)
        tab = "② 审核标注"
        if line_n:
            tab = f"② 审核标注（行待审{line_n}）"
        elif n:
            tab = f"② 审核标注（待审{n}）"
        elif labeled_n:
            tab = f"② 审核标注（已标{labeled_n}）"
        self.tabs.setTabText(self.TAB_REVIEW, tab)

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
            self._reload_verify_models()
            self._status.showMessage(f"已自动打开最近项目：{self.project.config.game_id}")
        except Exception:
            pass

    # ---------- project ----------
    def _new_project(self) -> None:
        gid = self.game_id_edit.text().strip()
        if not gid:
            QMessageBox.warning(self, "提示", "请填写项目名")
            return
        key = self.template_combo.currentData() or "coins"
        with_sym, with_units, force = resolve_template(str(key))
        # 勾选可覆盖模板的 symbols/units（force 模板除外）
        if force is None:
            with_sym = self.chk_symbols.isChecked() or with_sym
            with_units = self.chk_units.isChecked() or with_units
        try:
            self.project = create_project(
                gid,
                with_symbols=with_sym,
                with_units=with_units,
                classes=force,
            )
        except Exception as exc:
            QMessageBox.critical(self, "新建失败", str(exc))
            return
        self._sync_preprocess_ui_from_project()
        self._refresh_project_info()
        self._update_header()
        self._reload_verify_models()
        self.tabs.setCurrentIndex(self.TAB_WORK)
        self._status.showMessage(f"已创建（模板：{self.template_combo.currentText()}）")

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
        self._reload_verify_models()
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
        self.conf_spin.blockSignals(True)
        self.conf_spin.setValue(float(self.project.config.confirm_threshold))
        self.conf_spin.blockSignals(False)
        self.chk_augment.blockSignals(True)
        self.chk_augment.setChecked(bool(self.project.config.augment))
        self.chk_augment.blockSignals(False)
        if hasattr(self, "gap_spin"):
            self.gap_spin.blockSignals(True)
            self.gap_spin.setValue(int(getattr(self.project.config, "last_segment_gap", 3) or 3))
            self.gap_spin.blockSignals(False)
        self._reload_roi_preset_combo()
        self._reload_segment_presets()

    def _open_project_folder(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        os.startfile(proj.root)  # noqa: S606

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

    def _add_dot_to_project(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        added = ensure_dot_class(proj)
        self._refresh_project_info()
        self._reload_dataset_browser()
        if added:
            QMessageBox.information(
                self,
                "已添加",
                "已加入小数点「.」。请补标含小数点的样本后重新训练/导出。\n审核页可点「.」或键盘句号键。",
            )
        else:
            QMessageBox.information(self, "提示", "本项目已有小数点类别")

    # ---------- capture / import ----------
    def _bootstrap_ui_prefs(self) -> None:
        prefs = self._ui_prefs
        geo = prefs.get("geometry")
        if geo:
            try:
                self.restoreGeometry(QByteArray.fromHex(str(geo).encode("ascii")))
            except Exception:
                pass
        self._try_autoload_last_project()
        mode = prefs.get("cut_mode") or "roi"
        if mode in ("char", "roi"):
            self._set_cut_mode(mode)
        last_roi = prefs.get("last_roi")
        if isinstance(last_roi, list) and len(last_roi) == 4:
            try:
                self._last_roi = tuple(int(v) for v in last_roi)  # type: ignore[assignment]
            except (TypeError, ValueError):
                pass
        if prefs.get("work_more_open") and hasattr(self, "btn_more"):
            self.btn_more.setChecked(True)
        sizes = prefs.get("work_splitter")
        if sizes and hasattr(self, "work_splitter"):
            try:
                self.work_splitter.setSizes([int(x) for x in sizes])
            except Exception:
                pass
        if not prefs.get("guide_done"):
            self._guide_step = 0
            self._update_guide_banner()
            self.guide_banner.setVisible(True)
        else:
            self.guide_banner.setVisible(False)
        self._update_export_dep_hint()
        self._reload_verify_models()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._persist_ui_prefs()
        super().closeEvent(event)

    def _persist_ui_prefs(self) -> None:
        payload = {
            "geometry": bytes(self.saveGeometry().toHex()).decode("ascii"),
            "cut_mode": self.import_canvas.mode() if hasattr(self, "import_canvas") else "roi",
            "guide_done": bool(self._ui_prefs.get("guide_done")),
            "work_more_open": bool(getattr(self, "btn_more", None) and self.btn_more.isChecked()),
        }
        if self._last_roi:
            payload["last_roi"] = list(self._last_roi)
        if hasattr(self, "work_splitter"):
            payload["work_splitter"] = self.work_splitter.sizes()
        if self.project:
            payload["last_roi_preset"] = self.roi_preset_combo.currentData()
            payload["confirm_threshold"] = float(self.conf_spin.value())
        self._ui_prefs = update_prefs(**payload)


def run_gui() -> int:
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    return app.exec()

