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
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from game_digit_trainer.backup import backup_project
from game_digit_trainer.balance import balance_warnings, boost_scarce_classes, format_balance_text, scarce_classes
from game_digit_trainer.box_ops import auto_fix_boxes
from game_digit_trainer.confusion import compute_confusion, format_confusion_text
from game_digit_trainer.gold import compare_preds, tokenize_expected
from game_digit_trainer.hard_examples import add_hard_example, list_hard_files, load_hard_index, remove_hard_file
from game_digit_trainer.quality import export_quality_report, verify_onnx_runtime
from game_digit_trainer.regression import add_regression_case, load_cases, run_regression
from game_digit_trainer.studio_pack import copy_exports_to_studio
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
    save_pending_chars,
    segment_binary,
    segment_image,
)
from game_digit_trainer.train import train_project


class TrainWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, project: GameProject, epochs: int, *, augment: bool = True) -> None:
        super().__init__()
        self.project = project
        self.epochs = epochs
        self.augment = augment

    def run(self) -> None:
        try:
            path = train_project(
                self.project,
                epochs=self.epochs,
                augment=self.augment,
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
        self._review_mode = "pending"  # pending | labeled
        self._labeled: list[tuple[Path, str]] = []
        self._labeled_idx = 0
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

        self.guide_banner = QFrame()
        self.guide_banner.setObjectName("guideBanner")
        gl = QHBoxLayout(self.guide_banner)
        gl.setContentsMargins(12, 8, 12, 8)
        self.guide_label = QLabel()
        self.guide_label.setWordWrap(True)
        self.guide_label.setObjectName("guideLabel")
        btn_guide_next = QPushButton("下一步")
        btn_guide_skip = QPushButton("跳过引导")
        btn_guide_next.clicked.connect(self._guide_next)
        btn_guide_skip.clicked.connect(self._guide_skip)
        gl.addWidget(self.guide_label, 1)
        gl.addWidget(btn_guide_next)
        gl.addWidget(btn_guide_skip)
        layout.addWidget(self.guide_banner)
        self.guide_banner.setVisible(False)

        cap = QHBoxLayout()
        step1 = QLabel("主操作")
        step1.setObjectName("stepLabel")
        cap.addWidget(step1)
        btn_region = QPushButton("框选截屏  F2")
        btn_region.setObjectName("primaryBtn")
        btn_adb = QPushButton("ADB")
        btn_win = QPushButton("雷电窗口")
        btn_win.setToolTip("按窗口标题截取（默认含「雷电」）")
        btn_win.clicked.connect(self._capture_ld_window)
        btn_recap = QPushButton("再截同 ROI")
        btn_recap.setToolTip("截一张新图并套用上次蓝框位置（适合反复刷同一 HUD）")
        btn_paste = QPushButton("粘贴")
        btn_pick = QPushButton("打开…")
        btn_region.clicked.connect(self._capture_region)
        btn_adb.clicked.connect(self._capture_adb)
        btn_recap.clicked.connect(self._recapture_same_roi)
        btn_paste.clicked.connect(self._capture_clipboard)
        btn_pick.clicked.connect(self._pick_images)
        cap.addWidget(btn_region)
        cap.addWidget(btn_adb)
        cap.addWidget(btn_win)
        cap.addWidget(btn_recap)
        cap.addWidget(btn_paste)
        cap.addWidget(btn_pick)
        cap.addStretch()
        layout.addLayout(cap)

        splitter = QSplitter()
        self.work_splitter = splitter
        layout.addWidget(splitter, 1)

        mid = QWidget()
        mid_l = QVBoxLayout(mid)
        mid_l.setContentsMargins(0, 0, 0, 0)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("框字"))
        self.btn_mode_roi = QPushButton("整行蓝框")
        self.btn_mode_char = QPushButton("逐字绿框（推荐）")
        self.btn_mode_char.setObjectName("primaryBtn")
        self.btn_mode_roi.clicked.connect(lambda: self._set_cut_mode("roi"))
        self.btn_mode_char.clicked.connect(lambda: self._set_cut_mode("char"))
        mode_row.addWidget(self.btn_mode_roi)
        mode_row.addWidget(self.btn_mode_char)
        self.crop_count_label = QLabel("字框 0")
        self.crop_count_label.setObjectName("hintLabel")
        mode_row.addWidget(self.crop_count_label)
        mode_row.addStretch()
        btn_undo = QPushButton("撤销框 Ctrl+Z")
        btn_undo.clicked.connect(self._undo_char_box)
        btn_clear_boxes = QPushButton("清空框")
        btn_clear_boxes.clicked.connect(self._clear_char_boxes)
        btn_seg = QPushButton("确认切字 Enter")
        btn_seg.setObjectName("successBtn")
        btn_seg.setToolTip("快捷键 Enter")
        btn_seg.clicked.connect(self._segment_current)
        mode_row.addWidget(btn_undo)
        mode_row.addWidget(btn_clear_boxes)
        mode_row.addWidget(btn_seg)
        mid_l.addLayout(mode_row)

        self.work_hint = QLabel(
            "拖绿框逐字框选；点选后可拖移/拖角改大小。Enter 切字 · Ctrl+Z 撤销上一框"
        )
        self.work_hint.setObjectName("hintLabel")
        self.work_hint.setWordWrap(True)
        mid_l.addWidget(self.work_hint)

        self.import_canvas = ImageCanvas()
        self.import_canvas.setMinimumHeight(280)
        self.import_canvas.roi_changed.connect(self._on_roi_changed)
        self.import_canvas.boxes_changed.connect(self._on_boxes_changed)
        self.import_canvas.view_changed.connect(self._on_view_changed)
        self.import_canvas.selection_changed.connect(self._on_box_selection_changed)
        mid_l.addWidget(self.import_canvas, 1)

        self.preview_big = QLabel("识别预览：框选数字后点「预览识别」")
        self.preview_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_big.setFixedHeight(44)
        self.preview_big.setStyleSheet(
            "background:#111827; color:#fbbf24; border-radius:10px; font-size:22px; font-weight:800; padding:4px;"
        )
        mid_l.addWidget(self.preview_big)

        # 选中字框：单行紧凑（避免再挤掉画布高度导致截图顶部被裁）
        sel_bar = QHBoxLayout()
        sel_bar.setContentsMargins(0, 0, 0, 0)
        self.selected_crop_preview = QLabel("未选")
        self.selected_crop_preview.setFixedSize(56, 56)
        self.selected_crop_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_crop_preview.setStyleSheet(
            "background:#0b1220; color:#94a3b8; border:2px solid #64748b; "
            "border-radius:6px; font-size:11px;"
        )
        self.selected_crop_preview.setToolTip("当前选中字框放大预览")
        sel_bar.addWidget(self.selected_crop_preview)
        self.selected_box_label = QLabel("点绿框选中后可改宽高")
        self.selected_box_label.setObjectName("hintLabel")
        sel_bar.addWidget(self.selected_box_label)
        sel_bar.addWidget(QLabel("宽"))
        self.box_w_spin = QSpinBox()
        self.box_w_spin.setRange(2, 400)
        self.box_w_spin.setEnabled(False)
        self.box_w_spin.setFixedWidth(64)
        self.box_w_spin.setToolTip("精确调整选中字框宽度（像素）")
        self.box_w_spin.valueChanged.connect(self._on_box_size_spin)
        sel_bar.addWidget(self.box_w_spin)
        btn_w_minus = QPushButton("窄")
        btn_w_minus.setFixedWidth(32)
        btn_w_minus.clicked.connect(lambda: self._nudge_selected_size(dw=-1))
        btn_w_plus = QPushButton("宽+")
        btn_w_plus.setFixedWidth(36)
        btn_w_plus.clicked.connect(lambda: self._nudge_selected_size(dw=1))
        sel_bar.addWidget(btn_w_minus)
        sel_bar.addWidget(btn_w_plus)
        sel_bar.addWidget(QLabel("高"))
        self.box_h_spin = QSpinBox()
        self.box_h_spin.setRange(2, 400)
        self.box_h_spin.setEnabled(False)
        self.box_h_spin.setFixedWidth(64)
        self.box_h_spin.setToolTip("精确调整选中字框高度（像素）")
        self.box_h_spin.valueChanged.connect(self._on_box_size_spin)
        sel_bar.addWidget(self.box_h_spin)
        btn_h_minus = QPushButton("矮")
        btn_h_minus.setFixedWidth(32)
        btn_h_minus.clicked.connect(lambda: self._nudge_selected_size(dh=-1))
        btn_h_plus = QPushButton("高+")
        btn_h_plus.setFixedWidth(36)
        btn_h_plus.clicked.connect(lambda: self._nudge_selected_size(dh=1))
        sel_bar.addWidget(btn_h_minus)
        sel_bar.addWidget(btn_h_plus)
        sel_bar.addStretch()
        mid_l.addLayout(sel_bar)
        self._syncing_box_spins = False

        main_tools = QHBoxLayout()
        self.btn_mode_draw = QPushButton("画框")
        self.btn_mode_pan = QPushButton("拖图")
        self.btn_mode_draw.setCheckable(True)
        self.btn_mode_pan.setCheckable(True)
        self.btn_mode_draw.setChecked(True)
        self.btn_mode_draw.setToolTip("左键拖出蓝/绿框（默认）")
        self.btn_mode_pan.setToolTip("左键拖移画面；也可按住空格临时拖图")
        self.btn_mode_draw.clicked.connect(lambda: self._set_canvas_interaction("draw"))
        self.btn_mode_pan.clicked.connect(lambda: self._set_canvas_interaction("pan"))
        main_tools.addWidget(self.btn_mode_draw)
        main_tools.addWidget(self.btn_mode_pan)
        btn_auto = QPushButton("自动预览切字")
        btn_auto.clicked.connect(self._auto_preview_boxes)
        main_tools.addWidget(QLabel("间距"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(1, 30)
        self.gap_spin.setValue(3)
        self.gap_spin.setToolTip("自动切字间距：切碎了调大，粘连了调小（改完会立刻重切）")
        self.gap_spin.valueChanged.connect(self._on_gap_changed)
        main_tools.addWidget(self.gap_spin)
        btn_preview = QPushButton("预览识别")
        btn_preview.setObjectName("primaryBtn")
        btn_preview.setToolTip("用当前工程最新 checkpoint 识别绿框（训练中快速试跑）")
        btn_preview.clicked.connect(self._preview_recognize)
        btn_split = QPushButton("拆粘连")
        btn_split.setToolTip("点选绿框后拆开；未选中时自动拆最宽的框")
        btn_split.clicked.connect(self._split_selected_box)
        main_tools.addWidget(btn_auto)
        main_tools.addWidget(btn_preview)
        main_tools.addWidget(btn_split)
        main_tools.addStretch()
        mid_l.addLayout(main_tools)

        # 验模型：常驻，可切换 / 浏览外部 ONNX（其它电脑导出的包）
        verify_box = QGroupBox("验模型（可选导出 ONNX 或本机 checkpoint）")
        verify_l = QVBoxLayout(verify_box)
        verify_row = QHBoxLayout()
        verify_row.addWidget(QLabel("模型"))
        self.verify_model_combo = QComboBox()
        self.verify_model_combo.setMinimumWidth(220)
        self.verify_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.verify_model_combo.setToolTip("列表含本工程 exports、runs，以及你浏览过的外部 ONNX 包")
        btn_verify_refresh = QPushButton("刷新")
        btn_verify_refresh.setToolTip("重新扫描本工程导出与 checkpoint")
        btn_verify_refresh.clicked.connect(self._reload_verify_models)
        btn_verify_browse = QPushButton("浏览 ONNX…")
        btn_verify_browse.setToolTip("选择其它电脑拷来的 digits.onnx 或导出目录")
        btn_verify_browse.clicked.connect(self._browse_verify_onnx)
        btn_verify_run = QPushButton("用所选模型识别")
        btn_verify_run.setObjectName("primaryBtn")
        btn_verify_run.setToolTip("对当前截图绿字框/蓝 ROI 用下拉所选模型推理")
        btn_verify_run.clicked.connect(self._verify_recognize_selected)
        verify_row.addWidget(self.verify_model_combo, 1)
        verify_row.addWidget(btn_verify_refresh)
        verify_row.addWidget(btn_verify_browse)
        verify_row.addWidget(btn_verify_run)
        verify_l.addLayout(verify_row)
        self.verify_hint = QLabel("可加载外部导出包；圈选区域后点「用所选模型识别」。")
        self.verify_hint.setObjectName("hintLabel")
        self.verify_hint.setWordWrap(True)
        verify_l.addWidget(self.verify_hint)
        mid_l.addWidget(verify_box)

        gold_row = QHBoxLayout()
        gold_row.addWidget(QLabel("金标"))
        self.gold_edit = QLineEdit()
        self.gold_edit.setPlaceholderText("例如 1234万 或 2:03 — 与预览对比，错字进难例/待审")
        btn_gold = QPushButton("对比回流")
        btn_gold.clicked.connect(self._gold_compare_reflow)
        btn_reg_add = QPushButton("加入回归集")
        btn_reg_add.setToolTip("把当前图+金标存为固定回归用例")
        btn_reg_add.clicked.connect(self._add_regression_case)
        gold_row.addWidget(self.gold_edit, 1)
        gold_row.addWidget(btn_gold)
        gold_row.addWidget(btn_reg_add)
        mid_l.addLayout(gold_row)

        self.trial_result = QLabel("预览明细：训练后可用")
        self.trial_result.setObjectName("hintLabel")
        self.trial_result.setWordWrap(True)
        mid_l.addWidget(self.trial_result)

        # 高级选项：独立一行大按钮，收起后仍在原处可再展开
        more_toggle_row = QHBoxLayout()
        self.btn_more = QPushButton("高级选项 ▾（ROI 预设 / 缩放 / 定时刷样…）")
        self.btn_more.setCheckable(True)
        self.btn_more.setChecked(False)
        self.btn_more.setMinimumHeight(36)
        self.btn_more.setToolTip("展开或收起进阶工具；按钮始终留在此行，不会消失")
        self.btn_more.toggled.connect(self._toggle_work_more)
        more_toggle_row.addWidget(self.btn_more)
        mid_l.addLayout(more_toggle_row)

        self.work_more = QWidget()
        more_l = QVBoxLayout(self.work_more)
        more_l.setContentsMargins(0, 4, 0, 0)
        zoom_row = QHBoxLayout()
        self.zoom_label = QLabel("缩放 1.0x")
        self.zoom_label.setObjectName("hintLabel")
        btn_zoom_roi = QPushButton("放大蓝框")
        btn_zoom_fit = QPushButton("适应窗口")
        btn_zoom_in = QPushButton("+")
        btn_zoom_out = QPushButton("-")
        btn_zoom_roi.clicked.connect(self._zoom_to_roi)
        btn_zoom_fit.clicked.connect(self.import_canvas.reset_view)
        btn_zoom_in.clicked.connect(lambda: self._nudge_zoom(1.25))
        btn_zoom_out.clicked.connect(lambda: self._nudge_zoom(0.8))
        zoom_row.addWidget(self.zoom_label)
        zoom_row.addWidget(btn_zoom_roi)
        zoom_row.addWidget(btn_zoom_in)
        zoom_row.addWidget(btn_zoom_out)
        zoom_row.addWidget(btn_zoom_fit)
        zoom_row.addStretch()
        more_l.addLayout(zoom_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("ROI 预设"))
        self.roi_preset_combo = QComboBox()
        self.roi_preset_combo.setMinimumWidth(140)
        btn_preset_apply = QPushButton("套用")
        btn_preset_save = QPushButton("保存蓝框")
        btn_preset_del = QPushButton("删")
        btn_preset_apply.clicked.connect(self._apply_roi_preset)
        btn_preset_save.clicked.connect(self._save_roi_preset)
        btn_preset_del.clicked.connect(self._delete_roi_preset)
        preset_row.addWidget(self.roi_preset_combo, 1)
        preset_row.addWidget(btn_preset_apply)
        preset_row.addWidget(btn_preset_save)
        preset_row.addWidget(btn_preset_del)
        more_l.addLayout(preset_row)

        tools = QHBoxLayout()
        self.chk_show_binary = QCheckBox("二值图")
        self.chk_show_binary.stateChanged.connect(lambda _: self._refresh_import_preview())
        self.chk_invert = QCheckBox("反色")
        self.chk_invert.stateChanged.connect(lambda _: self._refresh_import_preview())
        self.binarize_combo = QComboBox()
        self.binarize_combo.addItems(["otsu", "adaptive", "none"])
        self.binarize_combo.currentTextChanged.connect(lambda _: self._refresh_import_preview())
        btn_clear_roi = QPushButton("取消蓝框")
        btn_clear_roi.clicked.connect(self._clear_roi)
        tools.addWidget(self.chk_show_binary)
        tools.addWidget(self.chk_invert)
        tools.addWidget(QLabel("二值化"))
        tools.addWidget(self.binarize_combo)
        tools.addWidget(btn_clear_roi)
        tools.addStretch()
        more_l.addLayout(tools)

        prep_prev = QHBoxLayout()
        prep_prev.addWidget(QLabel("预处理预览"))
        self.preprocess_preview = QLabel("调预处理看这里")
        self.preprocess_preview.setFixedSize(120, 48)
        self.preprocess_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preprocess_preview.setStyleSheet(
            "background:#0b1220; color:#94a3b8; border:1px solid #334155; border-radius:6px; font-size:11px;"
        )
        prep_prev.addWidget(self.preprocess_preview)
        prep_prev.addStretch()
        more_l.addLayout(prep_prev)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("切字预设"))
        self.seg_preset_combo = QComboBox()
        self.seg_preset_combo.setMinimumWidth(120)
        btn_seg_apply = QPushButton("套用")
        btn_seg_save = QPushButton("保存当前")
        btn_seg_del = QPushButton("删")
        btn_seg_apply.clicked.connect(self._apply_segment_preset)
        btn_seg_save.clicked.connect(self._save_segment_preset)
        btn_seg_del.clicked.connect(self._delete_segment_preset)
        seg_row.addWidget(self.seg_preset_combo, 1)
        seg_row.addWidget(btn_seg_apply)
        seg_row.addWidget(btn_seg_save)
        seg_row.addWidget(btn_seg_del)
        more_l.addLayout(seg_row)

        extra_tools = QHBoxLayout()
        btn_merge = QPushButton("合并框")
        btn_merge.clicked.connect(self._merge_selected_box)
        btn_fix = QPushButton("修碎框")
        btn_fix.clicked.connect(self._autofix_boxes)
        btn_multi = QPushButton("多ROI刷样")
        btn_multi.clicked.connect(self._multi_roi_sample)
        extra_tools.addWidget(btn_merge)
        extra_tools.addWidget(btn_fix)
        extra_tools.addWidget(btn_multi)
        extra_tools.addStretch()
        more_l.addLayout(extra_tools)

        auto_row = QHBoxLayout()
        self.chk_auto_roi = QCheckBox("定时刷样")
        self.chk_auto_roi.setToolTip("按间隔对当前蓝框/ROI 预设自动截图切字")
        self.auto_roi_spin = QSpinBox()
        self.auto_roi_spin.setRange(3, 120)
        self.auto_roi_spin.setValue(8)
        self.auto_roi_spin.setSuffix(" 秒")
        self.chk_auto_roi.toggled.connect(self._toggle_auto_roi_sample)
        auto_row.addWidget(self.chk_auto_roi)
        auto_row.addWidget(self.auto_roi_spin)
        auto_row.addStretch()
        more_l.addLayout(auto_row)

        color_row = QHBoxLayout()
        self.chk_color_filter = QCheckBox("颜色过滤(HSV)")
        self.chk_color_filter.setToolTip("只保留指定色相范围的像素再切字")
        self.color_h_min = QSpinBox()
        self.color_h_min.setRange(0, 180)
        self.color_h_min.setValue(0)
        self.color_h_max = QSpinBox()
        self.color_h_max.setRange(0, 180)
        self.color_h_max.setValue(180)
        self.color_s_min = QSpinBox()
        self.color_s_min.setRange(0, 255)
        self.color_s_min.setValue(40)
        for w in (self.chk_color_filter, self.color_h_min, self.color_h_max, self.color_s_min):
            if w is self.chk_color_filter:
                w.stateChanged.connect(lambda _: self._on_color_filter_changed())
            else:
                w.valueChanged.connect(lambda _=0: self._on_color_filter_changed())
        color_row.addWidget(self.chk_color_filter)
        color_row.addWidget(QLabel("H"))
        color_row.addWidget(self.color_h_min)
        color_row.addWidget(QLabel("-"))
        color_row.addWidget(self.color_h_max)
        color_row.addWidget(QLabel("S≥"))
        color_row.addWidget(self.color_s_min)
        color_row.addStretch()
        more_l.addLayout(color_row)

        mid_l.addWidget(self.work_more)
        self.work_more.setVisible(False)
        self._auto_roi_timer = QTimer(self)
        self._auto_roi_timer.timeout.connect(self._auto_roi_tick)

        splitter.addWidget(mid)

        self._set_cut_mode("char")

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
        layout.setSpacing(12)

        gallery = QVBoxLayout()
        mode_row = QHBoxLayout()
        self.btn_rev_pending = QPushButton('待审')
        self.btn_rev_labeled = QPushButton('已标注（可改）')
        self.btn_rev_hard = QPushButton('难例')
        self.btn_rev_pending.setObjectName('primaryBtn')
        self.btn_rev_pending.clicked.connect(lambda: self._set_review_mode('pending'))
        self.btn_rev_labeled.clicked.connect(lambda: self._set_review_mode('labeled'))
        self.btn_rev_hard.clicked.connect(lambda: self._set_review_mode('hard'))
        mode_row.addWidget(self.btn_rev_pending)
        mode_row.addWidget(self.btn_rev_labeled)
        mode_row.addWidget(self.btn_rev_hard)
        gallery.addLayout(mode_row)

        gal_head = QHBoxLayout()
        self.gallery_title = QLabel('待审预览（点击选择）')
        btn_reload = QPushButton('刷新')
        btn_reload.clicked.connect(self._reload_review_lists)
        gal_head.addWidget(self.gallery_title)
        gal_head.addWidget(btn_reload)
        gallery.addLayout(gal_head)

        self.pending_list = QListWidget()
        self.pending_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.pending_list.setIconSize(QSize(72, 72))
        self.pending_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.pending_list.setMovement(QListWidget.Movement.Static)
        self.pending_list.setSpacing(6)
        self.pending_list.setMinimumWidth(220)
        self.pending_list.setMaximumWidth(280)
        self.pending_list.setWordWrap(True)
        self.pending_list.currentRowChanged.connect(self._on_gallery_selected)
        gallery.addWidget(self.pending_list, 1)
        layout.addLayout(gallery)

        center = QVBoxLayout()
        self.review_meta = QLabel('无待审核')
        self.review_meta.setObjectName('titleLabel')
        center.addWidget(self.review_meta)

        self.review_progress = QProgressBar()
        self.review_progress.setTextVisible(True)
        self.review_progress.setFormat('已标 %v / 共 %m')
        self.review_progress.setMinimum(0)
        self.review_progress.setMaximum(1)
        self.review_progress.setValue(0)
        center.addWidget(self.review_progress)

        self.batch_stop_label = QLabel('')
        self.batch_stop_label.setObjectName('hintLabel')
        self.batch_stop_label.setWordWrap(True)
        self.batch_stop_label.setVisible(False)
        center.addWidget(self.batch_stop_label)

        self.char_view = QLabel('左侧点选待审缩略图\n或去「截图切字」')
        self.char_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.char_view.setMinimumSize(360, 360)
        self.char_view.setStyleSheet(
            'background:#111827; color:#9ca3af; border-radius:12px; font-size:16px;'
        )
        center.addWidget(self.char_view, 1)

        self.context_view = QLabel("原图对照：切字后显示整行 + 当前字高亮")
        self.context_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.context_view.setMinimumHeight(120)
        self.context_view.setMaximumHeight(160)
        self.context_view.setStyleSheet(
            "background:#0f172a; color:#94a3b8; border-radius:10px; font-size:13px;"
        )
        center.addWidget(self.context_view)

        nav = QHBoxLayout()
        btn_prev = QPushButton('← 上一张')
        btn_next = QPushButton('下一张 →')
        btn_prev.clicked.connect(self._prev_pending)
        btn_next.clicked.connect(self._next_pending)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        center.addLayout(nav)
        layout.addLayout(center, 3)

        right = QVBoxLayout()
        self.pred_label_ui = QLabel('预测：—')
        self.pred_label_ui.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_label_ui.setStyleSheet('font-size:22px; font-weight:700; color:#059669;')
        right.addWidget(self.pred_label_ui)

        self.btn_confirm = QPushButton('空格：确认预测')
        self.btn_confirm.setObjectName('successBtn')
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._confirm_prediction)
        right.addWidget(self.btn_confirm)

        conf_row = QHBoxLayout()
        self.chk_batch_confirm = QCheckBox('批量确认（≥阈值连过）')
        self.chk_batch_confirm.setChecked(True)
        self.chk_batch_confirm.setToolTip('空格确认后，自动继续确认高置信预测，直到低于阈值')
        conf_row.addWidget(self.chk_batch_confirm)
        conf_row.addWidget(QLabel('阈值'))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.5, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.85)
        self.conf_spin.setDecimals(2)
        self.conf_spin.valueChanged.connect(self._persist_confirm_threshold)
        conf_row.addWidget(self.conf_spin)
        right.addLayout(conf_row)

        self.chk_batch_same = QCheckBox('同类批量（同预测标签连过）')
        self.chk_batch_same.setChecked(True)
        self.chk_batch_same.setToolTip('批量确认时只连过与当前预测相同的标签')
        right.addWidget(self.chk_batch_same)

        self.chk_sort_conf = QCheckBox('按置信度升序（先标难的）')
        self.chk_sort_conf.setChecked(True)
        self.chk_sort_conf.toggled.connect(self._on_sort_conf_toggled)
        right.addWidget(self.chk_sort_conf)

        btn_prelabel = QPushButton('全部预标排序')
        btn_prelabel.setToolTip('用当前模型给全部待审打分并按置信度排序')
        btn_prelabel.clicked.connect(self._prelabel_all_pending)
        right.addWidget(btn_prelabel)

        btn_jump_last = QPushButton('查看刚标的那张')
        btn_jump_last.setToolTip('跳到最近一次标注的样本，方便立刻改错')
        btn_jump_last.clicked.connect(self._jump_to_last_labeled)
        right.addWidget(btn_jump_last)

        right.addWidget(QLabel('点下面按钮标注 / 改标当前大图'))
        grid = QGridLayout()
        grid.setSpacing(8)
        self._digit_btns = {}
        for i in range(10):
            b = QPushButton(str(i))
            b.setObjectName('digitBtn')
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, d=str(i): self._assign(d))
            grid.addWidget(b, i // 5, i % 5)
            self._digit_btns[str(i)] = b
        right.addLayout(grid)

        right.addWidget(QLabel('单位 / 符号'))
        unit_row = QHBoxLayout()
        self._unit_btns = {}
        for text, lab in [('万 W', '万'), ('亿 Y', '亿'), (',', ','), ('/', '/'), ('%', '%'), (':', ':')]:
            b = QPushButton(text)
            b.setObjectName('unitBtn')
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, d=lab: self._assign(d))
            unit_row.addWidget(b)
            self._unit_btns[lab] = b
        right.addLayout(unit_row)

        act = QHBoxLayout()
        self.btn_undo_label = QPushButton('撤销上一标 Ctrl+Z')
        self.btn_undo_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_undo_label.clicked.connect(self._undo_last_label)
        btn_back = QPushButton('退回待审')
        btn_back.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_back.setToolTip('把当前已标注样本退回待审列表')
        btn_back.clicked.connect(self._return_current_to_pending)
        btn_del = QPushButton('删除 Del')
        btn_del.setObjectName('dangerBtn')
        btn_del.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_skip = QPushButton('跳过')
        btn_skip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_del.clicked.connect(self._delete_current)
        btn_skip.clicked.connect(self._next_pending)
        act.addWidget(self.btn_undo_label)
        act.addWidget(btn_back)
        act.addWidget(btn_del)
        act.addWidget(btn_skip)
        right.addLayout(act)

        tip = QLabel(
            '标错了：点「已标注（可改）」选中后重新点数字改标，或「退回待审」/「撤销上一标」。'
        )
        tip.setObjectName('hintLabel')
        tip.setWordWrap(True)
        right.addWidget(tip)
        right.addStretch()
        layout.addLayout(right, 2)

        for key, handler in [
            (Qt.Key.Key_Space, self._confirm_prediction),
            (Qt.Key.Key_Delete, self._delete_current),
            (Qt.Key.Key_Left, self._prev_pending),
            (Qt.Key.Key_Right, self._next_pending),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda h=handler: self._review_only(h))
        for d in '0123456789':
            sc = QShortcut(QKeySequence(d), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda d=d: self._review_only(lambda: self._assign(d)))
        for key, lab in [('W', '万'), ('Y', '亿')]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda lab=lab: self._review_only(lambda: self._assign(lab)))
        sc = QShortcut(QKeySequence('Ctrl+Z'), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._undo_contextual)
        sc_redo = QShortcut(QKeySequence('Ctrl+Y'), self)
        sc_redo.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_redo.activated.connect(self._redo_contextual)
        sc_enter = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        sc_enter.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_enter.activated.connect(self._enter_contextual)
        sc_enter2 = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        sc_enter2.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_enter2.activated.connect(self._enter_contextual)

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
        self.chk_augment = QCheckBox("数据增强")
        self.chk_augment.setChecked(True)
        self.chk_augment.setToolTip("训练时轻微位移/对比度/缩放/噪声，截图少也能更稳")
        self.chk_augment.stateChanged.connect(self._persist_augment)
        form.addWidget(self.chk_augment)
        self.chk_force_export = QCheckBox("强制导出")
        self.chk_force_export.setToolTip("忽略质量门禁（样本不足/准确率偏低）")
        form.addWidget(self.chk_force_export)
        btn_train = QPushButton("开始训练")
        btn_train.setObjectName("primaryBtn")
        btn_train.clicked.connect(self._start_train)
        form.addWidget(btn_train)
        form.addStretch()
        left.addWidget(box)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        left.addWidget(self.train_log, 1)

        self.curve_label = QLabel("训练曲线：训练后显示")
        self.curve_label.setObjectName("hintLabel")
        self.curve_label.setWordWrap(True)
        left.addWidget(self.curve_label)

        exp = QHBoxLayout()
        btn_export = QPushButton("导出 ONNX")
        btn_export.setObjectName("successBtn")
        btn_export.clicked.connect(self._export)
        btn_open_export = QPushButton("打开导出目录")
        btn_open_export.clicked.connect(self._open_export_dir)
        btn_copy_export = QPushButton("复制导出路径")
        btn_copy_export.clicked.connect(self._copy_export_path)
        btn_studio = QPushButton("拷到 Studio models/")
        btn_studio.setToolTip("选择脚本工程目录，自动写入 models/ 并生成 Lua 草稿")
        btn_studio.clicked.connect(self._copy_to_studio)
        self.export_dep_label = QLabel("")
        self.export_dep_label.setObjectName("hintLabel")
        exp.addWidget(btn_export)
        exp.addWidget(btn_open_export)
        exp.addWidget(btn_copy_export)
        exp.addWidget(btn_studio)
        exp.addWidget(self.export_dep_label)
        exp.addStretch()
        left.addLayout(exp)

        bal_row = QHBoxLayout()
        btn_boost = QPushButton("补齐稀缺类")
        btn_boost.setToolTip("对样本偏少的类做增强拷贝")
        btn_boost.clicked.connect(self._boost_scarce)
        btn_scarce = QPushButton("查看缺哪类")
        btn_scarce.clicked.connect(self._show_scarce_classes)
        btn_reg_run = QPushButton("跑回归集")
        btn_reg_run.setToolTip("对 regression/ 金标用例做推理对比")
        btn_reg_run.clicked.connect(self._run_regression)
        bal_row.addWidget(btn_boost)
        bal_row.addWidget(btn_scarce)
        bal_row.addWidget(btn_reg_run)
        btn_conf = QPushButton("混淆矩阵")
        btn_conf.clicked.connect(self._show_confusion)
        btn_backup = QPushButton("备份项目")
        btn_backup.clicked.connect(self._backup_project)
        btn_cmp = QPushButton("多项目对比")
        btn_cmp.clicked.connect(self._compare_projects)
        bal_row.addWidget(btn_conf)
        bal_row.addWidget(btn_backup)
        bal_row.addWidget(btn_cmp)
        bal_row.addStretch()
        left.addLayout(bal_row)
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

    def _goto_review(self) -> None:
        self.tabs.setCurrentIndex(self.TAB_REVIEW)
        if (
            self.project
            and not self.project.pending_files()
            and sum(self.project.class_counts().values()) > 0
        ):
            self._set_review_mode("labeled")
        else:
            self._set_review_mode("pending")

    def _review_only(self, fn) -> None:
        if self.tabs.currentIndex() == self.TAB_REVIEW:
            fn()

    def _update_header(self) -> None:
        if not self.project:
            self.project_title.setText("未打开项目")
            self.badge_pending.setVisible(False)
            return
        n = len(self.project.pending_files())
        labeled_n = sum(self.project.class_counts().values())
        self.project_title.setText(f"项目：{self.project.config.game_id}")
        if n > 0:
            self.badge_pending.setText(f"待审 {n}（点此标注）")
            self.badge_pending.setVisible(True)
        elif labeled_n > 0:
            self.badge_pending.setText(f"已标 {labeled_n}（可点此改标）")
            self.badge_pending.setVisible(True)
        else:
            self.badge_pending.setVisible(False)
        tab = "② 审核标注"
        if n:
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
        lines.append("")
        lines.append(format_balance_text(counts))
        self.project_info.setPlainText("\n".join(lines))
        self._update_header()
        self._reload_roi_preset_combo()
        self._update_export_dep_hint()

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

    def _add_captured_path(self, path: Path, *, apply_last_roi: bool = False) -> None:
        self.import_list.addItem(str(path))
        self.import_list.setCurrentRow(self.import_list.count() - 1)
        self.tabs.setCurrentIndex(self.TAB_WORK)
        # 截完自动进逐字框，并略放大
        QTimer.singleShot(50, lambda: self._after_capture_ready(apply_last_roi=apply_last_roi))
        self._status.showMessage(f"已截取 {path.name} — 可直接拖绿框；Enter 切字")

    def _after_capture_ready(self, *, apply_last_roi: bool = False) -> None:
        self._set_cut_mode("char")
        if apply_last_roi and self._last_roi:
            self.import_canvas.set_roi(self._last_roi, auto_zoom=True)
            self._status.showMessage("已套用上次 ROI，可继续框字或自动预览")
        elif self.import_canvas.roi():
            self.import_canvas.zoom_to_image()
        else:
            self.import_canvas.zoom_to_image()
        if hasattr(self, "guide_banner") and self.guide_banner.isVisible() and self._guide_step == 0:
            self._guide_step = 1
            self._update_guide_banner()

    def _capture_region(self) -> None:
        dest = self._capture_dest_dir()
        if not dest:
            return
        apply_roi = bool(getattr(self, "_pending_apply_roi", False))
        self._pending_apply_roi = False
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
                    self._add_captured_path(path, apply_last_roi=apply_roi)
                except Exception as exc:
                    QMessageBox.critical(self, "截屏失败", str(exc))

            def on_cancelled() -> None:
                self.showNormal()
                self.raise_()
                self.activateWindow()

            overlay.captured.connect(on_captured)
            overlay.cancelled.connect(on_cancelled)
            overlay.show()

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
            self.import_list.setCurrentRow(self.import_list.count() - 1)
            QTimer.singleShot(50, lambda: self._after_capture_ready(apply_last_roi=False))

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
                "拖绿框逐字框选；点选后可拖移/拖角改大小。Enter 切字 · Ctrl+Z 撤销上一框"
            )
            self.btn_mode_char.setObjectName("primaryBtn")
            self.btn_mode_roi.setObjectName("")
            if self.import_canvas.roi():
                self.import_canvas.zoom_to_rect(self.import_canvas.roi(), margin=0.35)
        else:
            self.work_hint.setText(
                "拖蓝框框住数字行（自动放大）→ 可自动预览或改逐字框。ROI 可在「更多」里保存预设"
            )
            self.btn_mode_roi.setObjectName("primaryBtn")
            self.btn_mode_char.setObjectName("")
        self.btn_mode_char.style().unpolish(self.btn_mode_char)
        self.btn_mode_char.style().polish(self.btn_mode_char)
        self.btn_mode_roi.style().unpolish(self.btn_mode_roi)
        self.btn_mode_roi.style().polish(self.btn_mode_roi)
        self._update_box_count()
        update_prefs(cut_mode=mode)

    def _on_view_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"缩放 {zoom:.1f}x")

    def _zoom_to_roi(self) -> None:
        roi = self.import_canvas.roi()
        if not roi:
            QMessageBox.information(self, "提示", "请先拖一个蓝框框住数字区域")
            return
        self.import_canvas.zoom_to_rect(roi, margin=0.35)
        self._status.showMessage("已放大到蓝框 — 可继续滚轮放大，再手动框每个字")

    def _nudge_zoom(self, factor: float) -> None:
        z = self.import_canvas.zoom_factor() * factor
        # reuse zoom_to_rect center of current view via fake: set user zoom through wheel-like API
        c = self.import_canvas
        c._user_zoom = max(1.0, min(z, 16.0))
        c._repaint_canvas()
        c.view_changed.emit(c._user_zoom)

    def _on_roi_changed(self, _roi) -> None:
        if _roi:
            self._last_roi = tuple(int(v) for v in _roi)  # type: ignore[misc]
            update_prefs(last_roi=list(self._last_roi))
        if self.import_canvas.mode() == ImageCanvas.MODE_ROI:
            self._auto_preview_boxes()
            if _roi:
                self._status.showMessage("已自动放大蓝框区域 — 可改「逐字绿框」继续框小字")
        else:
            self._refresh_import_preview()
            if _roi:
                self._status.showMessage("已放大到蓝框 — 滚轮可再放大，然后逐字框选")

    def _on_boxes_changed(self, boxes: list) -> None:
        self.crop_count_label.setText(f"字框 {len(boxes)}")
        self.import_canvas.set_predictions([])
        if hasattr(self, "preview_big"):
            self.preview_big.setText("识别预览：框好后点「预览识别」或稍等自动预览")
        self._status.showMessage(f"已手动框 {len(boxes)} 个字 — Enter 切字 · 可点预览识别")
        self._refresh_selected_box_ui()
        if boxes and self.project and latest_checkpoint(self.project):
            self._preview_timer.start(450)

    def _on_box_selection_changed(self, _index: int = -1) -> None:
        self._refresh_selected_box_ui()

    def _refresh_selected_box_ui(self) -> None:
        if not hasattr(self, "selected_crop_preview"):
            return
        box = self.import_canvas.selected_box()
        idx = self.import_canvas.selected_box_index()
        if box is None or not self._current_import:
            self.selected_crop_preview.clear()
            self.selected_crop_preview.setText("未选")
            self.selected_crop_preview.setStyleSheet(
                "background:#0b1220; color:#94a3b8; border:2px solid #64748b; "
                "border-radius:6px; font-size:11px;"
            )
            self.selected_box_label.setText("点绿框选中后可改宽高")
            self._syncing_box_spins = True
            self.box_w_spin.setEnabled(False)
            self.box_h_spin.setEnabled(False)
            self._syncing_box_spins = False
            return
        x, y, w, h = box
        self._syncing_box_spins = True
        self.box_w_spin.setEnabled(True)
        self.box_h_spin.setEnabled(True)
        self.box_w_spin.setValue(int(w))
        self.box_h_spin.setValue(int(h))
        self._syncing_box_spins = False
        self.selected_box_label.setText(f"第{idx + 1}框 {w}×{h}")
        self.selected_crop_preview.setStyleSheet(
            "background:#0b1220; color:#fff; border:2px solid #ef4444; "
            "border-radius:6px; font-size:11px;"
        )
        try:
            bgr = load_bgr(self._current_import)
            crop = crop_bgr(bgr, (x, y, w, h))
            if crop is None or crop.size == 0:
                self.selected_crop_preview.setText("空")
                return
            pix = numpy_to_pixmap(crop).scaled(
                52,
                52,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.selected_crop_preview.setPixmap(pix)
            self.selected_crop_preview.setText("")
        except Exception:
            self.selected_crop_preview.setText("失败")

    def _on_box_size_spin(self, _value: int = 0) -> None:
        if getattr(self, "_syncing_box_spins", False):
            return
        if self.import_canvas.selected_box() is None:
            return
        ok = self.import_canvas.set_selected_box_size(
            width=self.box_w_spin.value(),
            height=self.box_h_spin.value(),
        )
        if ok:
            self._status.showMessage(
                f"已改选中框为 {self.box_w_spin.value()}×{self.box_h_spin.value()}px"
            )

    def _nudge_selected_size(self, dw: int = 0, dh: int = 0) -> None:
        if self.import_canvas.selected_box() is None:
            QMessageBox.information(self, "提示", "请先点选一个绿框（选中后变红框）")
            return
        if self.import_canvas.nudge_selected_box(dw=dw, dh=dh):
            self._refresh_selected_box_ui()
            box = self.import_canvas.selected_box()
            if box:
                self._status.showMessage(f"选中框现为 {box[2]}×{box[3]}px")

    def _update_box_count(self) -> None:
        n = len(self.import_canvas.boxes())
        self.crop_count_label.setText(f"字框 {n}")

    def _undo_char_box(self) -> None:
        self.import_canvas.undo_box_edit()
        self._update_box_count()
        self._refresh_selected_box_ui()

    def _clear_char_boxes(self) -> None:
        self.import_canvas.clear_boxes()
        self._update_box_count()
        self._refresh_import_preview()

    def _clear_roi(self) -> None:
        self.import_canvas.clear_roi()
        if self.import_canvas.mode() == ImageCanvas.MODE_ROI:
            self.import_canvas.clear_boxes()
        self._refresh_import_preview()
        self.zoom_label.setText("缩放 1.0x")

    def _on_gap_changed(self, _value: int = 0) -> None:
        """间距一变就重跑自动切字，并刷新二值预览。"""
        self._refresh_import_preview()
        if self.project:
            self.project.config.last_segment_gap = int(self.gap_spin.value())
            try:
                self.project.save_config()
            except Exception:
                pass
        if self.import_canvas.roi() or self.import_canvas.boxes():
            self._auto_preview_boxes()

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
            fixed, tips = auto_fix_boxes(boxes)
            self.import_canvas.set_boxes(fixed)
            self._refresh_import_preview(keep_manual_boxes=True)
            gap = self.gap_spin.value()
            tip = ""
            if tips:
                tip = f"；建议拆粘连：第 {', '.join(str(i + 1) for i in tips[:5])} 框偏宽"
            self._status.showMessage(
                f"自动预览 {len(fixed)} 个字框（间距={gap}，已修碎框）{tip}"
            )
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
            self.import_canvas.set_image_bgr_or_gray(
                show, boxes=None, draw_stored_roi=True, keep_boxes=True
            )
            self._update_box_count()
            self._update_preprocess_mini_preview(bgr, roi)
        except Exception as exc:
            self.import_canvas.setText(str(exc))

    def _update_preprocess_mini_preview(self, bgr, roi) -> None:
        if not hasattr(self, "preprocess_preview") or not self.project:
            return
        try:
            sliced = crop_bgr(bgr, roi)
            binary = apply_preprocess(sliced, self.project.config.preprocess)
            pix = numpy_to_pixmap(binary).scaled(
                116,
                44,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.preprocess_preview.setPixmap(pix)
            self.preprocess_preview.setText("")
        except Exception:
            self.preprocess_preview.setText("预览失败")

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
    def _set_review_mode(self, mode: str) -> None:
        self._review_mode = mode if mode in ("pending", "labeled", "hard") else "pending"
        self.btn_rev_pending.setObjectName("primaryBtn" if self._review_mode == "pending" else "")
        self.btn_rev_labeled.setObjectName("primaryBtn" if self._review_mode == "labeled" else "")
        if hasattr(self, "btn_rev_hard"):
            self.btn_rev_hard.setObjectName("primaryBtn" if self._review_mode == "hard" else "")
        if self._review_mode == "pending":
            self.gallery_title.setText("待审预览（点击选择）")
        elif self._review_mode == "labeled":
            self.gallery_title.setText("已标注（点选后可改标/退回）")
        else:
            self.gallery_title.setText("难例（低置信/改标/金标失败）")
        for b in (self.btn_rev_pending, self.btn_rev_labeled, getattr(self, "btn_rev_hard", None)):
            if b is None:
                continue
            b.style().unpolish(b)
            b.style().polish(b)
        self._reload_review_lists()

    def _reload_review_lists(self) -> None:
        if not self.project:
            return
        self._pending = self.project.pending_files()
        if self._sort_pending_by_conf and self._pending_scores:
            self._pending.sort(
                key=lambda p: self._pending_scores.get(p.name, ("", 1.0))[1]
            )
        self._labeled = list_all_labeled(self.project)
        self._hard = list_hard_files(self.project)
        self._rebuild_gallery()
        self._show_current()
        self._update_header()
        self._update_review_progress()

    def _reload_pending(self) -> None:
        self._reload_review_lists()

    def _rebuild_gallery(self) -> None:
        self.pending_list.blockSignals(True)
        self.pending_list.clear()
        if self._review_mode == "pending":
            items = [(p, None) for p in self._pending]
        elif self._review_mode == "hard":
            meta = {x.get("file"): x for x in load_hard_index(self.project)} if self.project else {}
            items = []
            for p in self._hard:
                reason = (meta.get(p.name) or {}).get("reason") or "难例"
                items.append((p, f"难:{reason}"))
        else:
            items = [(p, lab) for p, lab in self._labeled]
        for i, (path, lab) in enumerate(items):
            if lab is None:
                score = self._pending_scores.get(path.name)
                if score:
                    title = f"{display_label(score[0])} {score[1]:.0%}"
                else:
                    title = f"{i + 1}"
            elif str(lab).startswith("难:"):
                title = str(lab)[2:6]
            else:
                title = f"{display_label(lab)}"
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            tip = path.name if lab is None else f"{lab} · {path.name}"
            if lab is None and path.name in self._pending_scores:
                tip += f" · pred {self._pending_scores[path.name]}"
            item.setToolTip(tip)
            raw = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pix = numpy_to_pixmap(img).scaled(
                    72, 72, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                item.setIcon(QIcon(pix))
            self.pending_list.addItem(item)
        self.pending_list.blockSignals(False)
        if self._review_mode == "pending" and self._pending:
            self._idx = min(self._idx, len(self._pending) - 1)
            self.pending_list.blockSignals(True)
            self.pending_list.setCurrentRow(self._idx)
            self.pending_list.blockSignals(False)
        elif self._review_mode == "labeled" and self._labeled:
            self._labeled_idx = min(self._labeled_idx, len(self._labeled) - 1)
            self.pending_list.blockSignals(True)
            self.pending_list.setCurrentRow(self._labeled_idx)
            self.pending_list.blockSignals(False)
        elif self._review_mode == "hard" and self._hard:
            self._hard_idx = min(self._hard_idx, len(self._hard) - 1)
            self.pending_list.blockSignals(True)
            self.pending_list.setCurrentRow(self._hard_idx)
            self.pending_list.blockSignals(False)

    def _on_gallery_selected(self, row: int) -> None:
        if row < 0:
            return
        if self._review_mode == "pending":
            if row >= len(self._pending):
                return
            self._idx = row
        elif self._review_mode == "hard":
            if row >= len(self._hard):
                return
            self._hard_idx = row
        else:
            if row >= len(self._labeled):
                return
            self._labeled_idx = row
        self._show_current(sync_list=False)

    def _on_pending_selected(self, row: int) -> None:
        self._on_gallery_selected(row)

    def _rebuild_pending_gallery(self) -> None:
        self._rebuild_gallery()

    def _current_review_path(self) -> Path | None:
        if self._review_mode == "pending":
            if not self._pending:
                return None
            self._idx = max(0, min(self._idx, len(self._pending) - 1))
            return self._pending[self._idx]
        if self._review_mode == "hard":
            if not self._hard:
                return None
            self._hard_idx = max(0, min(self._hard_idx, len(self._hard) - 1))
            return self._hard[self._hard_idx]
        if not self._labeled:
            return None
        self._labeled_idx = max(0, min(self._labeled_idx, len(self._labeled) - 1))
        return self._labeled[self._labeled_idx][0]

    def _current_labeled_class(self) -> str | None:
        if self._review_mode != "labeled" or not self._labeled:
            return None
        self._labeled_idx = max(0, min(self._labeled_idx, len(self._labeled) - 1))
        return self._labeled[self._labeled_idx][1]

    def _show_current(self, sync_list: bool = True) -> None:
        self._pred_label = None
        self._pred_conf = 0.0
        path = self._current_review_path()
        if path is None:
            if self._review_mode == "pending":
                self.char_view.setText("没有待审核\n去「截图切字」继续")
                self.review_meta.setText("无待审核")
            elif self._review_mode == "hard":
                self.char_view.setText("难例队列为空")
                self.review_meta.setText("无难例")
            else:
                self.char_view.setText("还没有已标注样本")
                self.review_meta.setText("无已标注")
            self.pred_label_ui.setText("预测：—")
            self.btn_confirm.setEnabled(False)
            self.context_view.setText("原图对照：无样本")
            self._highlight_suggested_label(None)
            self._update_review_progress()
            return

        if sync_list:
            if self._review_mode == "pending":
                row = self._idx
            elif self._review_mode == "hard":
                row = self._hard_idx
            else:
                row = self._labeled_idx
            if self.pending_list.currentRow() != row:
                self.pending_list.blockSignals(True)
                self.pending_list.setCurrentRow(row)
                self.pending_list.blockSignals(False)

        raw = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.char_view.setText("无法读取")
            return
        pix = numpy_to_pixmap(img)
        self.char_view.setPixmap(
            pix.scaled(340, 340, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        )

        if self._review_mode == "pending":
            self.review_meta.setText(f"待审 {self._idx + 1} / {len(self._pending)}  ·  {path.name}")
            self._update_prediction(path)
        elif self._review_mode == "hard":
            self.review_meta.setText(f"难例 {self._hard_idx + 1} / {len(self._hard)}  ·  {path.name}")
            self._update_prediction(path)
            self.btn_confirm.setText("难例：点数字标注后移入数据集")
        else:
            lab = self._current_labeled_class() or "?"
            self.review_meta.setText(
                f"已标注 {self._labeled_idx + 1} / {len(self._labeled)}  ·  当前类「{display_label(lab)}」  ·  {path.name}"
            )
            self.pred_label_ui.setText(f"当前标签：{display_label(lab)}（可直接改标）")
            self.pred_label_ui.setStyleSheet("font-size:20px; font-weight:700; color:#2563eb;")
            self.btn_confirm.setEnabled(False)
            self.btn_confirm.setText("已标注模式：点数字即可改标")
            self._highlight_suggested_label(lab)
        self._update_context_view(path)
        self._update_review_progress()

    def _update_context_view(self, path: Path) -> None:
        if not self.project:
            self.context_view.setText("原图对照：无项目")
            return
        meta = get_meta(self.project, path.name)
        if not meta:
            self.context_view.setText("原图对照：无来源信息（旧样本可忽略）")
            return
        src = resolve_source(self.project, meta)
        box = meta.get("box")
        line_box = meta.get("line_box") or box
        if not src or not box or not line_box:
            self.context_view.setText("原图对照：找不到源截图")
            return
        try:
            bgr = load_bgr(src)
        except Exception:
            self.context_view.setText("原图对照：源图读取失败")
            return
        lx, ly, lw, lh = [int(v) for v in line_box]
        pad = 12
        H, W = bgr.shape[:2]
        x0 = max(0, lx - pad)
        y0 = max(0, ly - pad)
        x1 = min(W, lx + lw + pad)
        y1 = min(H, ly + lh + pad)
        strip = bgr[y0:y1, x0:x1].copy()
        bx, by, bw, bh = [int(v) for v in box]
        # highlight relative to strip
        rx, ry = bx - x0, by - y0
        cv2.rectangle(strip, (rx, ry), (rx + bw, ry + bh), (0, 220, 80), 2)
        pix = numpy_to_pixmap(strip)
        self.context_view.setPixmap(
            pix.scaled(
                max(self.context_view.width(), 480),
                150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_prediction(self, path: Path) -> None:
        if not self.project:
            return
        ckpt = latest_checkpoint(self.project)
        if not ckpt:
            self.pred_label_ui.setText("尚无模型 · 先手动标一些再训练")
            self.btn_confirm.setEnabled(False)
            self.btn_confirm.setText("空格确认（需先训练）")
            self._highlight_suggested_label(None)
            return
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
            self.pred_label_ui.setStyleSheet(f"font-size:22px; font-weight:700; color:{color};")
            self.btn_confirm.setEnabled(True)
            self.btn_confirm.setText(f"空格：确认「{shown}」")
            self._highlight_suggested_label(lab)
        except Exception:
            self.pred_label_ui.setText("预测失败")
            self.btn_confirm.setEnabled(False)
            self._highlight_suggested_label(None)

    def _confirm_prediction(self) -> None:
        if self._review_mode != "pending":
            return
        if not self._pred_label:
            return
        if hasattr(self, "batch_stop_label"):
            self.batch_stop_label.setVisible(False)
        seed_label = normalize_label(self._pred_label)
        self._assign(self._pred_label)
        if not self.chk_batch_confirm.isChecked():
            return
        same_only = bool(getattr(self, "chk_batch_same", None) and self.chk_batch_same.isChecked())
        self._batch_confirming = True
        stopped_reason = ""
        try:
            while self._review_mode == "pending" and self._pending:
                if not self._pred_label:
                    stopped_reason = "当前无预测，已停下"
                    break
                if self._pred_conf < float(self.conf_spin.value()):
                    stopped_reason = (
                        f"批量确认已停下：预测「{display_label(self._pred_label)}」"
                        f"置信度 {self._pred_conf:.0%} < 阈值 {float(self.conf_spin.value()):.0%}，请手标"
                    )
                    break
                if same_only and normalize_label(self._pred_label) != seed_label:
                    stopped_reason = (
                        f"同类批量已停下：下一张预测是「{display_label(self._pred_label)}」"
                        f"（当前批为「{display_label(seed_label)}」）"
                    )
                    break
                self._assign(self._pred_label)
        finally:
            self._batch_confirming = False
        if stopped_reason and hasattr(self, "batch_stop_label"):
            self.batch_stop_label.setText(stopped_reason)
            self.batch_stop_label.setVisible(True)
            self._status.showMessage(stopped_reason)

    def _assign(self, label: str) -> None:
        proj = self._require_project()
        if not proj:
            return
        if self._review_mode == "labeled":
            self._relabel_current(label)
            return
        if self._review_mode == "hard":
            if not self._hard:
                return
            path = self._hard[self._hard_idx]
            try:
                dest = move_to_label(proj, path, label)
            except Exception as exc:
                QMessageBox.warning(self, "标注失败", str(exc))
                return
            remove_hard_file(proj, path)
            self._last_labeled_path = dest
            self._reload_review_lists()
            self._refresh_project_info()
            self._status.showMessage(f"难例已标为 {display_label(normalize_label(label))}")
            return
        if not self._pending:
            return
        path = self._pending[self._idx]
        try:
            dest = move_to_label(proj, path, label)
        except Exception as exc:
            QMessageBox.warning(self, "标注失败", str(exc))
            return
        # 低置信或与预测不一致 → 难例备份
        try:
            nl = normalize_label(label)
            if self._pred_label and (
                normalize_label(self._pred_label) != nl or self._pred_conf < 0.75
            ):
                add_hard_example(
                    proj,
                    dest,
                    reason="低置信或改预测",
                    pred=self._pred_label,
                    conf=self._pred_conf,
                    expected=nl,
                )
        except Exception:
            pass
        self._undo_stack.append(
            {
                "action": "label",
                "dest": dest,
                "label": normalize_label(label),
                "pending_name": path.name,
            }
        )
        self._last_labeled_path = dest
        self._pending.pop(self._idx)
        if self._idx >= len(self._pending) and self._pending:
            self._idx = len(self._pending) - 1
        self._labeled = list_all_labeled(proj)
        self._rebuild_gallery()
        if self._pending:
            self.pending_list.blockSignals(True)
            self.pending_list.setCurrentRow(self._idx)
            self.pending_list.blockSignals(False)
        self._show_current(sync_list=False)
        self._refresh_project_info()
        msg = f"已标为 {display_label(normalize_label(label))} — 标错可点「已标注」改，或 Ctrl+Z /「查看刚标的那张」"
        if not self._batch_confirming:
            self._status.showMessage(msg)
        self._update_review_progress()

    def _relabel_current(self, label: str) -> None:
        proj = self._require_project()
        path = self._current_review_path()
        if not proj or not path:
            return
        old_lab = self._current_labeled_class()
        try:
            dest = relabel_dataset_file(proj, path, label)
        except Exception as exc:
            QMessageBox.warning(self, "改标失败", str(exc))
            return
        self._undo_stack.append(
            {
                "action": "relabel",
                "dest": dest,
                "label": normalize_label(label),
                "old_label": old_lab,
            }
        )
        self._labeled = list_all_labeled(proj)
        # stay on same logical item by finding dest
        for i, (p, _) in enumerate(self._labeled):
            if p == dest:
                self._labeled_idx = i
                break
        self._rebuild_gallery()
        self.pending_list.blockSignals(True)
        self.pending_list.setCurrentRow(self._labeled_idx)
        self.pending_list.blockSignals(False)
        self._show_current(sync_list=False)
        self._refresh_project_info()
        self._last_labeled_path = dest
        try:
            add_hard_example(proj, dest, reason="人工改标", expected=normalize_label(label), pred=old_lab)
        except Exception:
            pass
        self._status.showMessage(f"已改标为 {display_label(normalize_label(label))}")

    def _return_current_to_pending(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if self._review_mode != "labeled":
            QMessageBox.information(self, "提示", "请先切换到「已标注（可改）」再退回待审")
            self._set_review_mode("labeled")
            return
        path = self._current_review_path()
        if not path:
            return
        old_lab = self._current_labeled_class()
        try:
            dest = move_to_pending(proj, path)
        except Exception as exc:
            QMessageBox.warning(self, "退回失败", str(exc))
            return
        self._undo_stack.append(
            {"action": "return", "dest": dest, "old_label": old_lab}
        )
        self._review_mode = "pending"
        self._reload_review_lists()
        self.btn_rev_pending.setObjectName("primaryBtn")
        self.btn_rev_labeled.setObjectName("")
        for b in (self.btn_rev_pending, self.btn_rev_labeled):
            b.style().unpolish(b)
            b.style().polish(b)
        self.gallery_title.setText("待审预览（点击选择）")
        self._status.showMessage("已退回待审，可重新标注")

    def _undo_last_label(self) -> None:
        proj = self._require_project()
        if not proj or not self._undo_stack:
            self._status.showMessage("没有可撤销的操作")
            return
        item = self._undo_stack.pop()
        action = item.get("action")
        next_mode = self._review_mode
        try:
            if action == "label":
                dest: Path = item["dest"]
                if dest.exists():
                    move_to_pending(proj, dest)
                next_mode = "pending"
                self._status.showMessage("已撤销上一标注，样本回到待审")
            elif action == "relabel":
                dest = item["dest"]
                old = item.get("old_label")
                if dest.exists() and old:
                    relabel_dataset_file(proj, dest, old)
                next_mode = "labeled"
                self._status.showMessage("已撤销改标")
            elif action == "return":
                dest = item["dest"]
                old = item.get("old_label")
                if dest.exists() and old:
                    move_to_label(proj, dest, old)
                next_mode = "labeled"
                self._status.showMessage("已撤销退回待审")
            else:
                self._status.showMessage("无法撤销该操作")
                return
        except Exception as exc:
            QMessageBox.warning(self, "撤销失败", str(exc))
            return
        # 直接设模式再刷新，避免 _set_review_mode 重复 reload
        self._review_mode = next_mode
        if next_mode == "pending":
            self.gallery_title.setText("待审预览（点击选择）")
            self.btn_rev_pending.setObjectName("primaryBtn")
            self.btn_rev_labeled.setObjectName("")
        else:
            self.gallery_title.setText("已标注（点选后可改标/退回）")
            self.btn_rev_labeled.setObjectName("primaryBtn")
            self.btn_rev_pending.setObjectName("")
        for b in (self.btn_rev_pending, self.btn_rev_labeled):
            b.style().unpolish(b)
            b.style().polish(b)
        self._reload_review_lists()
        self._refresh_project_info()

    def _delete_current(self) -> None:
        path = self._current_review_path()
        if not path:
            return
        if self._review_mode in ("labeled", "hard"):
            reply = QMessageBox.question(
                self,
                "确认删除",
                f"删除样本？\n{path.name}\n（不可撤销）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if self._review_mode == "hard" and self.project:
            remove_hard_file(self.project, path)
            self._hard = list_hard_files(self.project)
            if self._hard_idx >= len(self._hard) and self._hard:
                self._hard_idx = len(self._hard) - 1
        else:
            path.unlink(missing_ok=True)
            if self._review_mode == "pending":
                self._pending.pop(self._idx)
                if self._idx >= len(self._pending) and self._pending:
                    self._idx = len(self._pending) - 1
            else:
                self._labeled.pop(self._labeled_idx)
                if self._labeled_idx >= len(self._labeled) and self._labeled:
                    self._labeled_idx = len(self._labeled) - 1
        self._rebuild_gallery()
        self._show_current(sync_list=True)
        self._update_header()
        self._refresh_project_info()

    def _next_pending(self) -> None:
        if self._review_mode == "pending":
            if self._pending:
                self._idx = min(self._idx + 1, len(self._pending) - 1)
                self._show_current()
        elif self._review_mode == "hard":
            if self._hard:
                self._hard_idx = min(self._hard_idx + 1, len(self._hard) - 1)
                self._show_current()
        elif self._labeled:
            self._labeled_idx = min(self._labeled_idx + 1, len(self._labeled) - 1)
            self._show_current()

    def _prev_pending(self) -> None:
        if self._review_mode == "pending":
            if self._pending:
                self._idx = max(self._idx - 1, 0)
                self._show_current()
        elif self._review_mode == "hard":
            if self._hard:
                self._hard_idx = max(self._hard_idx - 1, 0)
                self._show_current()
        elif self._labeled:
            self._labeled_idx = max(self._labeled_idx - 1, 0)
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
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"删除样本 {path.name}？（不可撤销）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
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
        warns = balance_warnings(counts)
        if warns:
            detail = "\n".join(f"· {w}" for w in warns)
            reply = QMessageBox.question(
                self,
                "类别均衡提示",
                f"检测到可能影响效果的问题：\n{detail}\n\n仍要开始训练吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        augment = self.chk_augment.isChecked()
        proj.config.augment = augment
        proj.save_config()
        self.train_log.clear()
        self._worker = TrainWorker(proj, self.epochs_spin.value(), augment=augment)
        self._worker.log.connect(lambda m: self.train_log.append(m))
        self._worker.done.connect(self._train_done)
        self._worker.failed.connect(lambda e: QMessageBox.critical(self, "训练失败", e))
        self._worker.start()

    def _train_done(self, path: str) -> None:
        self.train_log.append(f"完成: {path}")
        self._refresh_project_info()
        self._refresh_train_curve()
        QMessageBox.information(
            self,
            "完成",
            "训练完成。回切字页点「预览识别」可看整行读数；审核页可空格批量确认。",
        )
        self.tabs.setCurrentIndex(self.TAB_WORK)

    def _refresh_train_curve(self) -> None:
        if not hasattr(self, "curve_label") or not self.project:
            return
        runs = sorted(self.project.runs_dir.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            self.curve_label.setText("训练曲线：尚无")
            return
        import json

        try:
            data = json.loads(runs[0].read_text(encoding="utf-8"))
            hist = data.get("history") or []
        except Exception:
            self.curve_label.setText("训练曲线：读取失败")
            return
        if not hist:
            return
        lines = ["训练曲线（最近一次）:"]
        for h in hist[-8:]:
            bar = "█" * max(1, int(float(h.get("val_acc", 0)) * 20))
            lines.append(
                f"  ep{h.get('epoch')}: loss={float(h.get('loss', 0)):.3f}  "
                f"val={float(h.get('val_acc', 0)):.0%} {bar}"
            )
        best = data.get("best_val_acc")
        if best is not None:
            lines.append(f"最佳 val_acc={float(best):.1%}")
        self.curve_label.setText("\n".join(lines))

    def _export(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        ok, msg = check_onnx_dependency()
        self._update_export_dep_hint()
        if not ok:
            QMessageBox.warning(self, "缺少依赖", msg)
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            QMessageBox.warning(self, "提示", "请先训练")
            return
        errors, warnings = export_quality_report(proj, ckpt)
        force = hasattr(self, "chk_force_export") and self.chk_force_export.isChecked()
        if errors and not force:
            QMessageBox.warning(
                self,
                "质量门禁未通过",
                "\n".join(f"· {e}" for e in errors)
                + "\n\n可勾选「强制导出」跳过，或先补样本/再训练。",
            )
            return
        if warnings or (errors and force):
            detail = "\n".join(f"· {w}" for w in (errors + warnings))
            reply = QMessageBox.question(
                self,
                "导出确认",
                f"{detail}\n\n仍要导出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            out = export_onnx(proj, ckpt)
            verify_msg = verify_onnx_runtime(
                out,
                width=proj.config.input_width,
                height=proj.config.input_height,
            )
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"{exc}\n{traceback.format_exc()}")
            return
        QMessageBox.information(
            self,
            "已导出",
            f"{out.parent}\n\n{verify_msg}\n\n拷到 Studio models/ 见 docs/studio-recognize-digits.md",
        )
        self._reload_verify_models(prefer_key=f"onnx:{out.resolve()}")

    def _open_export_dir(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        proj.exports_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(proj.exports_dir)  # noqa: S606

    def _copy_export_path(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        proj.exports_dir.mkdir(parents=True, exist_ok=True)
        QApplication.clipboard().setText(str(proj.exports_dir.resolve()))
        self._status.showMessage(f"已复制导出路径: {proj.exports_dir}")

    def _update_export_dep_hint(self) -> None:
        if not hasattr(self, "export_dep_label"):
            return
        ok, msg = check_onnx_dependency()
        self.export_dep_label.setText("导出依赖 OK" if ok else msg)
        self.export_dep_label.setStyleSheet(
            "color:#059669;" if ok else "color:#dc2626; font-weight:600;"
        )

    def _persist_confirm_threshold(self, value: float) -> None:
        if not self.project:
            return
        self.project.config.confirm_threshold = float(value)
        self.project.save_config()
        update_prefs(confirm_threshold=float(value))

    def _persist_augment(self, _state: int = 0) -> None:
        if not self.project:
            return
        self.project.config.augment = self.chk_augment.isChecked()
        self.project.save_config()

    def _reload_roi_preset_combo(self) -> None:
        if not hasattr(self, "roi_preset_combo"):
            return
        self.roi_preset_combo.blockSignals(True)
        self.roi_preset_combo.clear()
        if self.project:
            for p in self.project.config.roi_presets:
                self.roi_preset_combo.addItem(f"{p.name} ({p.w}x{p.h})", p.name)
        self.roi_preset_combo.blockSignals(False)
        name = self._ui_prefs.get("last_roi_preset")
        if name:
            idx = self.roi_preset_combo.findData(name)
            if idx >= 0:
                self.roi_preset_combo.setCurrentIndex(idx)

    def _save_roi_preset(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        roi = self.import_canvas.roi()
        if not roi:
            QMessageBox.information(self, "提示", "请先拖一个蓝框（框整行模式）")
            return
        name, ok = QInputDialog.getText(self, "保存 ROI 预设", "名称（如：金币行）")
        if not ok or not name.strip():
            return
        name = name.strip()
        x, y, w, h = roi
        presets = [p for p in proj.config.roi_presets if p.name != name]
        presets.append(RoiPreset(name=name, x=x, y=y, w=w, h=h))
        proj.config.roi_presets = presets
        proj.save_config()
        self._reload_roi_preset_combo()
        idx = self.roi_preset_combo.findData(name)
        if idx >= 0:
            self.roi_preset_combo.setCurrentIndex(idx)
        update_prefs(last_roi_preset=name)
        self._status.showMessage(f"已保存 ROI 预设「{name}」")

    def _apply_roi_preset(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        name = self.roi_preset_combo.currentData()
        preset = next((p for p in proj.config.roi_presets if p.name == name), None)
        if not preset:
            QMessageBox.information(self, "提示", "没有可选预设")
            return
        self._set_cut_mode("roi")
        self.import_canvas.set_roi((preset.x, preset.y, preset.w, preset.h))
        self._last_roi = (preset.x, preset.y, preset.w, preset.h)
        update_prefs(last_roi_preset=preset.name, last_roi=list(self._last_roi))
        self._status.showMessage(f"已套用 ROI「{preset.name}」— 可再自动预览切字或改手动框")

    def _delete_roi_preset(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        name = self.roi_preset_combo.currentData()
        if not name:
            return
        proj.config.roi_presets = [p for p in proj.config.roi_presets if p.name != name]
        proj.save_config()
        self._reload_roi_preset_combo()
        self._status.showMessage(f"已删除预设「{name}」")

    def _trial_infer_boxes(self) -> None:
        self._preview_recognize()

    def _preview_recognize_silent(self) -> None:
        try:
            self._preview_recognize(silent=True)
        except Exception:
            pass

    def _preview_recognize(self, silent: bool = False) -> None:
        proj = self._require_project()
        if not proj:
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            if not silent:
                QMessageBox.information(self, "提示", "请先训练模型再预览识别")
            if hasattr(self, "preview_big"):
                self.preview_big.setText("识别预览：需先训练")
            return
        if not self._current_import:
            if not silent:
                QMessageBox.information(self, "提示", "请先打开一张截图")
            return
        boxes = self.import_canvas.boxes()
        if not boxes:
            if not silent:
                self._auto_preview_boxes()
                boxes = self.import_canvas.boxes()
        if not boxes:
            if not silent:
                QMessageBox.warning(self, "提示", "请先框出字框")
            return
        try:
            bgr = load_bgr(self._current_import)
            text, parts = predict_boxes_string(
                proj,
                bgr,
                boxes,
                ckpt,
                conf_threshold=float(self.conf_spin.value()) if hasattr(self, "conf_spin") else 0.5,
            )
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "预览失败", str(exc))
            return
        self.import_canvas.set_predictions(parts)
        if hasattr(self, "preview_big"):
            self.preview_big.setText(text or "（空）")
        detail = " ".join(f"{display_label(l)}({c:.0%})" for l, c in parts)
        self.trial_result.setText(f"预览明细：{detail}")
        if not silent:
            self._status.showMessage(f"识别预览：{text}")

    def _set_canvas_interaction(self, mode: str) -> None:
        self.import_canvas.set_interaction_mode(mode)
        if hasattr(self, "btn_mode_draw"):
            self.btn_mode_draw.setChecked(mode == "draw")
            self.btn_mode_pan.setChecked(mode == "pan")
        self._status.showMessage("拖图模式：左键拖画面" if mode == "pan" else "画框模式：左键拖出字框")

    def _merge_selected_box(self) -> None:
        if not self.import_canvas.boxes():
            QMessageBox.information(self, "提示", "还没有绿框")
            return
        if self.import_canvas.selected_box_index() < 0:
            QMessageBox.information(self, "提示", "请先点选一个绿框再合并")
            return
        if not self.import_canvas.merge_selected_with_neighbor():
            QMessageBox.information(self, "提示", "无法合并（至少需要两个框）")
            return
        self._status.showMessage("已合并相邻字框")
        self._preview_timer.start(300)

    def _autofix_boxes(self) -> None:
        boxes = self.import_canvas.boxes()
        if not boxes:
            QMessageBox.information(self, "提示", "还没有绿框")
            return
        fixed, tips = auto_fix_boxes(boxes)
        self.import_canvas.set_boxes(fixed)
        msg = f"已修碎框 → {len(fixed)} 个"
        if tips:
            msg += f"；建议拆：第 {', '.join(str(i + 1) for i in tips[:6])} 框"
            # 自动选中最宽建议框
            self.import_canvas.select_box(tips[0])
        self._status.showMessage(msg)

    def _reload_segment_presets(self) -> None:
        if not hasattr(self, "seg_preset_combo") or not self.project:
            return
        self.seg_preset_combo.clear()
        for p in self.project.config.segment_presets:
            self.seg_preset_combo.addItem(f"{p.name} (gap={p.gap})", p.name)

    def _apply_segment_preset(self) -> None:
        if not self.project or not hasattr(self, "seg_preset_combo"):
            return
        name = self.seg_preset_combo.currentData()
        preset = next((p for p in self.project.config.segment_presets if p.name == name), None)
        if not preset:
            return
        self.gap_spin.setValue(int(preset.gap))
        self.chk_invert.setChecked(bool(preset.invert))
        idx = self.binarize_combo.findText(str(preset.binarize))
        if idx >= 0:
            self.binarize_combo.setCurrentIndex(idx)
        self._auto_preview_boxes()
        self._status.showMessage(f"已套用切字预设「{preset.name}」")

    def _save_segment_preset(self) -> None:
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "切字预设", "名称：", text="默认切字")
        if not ok or not name.strip():
            return
        name = name.strip()
        preset = SegmentPreset(
            name=name,
            gap=int(self.gap_spin.value()),
            invert=bool(self.chk_invert.isChecked()),
            binarize=str(self.binarize_combo.currentText()),
        )
        cfg = self.project.config
        cfg.segment_presets = [p for p in cfg.segment_presets if p.name != name] + [preset]
        self.project.save_config()
        self._reload_segment_presets()
        self._status.showMessage(f"已保存切字预设「{name}」")

    def _delete_segment_preset(self) -> None:
        if not self.project or not hasattr(self, "seg_preset_combo"):
            return
        name = self.seg_preset_combo.currentData()
        if not name:
            return
        cfg = self.project.config
        cfg.segment_presets = [p for p in cfg.segment_presets if p.name != name]
        self.project.save_config()
        self._reload_segment_presets()

    def _on_sort_conf_toggled(self, checked: bool) -> None:
        self._sort_pending_by_conf = bool(checked)
        self._reload_review_lists()

    def _prelabel_all_pending(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            QMessageBox.information(self, "提示", "请先训练出模型再预标")
            return
        paths = proj.pending_files()
        if not paths:
            QMessageBox.information(self, "提示", "没有待审样本")
            return
        scored = score_pending_files(
            ckpt,
            paths,
            proj.config.classes,
            proj.config.input_width,
            proj.config.input_height,
        )
        self._pending_scores = {p.name: (lab, conf) for p, lab, conf in scored}
        self._sort_pending_by_conf = True
        if hasattr(self, "chk_sort_conf"):
            self.chk_sort_conf.setChecked(True)
        self._set_review_mode("pending")
        self._reload_review_lists()
        self._status.showMessage(f"已预标 {len(scored)} 张，按置信度升序（难的在前）")

    def _copy_to_studio(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        path = QFileDialog.getExistingDirectory(self, "选择脚本工程目录（将写入 models/）")
        if not path:
            return
        models = Path(path) / "models"
        try:
            copied = copy_exports_to_studio(proj.exports_dir, models)
        except Exception as exc:
            QMessageBox.warning(self, "拷贝失败", str(exc))
            return
        QMessageBox.information(
            self,
            "已拷到 Studio",
            f"目标：{models}\n文件：{', '.join(copied)}\n详见 docs/studio-recognize-digits.md",
        )

    def _boost_scarce(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        counts = proj.class_counts()
        added = boost_scarce_classes(proj.dataset_dir, proj.config.classes, counts, target=12)
        if not added:
            QMessageBox.information(self, "均衡", "没有需要补齐的稀缺类（或全是 0 样本）")
            return
        detail = "、".join(f"{k}+{v}" for k, v in added.items())
        self._refresh_project_info()
        self._reload_dataset_browser()
        QMessageBox.information(self, "已补齐", f"增强拷贝：{detail}")

    def _show_scarce_classes(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        counts = proj.class_counts()
        scarce = scarce_classes(counts)
        text = format_balance_text(counts)
        if scarce:
            text += "\n\n优先刷这些类：\n" + "\n".join(f"· {k}: {v}" for k, v in scarce)
        QMessageBox.information(self, "类别均衡", text)

    def _add_regression_case(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if not self._current_import:
            QMessageBox.information(self, "提示", "请先打开一张截图")
            return
        expected = self.gold_edit.text().strip()
        if not expected:
            QMessageBox.information(self, "提示", "请先填写金标")
            return
        try:
            item = add_regression_case(
                proj,
                image_path=self._current_import,
                expected=expected,
                boxes=self.import_canvas.boxes() or None,
            )
        except Exception as exc:
            QMessageBox.warning(self, "失败", str(exc))
            return
        self._status.showMessage(f"已加入回归集：{item.get('name')}（共 {len(load_cases(proj))} 条）")

    def _run_regression(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            QMessageBox.information(self, "提示", "请先训练")
            return
        cases = load_cases(proj)
        if not cases:
            QMessageBox.information(self, "提示", "回归集为空：在切字页填金标后点「加入回归集」")
            return
        report = run_regression(proj, ckpt)
        lines = [f"通过 {report['passed']}/{report['total']}"]
        for r in report.get("results") or []:
            if r.get("ok"):
                lines.append(f"✓ {r.get('name')}: {r.get('got')}")
            else:
                lines.append(
                    f"✗ {r.get('name')}: expect={r.get('expected')} got={r.get('got') or r.get('error')}"
                )
        QMessageBox.information(self, "回归结果", "\n".join(lines[:40]))

    def _toggle_auto_roi_sample(self, checked: bool) -> None:
        if checked:
            ms = max(3000, int(self.auto_roi_spin.value()) * 1000)
            self._auto_roi_timer.start(ms)
            self._status.showMessage(f"定时刷样已开：每 {self.auto_roi_spin.value()} 秒")
        else:
            self._auto_roi_timer.stop()
            self._status.showMessage("定时刷样已关")

    def _auto_roi_tick(self) -> None:
        if not self.project:
            self.chk_auto_roi.setChecked(False)
            return
        try:
            if self.project.config.roi_presets:
                self._multi_roi_sample()
            else:
                dest = self._capture_dest_dir()
                if not dest:
                    return
                path = capture_adb(dest)
                self._add_captured_path(path, apply_last_roi=bool(self._last_roi))
                if self.import_canvas.roi() or self._last_roi:
                    if self._last_roi and not self.import_canvas.roi():
                        self.import_canvas.set_roi(self._last_roi)
                    self._auto_preview_boxes()
                    self._segment_current()
            self._status.showMessage("定时刷样：已采集一轮")
        except Exception as exc:
            self._status.showMessage(f"定时刷样失败: {exc}")

    def _show_confusion(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            QMessageBox.information(self, "提示", "请先训练")
            return
        try:
            report = compute_confusion(proj, ckpt)
        except Exception as exc:
            QMessageBox.warning(self, "失败", str(exc))
            return
        QMessageBox.information(self, "混淆矩阵 / 易错对", format_confusion_text(report))

    def _backup_project(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        try:
            path = backup_project(proj)
        except Exception as exc:
            QMessageBox.warning(self, "备份失败", str(exc))
            return
        QMessageBox.information(self, "已备份", str(path))
        self._status.showMessage(f"已备份: {path}")

    def _compare_projects(self) -> None:
        root = projects_root()
        if not root.is_dir():
            QMessageBox.information(self, "提示", f"尚无项目目录: {root}")
            return
        lines = ["多项目对比（样本数 / 最近 val_acc）：", ""]
        for p in sorted(root.iterdir()):
            if not (p / "config.json").is_file():
                continue
            try:
                proj = open_project(p)
                counts = proj.class_counts()
                total = sum(counts.values())
                scarce = scarce_classes(counts)
                runs = sorted(proj.runs_dir.glob("*/metrics.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                acc = "—"
                if runs:
                    import json

                    data = json.loads(runs[0].read_text(encoding="utf-8"))
                    if data.get("best_val_acc") is not None:
                        acc = f"{float(data['best_val_acc']):.0%}"
                tip = f"缺{len(scarce)}类" if scarce else "均衡尚可"
                lines.append(f"· {proj.config.game_id}: 样本 {total} · val {acc} · {tip}")
            except Exception as exc:
                lines.append(f"· {p.name}: 读取失败 ({exc})")
        QMessageBox.information(self, "多项目对比", "\n".join(lines[:40]))

    def _split_selected_box(self) -> None:
        before = self.import_canvas.selected_box_index()
        if not self.import_canvas.boxes():
            QMessageBox.information(self, "提示", "还没有绿框。请先「整行蓝框」+「自动预览切字」。")
            return
        if not self.import_canvas.split_selected_box_vertical():
            QMessageBox.information(self, "提示", "选中的框太窄，无法再拆")
            return
        after = self.import_canvas.selected_box_index()
        if before < 0:
            self._status.showMessage(f"未选中时已自动拆最宽的框 → 现已选中第 {after + 1} 框，可继续拆")
        else:
            self._status.showMessage(f"已拆开第 {before + 1} 框，可继续点「拆粘连」")
        self._preview_timer.start(300)

    def _capture_ld_window(self) -> None:
        dest = self._capture_dest_dir()
        if not dest:
            return
        title, ok = QInputDialog.getText(
            self, "窗口截图", "窗口标题包含：", text="雷电"
        )
        if not ok:
            return
        try:
            path = capture_window_by_title(dest, title.strip() or "雷电")
        except Exception as exc:
            QMessageBox.critical(self, "窗口截图失败", str(exc))
            return
        self._add_captured_path(path, apply_last_roi=bool(self._last_roi))

    def _multi_roi_sample(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        presets = list(proj.config.roi_presets)
        if not presets:
            QMessageBox.information(self, "提示", "请先在「更多」里保存多个 ROI 预设")
            return
        dest = self._capture_dest_dir()
        if not dest:
            return
        try:
            path = capture_adb(dest)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "ADB 失败",
                f"{exc}\n\n将使用当前图做多 ROI 切字（若已打开截图）",
            )
            path = self._current_import
            if not path:
                return
        self._add_captured_path(path)
        QApplication.processEvents()
        total = 0
        try:
            bgr = load_bgr(path)
        except Exception as exc:
            QMessageBox.critical(self, "读图失败", str(exc))
            return
        for preset in presets:
            boxes = []
            # auto segment inside each ROI
            try:
                _, crops, _ = segment_image(
                    path,
                    proj.config.preprocess,
                    roi=(preset.x, preset.y, preset.w, preset.h),
                    max_gap=self.gap_spin.value(),
                )
                if crops:
                    # convert to full-image boxes
                    boxes = [(c.x + preset.x, c.y + preset.y, c.w, c.h) for c in crops]
                else:
                    boxes = [(preset.x, preset.y, preset.w, preset.h)]
                crops2 = crops_from_full_boxes(bgr, boxes, proj.config.preprocess)
                paths = save_pending_chars(proj, path, crops2)
                total += len(paths)
            except Exception:
                continue
        self._reload_pending()
        self._refresh_project_info()
        self.tabs.setCurrentIndex(self.TAB_REVIEW)
        self._status.showMessage(f"多 ROI 刷样完成：共切出 {total} 个字")

    def _gold_compare_reflow(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        raw = self.gold_edit.text().strip()
        if not raw:
            QMessageBox.information(self, "提示", "请先填写金标，例如 1234万")
            return
        preds = self.import_canvas.predictions()
        if not preds:
            self._preview_recognize()
            preds = self.import_canvas.predictions()
        if not preds:
            return
        try:
            expected = tokenize_expected(raw, proj.config.classes)
        except Exception as exc:
            QMessageBox.warning(self, "金标无效", str(exc))
            return
        mismatches = compare_preds(expected, preds)
        if not mismatches:
            QMessageBox.information(self, "对比通过", f"与金标一致：{raw}")
            return
        boxes = self.import_canvas.boxes()
        if not self._current_import:
            return
        bgr = load_bgr(self._current_import)
        crop_list = []
        meta_mm = []
        for m in mismatches:
            i = int(m["index"])
            if i >= len(boxes):
                continue
            crops = crops_from_full_boxes(bgr, [boxes[i]], proj.config.preprocess)
            if not crops:
                continue
            crop_list.extend(crops)
            meta_mm.append(m)
        if not crop_list:
            QMessageBox.warning(self, "回流失败", "无法切出不符的字框")
            return
        paths = save_pending_chars(proj, self._current_import, crop_list)
        for p, m in zip(paths, meta_mm):
            add_hard_example(
                proj,
                p,
                reason="金标不符",
                pred=m.get("pred"),
                conf=m.get("conf"),
                expected=m.get("expected"),
            )
        n_added = len(paths)
        self._reload_pending()
        self._refresh_project_info()
        QMessageBox.information(
            self,
            "已回流",
            f"发现 {len(mismatches)} 处不符，已加入待审/难例 {n_added} 张。\n可去审核页继续标。",
        )
        self.tabs.setCurrentIndex(self.TAB_REVIEW)
        self._set_review_mode("hard")

    def _jump_to_last_labeled(self) -> None:
        if not self.project or not self._last_labeled_path:
            self._status.showMessage("还没有刚标注的样本")
            return
        target = self._last_labeled_path
        if not target.exists():
            # maybe renamed — match by name in labeled list
            self._set_review_mode("labeled")
            for i, (p, _) in enumerate(self._labeled):
                if p.name == target.name:
                    self._labeled_idx = i
                    self._show_current()
                    return
            self._status.showMessage("刚标的样本找不到了（可能已删除）")
            return
        self._set_review_mode("labeled")
        for i, (p, _) in enumerate(self._labeled):
            if p == target or p.name == target.name:
                self._labeled_idx = i
                self._show_current()
                self._status.showMessage(f"已跳到刚标：{display_label(self._labeled[i][1])}")
                return
        self._status.showMessage("刚标的样本不在列表中")

    # ---------- UX helpers ----------
    def _bootstrap_ui_prefs(self) -> None:
        prefs = self._ui_prefs
        geo = prefs.get("geometry")
        if geo:
            try:
                self.restoreGeometry(QByteArray.fromHex(str(geo).encode("ascii")))
            except Exception:
                pass
        self._try_autoload_last_project()
        mode = prefs.get("cut_mode") or "char"
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
            "cut_mode": self.import_canvas.mode() if hasattr(self, "import_canvas") else "char",
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

    def _toggle_work_more(self, checked: bool) -> None:
        self.work_more.setVisible(checked)
        self.btn_more.setText(
            "高级选项 ▴（点击收起）"
            if checked
            else "高级选项 ▾（ROI 预设 / 缩放 / 定时刷样…）"
        )
        update_prefs(work_more_open=checked)

    def _recent_onnx_paths(self) -> list[str]:
        raw = self._ui_prefs.get("recent_onnx_models") or []
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw if isinstance(x, str) and x.strip()]

    def _remember_onnx_path(self, onnx_path: Path) -> None:
        key = str(onnx_path.resolve())
        recent = [key] + [p for p in self._recent_onnx_paths() if p != key]
        recent = recent[:8]
        self._ui_prefs = update_prefs(recent_onnx_models=recent, last_verify_model=f"onnx:{key}")

    def _reload_verify_models(self, *, prefer_key: str | None = None) -> None:
        if not hasattr(self, "verify_model_combo"):
            return
        combo = self.verify_model_combo
        prev = prefer_key or combo.currentData()
        if not prev:
            prev = self._ui_prefs.get("last_verify_model")
        combo.blockSignals(True)
        combo.clear()
        refs = list_project_models(self.project, recent_onnx=self._recent_onnx_paths())
        if not refs:
            combo.addItem("（无模型：请先导出，或点「浏览 ONNX…」加载外部包）", None)
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)
            for ref in refs:
                combo.addItem(ref.display, ref.key())
            if prev:
                idx = combo.findData(prev)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        combo.blockSignals(False)
        ort_ok, ort_msg = check_onnxruntime_dependency()
        if hasattr(self, "verify_hint"):
            tip = "可加载外部导出包；圈选后点「用所选模型识别」。"
            if not ort_ok:
                tip += f" ONNX 需安装: {ort_msg}"
            self.verify_hint.setText(tip)

    def _selected_verify_model(self) -> ModelRef | None:
        if not hasattr(self, "verify_model_combo"):
            return None
        key = self.verify_model_combo.currentData()
        if not key or not isinstance(key, str):
            return None
        for ref in list_project_models(self.project, recent_onnx=self._recent_onnx_paths()):
            if ref.key() == key:
                return ref
        return None

    def _browse_verify_onnx(self) -> None:
        start = str(Path.home())
        if self.project and self.project.exports_dir.is_dir():
            start = str(self.project.exports_dir)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择导出的 digits.onnx（其它电脑拷贝过来的也可以）",
            start,
            "ONNX (*.onnx);;All (*)",
        )
        if not path:
            folder = QFileDialog.getExistingDirectory(
                self,
                "或选择导出目录（内含 digits.onnx + digits.labels）",
                start,
            )
            if not folder:
                return
            path = folder
        try:
            ref = resolve_onnx_pack(Path(path))
        except ValueError as exc:
            QMessageBox.warning(self, "无法加载", str(exc))
            return
        ok, msg = check_onnxruntime_dependency()
        if not ok:
            QMessageBox.warning(
                self,
                "缺少运行时",
                f"{msg}\n\n仍已加入列表；安装 onnxruntime 后即可用所选模型识别。",
            )
        self._remember_onnx_path(ref.path)
        self._reload_verify_models(prefer_key=ref.key())
        self._status.showMessage(f"已加入外部模型：{ref.display}")

    def _verify_recognize_selected(self) -> None:
        self._recognize_with_model(self._selected_verify_model(), silent=False)

    def _recognize_with_model(self, model: ModelRef | None, *, silent: bool = False) -> None:
        proj = self._require_project()
        if not proj:
            return
        if model is None:
            if not silent:
                QMessageBox.information(
                    self,
                    "提示",
                    "请先在「验模型」下拉中选择模型，或点「浏览 ONNX…」加载其它电脑导出的包。",
                )
            return
        if model.kind == "onnx":
            ok, msg = check_onnxruntime_dependency()
            if not ok:
                if not silent:
                    QMessageBox.warning(self, "无法运行 ONNX", msg)
                return
        if not self._current_import:
            if not silent:
                QMessageBox.information(self, "提示", "请先打开或截取一张图")
            return
        boxes = self.import_canvas.boxes()
        if not boxes:
            if not silent:
                self._auto_preview_boxes()
                boxes = self.import_canvas.boxes()
        if not boxes:
            if not silent:
                QMessageBox.warning(self, "提示", "请先框出字框或蓝 ROI")
            return
        try:
            bgr = load_bgr(self._current_import)
            conf = float(self.conf_spin.value()) if hasattr(self, "conf_spin") else 0.5
            text, parts = predict_boxes_with_model(
                proj, bgr, boxes, model, conf_threshold=conf
            )
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "识别失败", str(exc))
            return
        self.import_canvas.set_predictions(parts)
        if hasattr(self, "preview_big"):
            self.preview_big.setText(text or "（空）")
        detail = " ".join(f"{display_label(l)}({c:.0%})" for l, c in parts)
        self.trial_result.setText(f"验模明细（{model.display}）：{detail}")
        update_prefs(last_verify_model=model.key())
        if not silent:
            self._status.showMessage(f"识别：{text} ← {model.display}")

    def _undo_contextual(self) -> None:
        idx = self.tabs.currentIndex()
        if idx == self.TAB_WORK:
            self._undo_char_box()
        elif idx == self.TAB_REVIEW:
            self._undo_last_label()

    def _redo_contextual(self) -> None:
        if self.tabs.currentIndex() != self.TAB_WORK:
            return
        if self.import_canvas.redo_box_edit():
            self._update_box_count()
            self._refresh_selected_box_ui()
            self._status.showMessage("已重做字框变更")

    def _enter_contextual(self) -> None:
        if self.tabs.currentIndex() == self.TAB_WORK:
            self._segment_current()
        elif self.tabs.currentIndex() == self.TAB_REVIEW:
            self._confirm_prediction()

    def _highlight_suggested_label(self, label: str | None) -> None:
        for b in self._digit_btns.values():
            b.setObjectName("digitBtn")
            b.style().unpolish(b)
            b.style().polish(b)
        for b in self._unit_btns.values():
            b.setObjectName("unitBtn")
            b.style().unpolish(b)
            b.style().polish(b)
        if not label:
            return
        key = normalize_label(label) if label else ""
        # map wan/yi folder names back
        from game_digit_trainer.labels import UNIT_CLASS_NAMES

        btn = self._digit_btns.get(key) or self._unit_btns.get(label)
        if btn is None and key in UNIT_CLASS_NAMES:
            # display 万/亿 buttons keyed by 万/亿
            for lab, b in self._unit_btns.items():
                if normalize_label(lab) == key:
                    btn = b
                    break
        if btn is None:
            # try display match
            for lab, b in list(self._digit_btns.items()) + list(self._unit_btns.items()):
                if normalize_label(lab) == key or display_label(key) == lab:
                    btn = b
                    break
        if btn is not None:
            btn.setObjectName("suggestedBtn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _update_review_progress(self) -> None:
        if not hasattr(self, "review_progress") or not self.project:
            return
        labeled_n = sum(self.project.class_counts().values())
        pending_n = len(self.project.pending_files())
        total = labeled_n + pending_n
        self.review_progress.setMaximum(max(total, 1))
        self.review_progress.setValue(labeled_n)
        self.review_progress.setFormat(f"已标 {labeled_n} / 共 {total}（待审 {pending_n}）")

    def _guide_next(self) -> None:
        self._guide_step += 1
        if self._guide_step >= 3:
            self._guide_skip()
            return
        self._update_guide_banner()
        if self._guide_step == 1:
            self.tabs.setCurrentIndex(self.TAB_WORK)
        elif self._guide_step == 2:
            self.tabs.setCurrentIndex(self.TAB_REVIEW)

    def _guide_skip(self) -> None:
        self.guide_banner.setVisible(False)
        self._ui_prefs = update_prefs(guide_done=True)

    def _update_guide_banner(self) -> None:
        steps = [
            "① 点「框选截屏 F2」截取游戏数字区域（截完会自动进入框字）",
            "② 每个数字拖一个绿框（可拖角微调）→ 按 Enter「确认切字」",
            "③ 到审核页：空格确认预测，或按数字键标注；标错可改",
        ]
        step = max(0, min(self._guide_step, len(steps) - 1))
        self.guide_label.setText(f"快速上手（{step + 1}/3）：{steps[step]}")

    def _recapture_same_roi(self) -> None:
        """优先 ADB；失败则提示用 F2。截完套用上次 ROI。"""
        if not self._last_roi and not (self.project and self.project.config.roi_presets):
            QMessageBox.information(
                self,
                "提示",
                "还没有可用的 ROI。请先拖一个蓝框，或保存/套用 ROI 预设。",
            )
            return
        if not self._last_roi and self.project and self.project.config.roi_presets:
            p = self.project.config.roi_presets[0]
            name = self.roi_preset_combo.currentData()
            preset = next((x for x in self.project.config.roi_presets if x.name == name), None) or p
            self._last_roi = (preset.x, preset.y, preset.w, preset.h)
        # try ADB first for emulator workflow
        dest = self._capture_dest_dir()
        if not dest:
            return
        try:
            devices = list_adb_devices()
            if devices:
                path = capture_adb(dest)
                self._add_captured_path(path, apply_last_roi=True)
                return
        except Exception:
            pass
        QMessageBox.information(
            self,
            "再截同 ROI",
            "未检测到 ADB 设备。请用「框选截屏 F2」截一张，截完会自动套用上次蓝框。",
        )
        # mark next region capture to apply roi
        self._pending_apply_roi = True
        self._capture_region()


def run_gui() -> int:
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    return app.exec()
