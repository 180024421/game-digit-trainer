"""Shared symbols for GUI mixins (star-imported by tab modules)."""
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
from game_digit_trainer.quality import (
    export_quality_report,
    line_export_quality_report,
    verify_onnx_runtime,
)
from game_digit_trainer.line_eval import evaluate_line_samples, format_line_eval_report
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
from game_digit_trainer.predict_line import predict_line_gray, predict_line_roi, load_line_checkpoint
from game_digit_trainer.line_data import (
    LINE_DATASET_KEY,
    bootstrap_chars_from_line_samples,
    clear_bootstrap_chars,
    clear_line_samples,
    confirm_line_pending,
    count_line_labeled,
    coverage_fill_suggestions,
    delete_line_sample,
    get_line_pending_hint,
    line_coverage_report,
    list_line_labeled,
    list_line_pending,
    load_line_pending_hints,
    save_line_pending,
    save_line_sample,
    set_line_pending_hint,
    update_line_label,
)

__all__ = [n for n in dir() if not n.startswith("_")]
