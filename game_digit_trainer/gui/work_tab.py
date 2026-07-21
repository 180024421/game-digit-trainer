"""① 截图切字页：采集、框选、识别、验模型。"""
from __future__ import annotations

from game_digit_trainer.gui.deps import *  # noqa: F403


class WorkTabMixin:
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
        mid_outer = QVBoxLayout(mid)
        mid_outer.setContentsMargins(0, 0, 0, 0)
        mid_outer.setSpacing(0)

        self.work_scroll = QScrollArea()
        self.work_scroll.setWidgetResizable(True)
        self.work_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.work_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.work_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        mid_outer.addWidget(self.work_scroll)

        scroll_body = QWidget()
        mid_l = QVBoxLayout(scroll_body)
        mid_l.setContentsMargins(0, 0, 0, 0)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("框字"))
        self.btn_mode_roi = QPushButton("整行蓝框（推荐）")
        self.btn_mode_char = QPushButton("逐字绿框")
        self.btn_mode_roi.setObjectName("primaryBtn")
        self.btn_mode_roi.setToolTip("主路径：圈一整段数字 → 加入行待审 / 预览识别（行模型）")
        self.btn_mode_char.setToolTip("兼容：每个数字拖绿框后确认切字（单字模型 / Studio 旧包）")
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
        btn_seg.setObjectName("")
        btn_seg.setToolTip("快捷键 Enter：按绿框切成单字进待审")
        btn_seg.clicked.connect(self._segment_current)
        btn_line_pending = QPushButton("加入行待审")
        btn_line_pending.setObjectName("successBtn")
        btn_line_pending.setToolTip(
            "用「整行蓝框」圈住数字后点此：整行进 ② 行待审，稍后填整串文字（不切字）"
        )
        btn_line_pending.clicked.connect(self._add_line_pending)
        mode_row.addWidget(btn_undo)
        mode_row.addWidget(btn_clear_boxes)
        mode_row.addWidget(btn_seg)
        mode_row.addWidget(btn_line_pending)
        mid_l.addLayout(mode_row)

        self.work_hint = QLabel(
            "推荐：整行蓝框圈数字 →「加入行待审」或「预览识别」。逐字绿框仅用于单字模型/兼容 Studio。"
        )
        self.work_hint.setObjectName("hintLabel")
        self.work_hint.setWordWrap(True)
        mid_l.addWidget(self.work_hint)

        # 工具 / 验模型 / 高级选项放在画布上方，避免被画布挤出窗口
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
        btn_preview = QPushButton("识别")
        btn_preview.setObjectName("primaryBtn")
        btn_preview.setToolTip(
            "优先用下方所选/最新行模型：蓝框整行一次出串；无行模型时回退单字切字识别。"
        )
        btn_preview.clicked.connect(self._preview_recognize)
        btn_wrong_pending = QPushButton("读错→行待审")
        btn_wrong_pending.setObjectName("dangerBtn")
        btn_wrong_pending.setToolTip(
            "当前蓝框识别不对时：把该行加入 ② 行待审；若金标已填会带过去方便确认"
        )
        btn_wrong_pending.clicked.connect(self._wrong_to_line_pending)
        btn_split = QPushButton("拆粘连")
        btn_split.setToolTip("点选绿框后拆开；未选中时自动拆最宽的框")
        btn_split.clicked.connect(self._split_selected_box)
        main_tools.addWidget(btn_auto)
        main_tools.addWidget(btn_preview)
        main_tools.addWidget(btn_wrong_pending)
        main_tools.addWidget(btn_split)
        self.btn_verify_toggle = QPushButton("验模型 ▾")
        self.btn_verify_toggle.setCheckable(True)
        self.btn_verify_toggle.setChecked(False)
        self.btn_verify_toggle.setToolTip("展开后可选 exports / 行模型 / 外部 ONNX 做对比识别")
        self.btn_verify_toggle.toggled.connect(self._toggle_verify_panel)
        main_tools.addWidget(self.btn_verify_toggle)
        main_tools.addStretch()
        mid_l.addLayout(main_tools)

        self.verify_box = QGroupBox("验模型（可选导出 ONNX 或本机 checkpoint）")
        verify_l = QVBoxLayout(self.verify_box)
        verify_row = QHBoxLayout()
        verify_row.addWidget(QLabel("模型"))
        self.verify_model_combo = QComboBox()
        self.verify_model_combo.setMinimumWidth(200)
        self.verify_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.verify_model_combo.setToolTip("列表含本工程 exports、runs，以及你浏览过的外部 ONNX 包")
        btn_verify_refresh = QPushButton("刷新")
        btn_verify_refresh.setToolTip("重新扫描本工程导出与 checkpoint")
        btn_verify_refresh.clicked.connect(self._reload_verify_models)
        btn_verify_browse = QPushButton("浏览 ONNX…")
        btn_verify_browse.setToolTip("选择其它电脑拷来的 digits.onnx / digits_line.onnx 或导出目录")
        btn_verify_browse.clicked.connect(self._browse_verify_onnx)
        btn_verify_run = QPushButton("用所选模型识别")
        btn_verify_run.setObjectName("primaryBtn")
        btn_verify_run.setToolTip("用下拉所选模型识别当前蓝框/字框")
        btn_verify_run.clicked.connect(self._verify_recognize_selected)
        verify_row.addWidget(self.verify_model_combo, 1)
        verify_row.addWidget(btn_verify_refresh)
        verify_row.addWidget(btn_verify_browse)
        verify_row.addWidget(btn_verify_run)
        verify_l.addLayout(verify_row)
        self.verify_hint = QLabel(
            "默认可直接点「识别」（优先最新行模型）。此处用于对比导出包或历史 checkpoint。"
        )
        self.verify_hint.setObjectName("hintLabel")
        self.verify_hint.setWordWrap(True)
        verify_l.addWidget(self.verify_hint)
        self.verify_box.setVisible(False)
        mid_l.addWidget(self.verify_box)

        more_toggle_row = QHBoxLayout()
        self.btn_more = QPushButton("高级选项 ▾（ROI 预设 / 缩放 / 定时刷样…）")
        self.btn_more.setCheckable(True)
        self.btn_more.setChecked(False)
        self.btn_more.setMinimumHeight(36)
        self.btn_more.setToolTip("展开或收起进阶工具；内容变长时可上下滚动查看")
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
        btn_zoom_fit.clicked.connect(lambda: self.import_canvas.reset_view())
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
        self.preprocess_preview.setToolTip(
            "仅给「自动切字」看效果。行模型识别用原始灰度，与这里二值预览无关。"
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
        btn_multi.setToolTip("按 ROI 预设截图：默认写入「行待审」（主路径）；取消勾选则切单字")
        btn_multi.clicked.connect(self._multi_roi_sample)
        extra_tools.addWidget(btn_merge)
        extra_tools.addWidget(btn_fix)
        extra_tools.addWidget(btn_multi)
        extra_tools.addStretch()
        more_l.addLayout(extra_tools)

        auto_row = QHBoxLayout()
        self.chk_auto_roi = QCheckBox("定时刷样")
        self.chk_auto_roi.setToolTip("按间隔自动截图；默认写入行待审（与主路径一致）")
        self.auto_roi_spin = QSpinBox()
        self.auto_roi_spin.setRange(3, 120)
        self.auto_roi_spin.setValue(8)
        self.auto_roi_spin.setSuffix(" 秒")
        self.chk_auto_roi.toggled.connect(self._toggle_auto_roi_sample)
        self.chk_auto_line_pending = QCheckBox("刷样→行待审")
        self.chk_auto_line_pending.setChecked(True)
        self.chk_auto_line_pending.setToolTip(
            "勾选：定时/多ROI 写入行待审（推荐）。取消：走旧切字待审。"
        )
        self.chk_lowconf_line_pending = QCheckBox("低置信自动入队")
        self.chk_lowconf_line_pending.setChecked(True)
        self.chk_lowconf_line_pending.setToolTip(
            "行识别置信低于阈值时，自动把当前蓝框加入行待审（并预填预测）"
        )
        auto_row.addWidget(self.chk_auto_roi)
        auto_row.addWidget(self.auto_roi_spin)
        auto_row.addWidget(self.chk_auto_line_pending)
        auto_row.addWidget(self.chk_lowconf_line_pending)
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

        self.import_canvas = ImageCanvas()
        self.import_canvas.setMinimumHeight(280)
        self.import_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.import_canvas.roi_changed.connect(self._on_roi_changed)
        self.import_canvas.boxes_changed.connect(self._on_boxes_changed)
        self.import_canvas.view_changed.connect(self._on_view_changed)
        self.import_canvas.selection_changed.connect(self._on_box_selection_changed)
        mid_l.addWidget(self.import_canvas)

        work_footer = QWidget()
        work_footer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        foot_l = QVBoxLayout(work_footer)
        foot_l.setContentsMargins(0, 4, 0, 0)
        foot_l.setSpacing(4)

        self.preview_big = QLabel("识别预览：框选数字后点「预览识别」或「用所选模型识别」")
        self.preview_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_big.setMinimumHeight(44)
        self.preview_big.setStyleSheet(
            "background:#111827; color:#fbbf24; border-radius:10px; font-size:22px; font-weight:800; padding:4px;"
        )
        foot_l.addWidget(self.preview_big)

        sel_bar = QHBoxLayout()
        sel_bar.setContentsMargins(0, 0, 0, 0)
        self.selected_crop_preview = QLabel("未选")
        self.selected_crop_preview.setFixedSize(48, 48)
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
        foot_l.addLayout(sel_bar)
        self._syncing_box_spins = False

        gold_row = QHBoxLayout()
        gold_row.addWidget(QLabel("金标"))
        self.gold_edit = QLineEdit()
        self.gold_edit.setPlaceholderText("例如 1.2万 或 2:03 — 与预览对比，错字进难例/待审")
        btn_gold = QPushButton("对比回流")
        btn_gold.clicked.connect(self._gold_compare_reflow)
        btn_reg_add = QPushButton("加入回归集")
        btn_reg_add.setToolTip("把当前图+金标存为固定回归用例")
        btn_reg_add.clicked.connect(self._add_regression_case)
        btn_line_save = QPushButton("存行样本")
        btn_line_save.setToolTip("把当前蓝框/宽框 + 金标存入 lines/，供行模型训练（推荐多存真实 HUD）")
        btn_line_save.clicked.connect(self._save_line_sample)
        gold_row.addWidget(self.gold_edit, 1)
        gold_row.addWidget(btn_gold)
        gold_row.addWidget(btn_reg_add)
        gold_row.addWidget(btn_line_save)
        foot_l.addLayout(gold_row)

        self.trial_result = QLabel("预览明细：训练后 / 验模型后显示")
        self.trial_result.setObjectName("hintLabel")
        self.trial_result.setWordWrap(True)
        foot_l.addWidget(self.trial_result)

        mid_l.addWidget(work_footer)

        self.work_scroll.setWidget(scroll_body)
        splitter.addWidget(mid)

        self._set_cut_mode("roi")

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
        self._set_cut_mode("roi")
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
                "逐字模式（兼容）：拖绿框 → Enter 切字。主路径请改用「整行蓝框」+ 行待审。"
            )
            self.btn_mode_char.setObjectName("primaryBtn")
            self.btn_mode_roi.setObjectName("")
            if self.import_canvas.roi():
                self.import_canvas.zoom_to_rect(self.import_canvas.roi(), margin=0.35)
        else:
            self.work_hint.setText(
                "主路径：拖蓝框圈数字行 →「加入行待审」或点「识别」。ROI 预设在高级选项里。"
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
                self._status.showMessage(
                    "拖得太大已改为「整行蓝框」— 行识别可直接用；"
                    "要逐字请点「清空框」后框单个字，或点绿框边线再调"
                )

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
            from game_digit_trainer.box_ops import filter_giant_auto_boxes
            from game_digit_trainer.segment import boxes_from_region

            roi = self.import_canvas.roi()
            fixed = boxes_from_region(
                bgr,
                roi,
                self.project.config.preprocess,
                max_gap=self.gap_spin.value(),
            )
            h, w = bgr.shape[:2]
            filtered = filter_giant_auto_boxes(fixed, w, h)
            if fixed and not filtered:
                self._status.showMessage(
                    "自动切字得到整图大框已忽略 — 请缩小蓝框，或改小「间距」后重试"
                )
                return
            self.import_canvas.set_boxes(filtered)
            self._refresh_import_preview(keep_manual_boxes=True)
            gap = self.gap_spin.value()
            self._status.showMessage(f"自动预览 {len(filtered)} 个字框（间距={gap}）")
        except Exception as exc:
            QMessageBox.warning(self, "自动切字失败", str(exc))

    def _prepare_recognize_boxes(self, bgr) -> list[tuple[int, int, int, int]]:
        """识别前解析字框：区域优先自动切；多细绿框用手调。"""
        assert self.project is not None
        self._apply_preprocess_ui()
        boxes, mode = resolve_recognize_boxes(
            bgr,
            preprocess=self.project.config.preprocess,
            roi=self.import_canvas.roi(),
            boxes=list(self.import_canvas.boxes()),
            max_gap=int(self.gap_spin.value()),
        )
        if boxes and mode in {"roi", "wide_box", "full"}:
            self.import_canvas.set_boxes(boxes)
            self._update_box_count()
        return boxes

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
        if not self._current_import:
            if not silent:
                QMessageBox.information(self, "提示", "请先打开一张截图")
            return
        # 优先行模型（蓝框/宽区域 → 整串，不切字）
        if self._try_line_recognize(silent=silent):
            return
        ckpt = latest_checkpoint(proj)
        if not ckpt:
            if not silent:
                QMessageBox.information(
                    self,
                    "提示",
                    "还没有行模型 / 单字模型。\n"
                    "推荐：③ 训练页点「训练行模型」；或先训单字再用切字识别。",
                )
            if hasattr(self, "preview_big"):
                self.preview_big.setText("识别预览：需先训练行模型或单字模型")
            return
        try:
            bgr = load_bgr(self._current_import)
            boxes = self._prepare_recognize_boxes(bgr)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "预览失败", str(exc))
            return
        if not boxes:
            if not silent:
                QMessageBox.warning(
                    self,
                    "提示",
                    "无行模型且切字为空。请先「训练行模型」，或用蓝框圈紧后再试。",
                )
            return
        try:
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
        self.trial_result.setText(f"预览明细（单字路径）：{detail}")
        self._last_recognize_text = text or ""
        if not silent:
            self._status.showMessage(f"识别预览：{text}")

    def _region_for_line(self) -> tuple[int, int, int, int] | None:
        """行识别区域：优先蓝框，否则单个宽绿框。"""
        from game_digit_trainer.segment import looks_like_region_box

        roi = self.import_canvas.roi()
        if roi:
            return tuple(int(v) for v in roi)  # type: ignore[return-value]
        boxes = list(self.import_canvas.boxes())
        if len(boxes) == 1 and looks_like_region_box(boxes[0]):
            return boxes[0]
        return None

    def _try_line_recognize(self, *, silent: bool = False) -> bool:
        proj = self.project
        if not proj or not self._current_import:
            return False
        line_ckpt = latest_line_checkpoint(proj)
        if not line_ckpt:
            return False
        region = self._region_for_line()
        if region is None:
            return False
        try:
            bgr = load_bgr(self._current_import)
            text, parts, mean_conf = predict_line_roi(proj, bgr, region, line_ckpt)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "行识别失败", str(exc))
            return True
        if hasattr(self, "preview_big"):
            self.preview_big.setText(text or "（空）")
        detail = " ".join(f"{display_label(l)}({c:.0%})" for l, c in parts) or f"mean={mean_conf:.0%}"
        self.trial_result.setText(f"行模型明细：{detail}")
        self.import_canvas.set_predictions([])
        self._last_recognize_text = text or ""
        self._last_recognize_conf = float(mean_conf)
        if not silent:
            self._status.showMessage(f"行识别：{text} ← {line_ckpt.parent.name}/line_best.pt")
            thr = float(self.conf_spin.value()) if hasattr(self, "conf_spin") else 0.7
            if mean_conf < thr and self._current_import:
                try:
                    from game_digit_trainer.hard_examples import add_hard_example

                    add_hard_example(
                        proj,
                        self._current_import,
                        reason=f"行识别低置信 {mean_conf:.0%}",
                        expected="",
                        pred=text,
                    )
                except Exception:
                    pass
                auto_q = (
                    hasattr(self, "chk_lowconf_line_pending")
                    and self.chk_lowconf_line_pending.isChecked()
                )
                if auto_q and region is not None:
                    try:
                        path = save_line_pending(
                            proj,
                            bgr,
                            region,
                            source_name=f"lowconf_{self._current_import.stem}",
                            pred=text or "",
                            conf=float(mean_conf),
                            hint="",
                        )
                        self._status.showMessage(
                            f"行识别：{text}（置信 {mean_conf:.0%} 偏低，已入行待审 {path.name}）"
                        )
                        self._update_header()
                    except Exception:
                        self._status.showMessage(
                            f"行识别：{text}（置信 {mean_conf:.0%} 偏低，已记入难例）"
                        )
                else:
                    self._status.showMessage(
                        f"行识别：{text}（置信 {mean_conf:.0%} 偏低，已记入难例）"
                    )
        return True

    def _save_line_sample(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if not self._current_import:
            QMessageBox.information(self, "提示", "请先打开截图")
            return
        text = (self.gold_edit.text() or "").strip()
        if not text:
            QMessageBox.information(self, "提示", "请先填写金标，例如 3920万")
            return
        try:
            bgr = load_bgr(self._current_import)
            region = self._region_for_line()
            path = save_line_sample(proj, bgr, region, text)
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._status.showMessage(f"已存行样本：{path.name}（金标 {text}）")
        self._refresh_project_info()

    def _add_line_pending(self) -> None:
        """蓝框圈选后加入行待审（先不填字）。"""
        proj = self._require_project()
        if not proj:
            return
        if not self._current_import:
            QMessageBox.information(self, "提示", "请先打开或截取一张图")
            return
        region = self._region_for_line()
        if region is None:
            QMessageBox.information(
                self,
                "提示",
                "请先点「整行蓝框」，拖出蓝框圈住一整行数字，再点「加入行待审」。",
            )
            return
        try:
            bgr = load_bgr(self._current_import)
            path = save_line_pending(
                proj, bgr, region, source_name=self._current_import.name
            )
        except Exception as exc:
            QMessageBox.warning(self, "加入失败", str(exc))
            return
        self._update_header()
        self._status.showMessage(f"已加入行待审：{path.name}")
        reply = QMessageBox.question(
            self,
            "已加入行待审",
            f"已保存：{path.name}\n\n现在去 ②「行待审」填写整串文字吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.tabs.setCurrentIndex(self.TAB_REVIEW)
            self._set_review_mode("line")

    def _wrong_to_line_pending(self) -> None:
        """识别出错：当前蓝框 → 行待审；金标框若有内容会预填到审核页。"""
        proj = self._require_project()
        if not proj:
            return
        if not self._current_import:
            QMessageBox.information(self, "提示", "请先打开截图并框选数字")
            return
        region = self._region_for_line()
        if region is None:
            QMessageBox.information(self, "提示", "请先用「整行蓝框」圈住读错的那一行")
            return
        pred = getattr(self, "_last_recognize_text", "") or ""
        hint = (self.gold_edit.text() or "").strip()
        try:
            bgr = load_bgr(self._current_import)
            path = save_line_pending(
                proj, bgr, region, source_name=f"wrong_{self._current_import.stem}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "加入失败", str(exc))
            return
        # 把正确金标（若已填）或空串记到会话，审核页可预填
        self._pending_line_prefill = hint
        self._update_header()
        msg = f"已加入行待审：{path.name}"
        if pred:
            msg += f"\n模型误读为「{pred}」"
        if hint:
            msg += f"\n将预填金标「{hint}」"
        else:
            msg += "\n请到 ② 填写正确整串"
        self._status.showMessage(msg.split("\n")[0])
        go = QMessageBox.question(
            self,
            "读错已进待审",
            msg + "\n\n现在去 ②「行待审」确认吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if go == QMessageBox.StandardButton.Yes:
            self.tabs.setCurrentIndex(self.TAB_REVIEW)
            self._set_review_mode("line")
            if hint and hasattr(self, "line_label_edit"):
                self.line_label_edit.setText(hint)

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
                roi=self.import_canvas.roi(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "失败", str(exc))
            return
        self._status.showMessage(f"已加入回归集：{item.get('name')}（共 {len(load_cases(proj))} 条）")

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
            use_line = (
                hasattr(self, "chk_auto_line_pending") and self.chk_auto_line_pending.isChecked()
            )
            if self.project.config.roi_presets:
                self._multi_roi_sample(silent=True)
            elif use_line:
                n = self._auto_sample_line_pending_once()
                self._status.showMessage(f"定时刷样：已写入行待审 {n} 张")
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
                self._status.showMessage("定时刷样：已采集一轮（切字）")
        except Exception as exc:
            self._status.showMessage(f"定时刷样失败: {exc}")

    def _prefer_line_auto_sample(self) -> bool:
        return hasattr(self, "chk_auto_line_pending") and self.chk_auto_line_pending.isChecked()

    def _auto_sample_line_pending_once(self) -> int:
        """截一帧（或用当前图）+ 当前蓝框/上次 ROI → 行待审。返回写入张数。"""
        proj = self.project
        if not proj:
            return 0
        dest = self._capture_dest_dir()
        path = None
        if dest:
            try:
                path = capture_adb(dest)
                self._add_captured_path(path, apply_last_roi=bool(self._last_roi))
            except Exception:
                path = self._current_import
        else:
            path = self._current_import
        if not path:
            return 0
        if self._last_roi and not self.import_canvas.roi():
            self.import_canvas.set_roi(self._last_roi)
        region = self._region_for_line()
        if region is None and self._last_roi:
            region = tuple(int(v) for v in self._last_roi)  # type: ignore[assignment]
        if region is None:
            return 0
        bgr = load_bgr(path)
        pred, conf = "", 0.0
        line_ckpt = latest_line_checkpoint(proj)
        if line_ckpt:
            try:
                pred, _parts, conf = predict_line_roi(proj, bgr, region, line_ckpt)
            except Exception:
                pred, conf = "", 0.0
        out = save_line_pending(
            proj,
            bgr,
            region,
            source_name=f"auto_{Path(path).stem}",
            pred=pred,
            conf=float(conf),
        )
        self._update_header()
        return 1 if out else 0

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

    def _multi_roi_sample(self, silent: bool = False) -> None:
        proj = self._require_project()
        if not proj:
            return
        presets = list(proj.config.roi_presets)
        if not presets:
            if not silent:
                QMessageBox.information(self, "提示", "请先在「更多」里保存多个 ROI 预设")
            return
        dest = self._capture_dest_dir()
        if not dest:
            return
        try:
            path = capture_adb(dest)
        except Exception as exc:
            if not silent:
                QMessageBox.warning(
                    self,
                    "ADB 失败",
                    f"{exc}\n\n将使用当前图做多 ROI（若已打开截图）",
                )
            path = self._current_import
            if not path:
                return
        self._add_captured_path(path)
        QApplication.processEvents()
        try:
            bgr = load_bgr(path)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "读图失败", str(exc))
            return

        if self._prefer_line_auto_sample():
            total = 0
            line_ckpt = latest_line_checkpoint(proj)
            for preset in presets:
                region = (preset.x, preset.y, preset.w, preset.h)
                pred, conf = "", 0.0
                if line_ckpt:
                    try:
                        pred, _parts, conf = predict_line_roi(proj, bgr, region, line_ckpt)
                    except Exception:
                        pred, conf = "", 0.0
                try:
                    save_line_pending(
                        proj,
                        bgr,
                        region,
                        source_name=f"roi_{preset.name}_{Path(path).stem}",
                        pred=pred,
                        conf=float(conf),
                    )
                    total += 1
                except Exception:
                    continue
            self._update_header()
            self._refresh_project_info()
            msg = f"多 ROI 刷样：已写入行待审 {total} 张"
            self._status.showMessage(msg)
            if not silent:
                self.tabs.setCurrentIndex(self.TAB_REVIEW)
                self._set_review_mode("line")
                QMessageBox.information(self, "完成", msg + "\n请到 ②「行待审」确认金标。")
            return

        total = 0
        for preset in presets:
            try:
                _, crops, _ = segment_image(
                    path,
                    proj.config.preprocess,
                    roi=(preset.x, preset.y, preset.w, preset.h),
                    max_gap=self.gap_spin.value(),
                )
                if crops:
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
        if not silent:
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
        line_ckpt = latest_line_checkpoint(proj)
        region = self._region_for_line()
        if line_ckpt and region is not None and self._current_import:
            if not getattr(self, "_last_recognize_text", None):
                self._preview_recognize()
            pred = getattr(self, "_last_recognize_text", "") or ""
            if pred == raw:
                QMessageBox.information(self, "对比通过", f"与金标一致：{raw}")
                return
            conf = float(getattr(self, "_last_recognize_conf", 0.0) or 0.0)
            try:
                bgr = load_bgr(self._current_import)
                path = save_line_pending(
                    proj,
                    bgr,
                    region,
                    source_name=f"mismatch_{self._current_import.stem}",
                    pred=pred,
                    conf=conf,
                    hint=raw,
                )
            except Exception as exc:
                QMessageBox.warning(self, "加入失败", str(exc))
                return
            self._pending_line_prefill = raw
            self._update_header()
            go = QMessageBox.question(
                self,
                "金标不符",
                f"模型「{pred}」≠ 金标「{raw}」\n已加入行待审：{path.name}\n\n现在去确认吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if go == QMessageBox.StandardButton.Yes:
                self.tabs.setCurrentIndex(self.TAB_REVIEW)
                self._set_review_mode("line")
                if hasattr(self, "line_label_edit"):
                    self.line_label_edit.setText(raw)
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

    def _toggle_work_more(self, checked: bool) -> None:
        self.work_more.setVisible(checked)
        self.btn_more.setText(
            "高级选项 ▴（点击收起）"
            if checked
            else "高级选项 ▾（ROI 预设 / 缩放 / 定时刷样…）"
        )
        update_prefs(work_more_open=checked)
        if checked and hasattr(self, "work_scroll"):
            # 展开后内容变长，滚到高级区方便操作
            QTimer.singleShot(0, lambda: self.work_scroll.ensureWidgetVisible(self.btn_more))

    def _toggle_verify_panel(self, checked: bool) -> None:
        if hasattr(self, "verify_box"):
            self.verify_box.setVisible(checked)
        if hasattr(self, "btn_verify_toggle"):
            self.btn_verify_toggle.setText("验模型 ▴" if checked else "验模型 ▾")
        if checked:
            self._reload_verify_models()


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
                    "请先在「验模型」下拉中选择模型，或点「浏览 ONNX…」加载外部包。",
                )
            return
        if not self._current_import:
            if not silent:
                QMessageBox.information(self, "提示", "请先打开或截取一张图")
            return

        if model.kind == "line":
            region = self._region_for_line()
            if region is None:
                if not silent:
                    QMessageBox.warning(self, "提示", "行模型请先用「整行蓝框」圈住数字行")
                return
            try:
                bgr = load_bgr(self._current_import)
                text, parts, mean_conf = predict_line_roi(proj, bgr, region, model.path)
            except Exception as exc:
                if not silent:
                    QMessageBox.critical(self, "行识别失败", str(exc))
                return
            self.import_canvas.set_predictions([])
            if hasattr(self, "preview_big"):
                self.preview_big.setText(text or "（空）")
            detail = " ".join(f"{display_label(l)}({c:.0%})" for l, c in parts) or f"mean={mean_conf:.0%}"
            self.trial_result.setText(f"行模型明细（{model.display}）：{detail}")
            update_prefs(last_verify_model=model.key())
            if not silent:
                self._status.showMessage(f"行识别：{text} ← {model.display}")
            return

        if model.kind == "onnx":
            ok, msg = check_onnxruntime_dependency()
            if not ok:
                if not silent:
                    QMessageBox.warning(self, "无法运行 ONNX", msg)
                return
        try:
            bgr = load_bgr(self._current_import)
            boxes = self._prepare_recognize_boxes(bgr)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "识别失败", str(exc))
            return
        if not boxes:
            if not silent:
                QMessageBox.warning(
                    self,
                    "提示",
                    "区域内没有切出字。请用「整行蓝框」圈紧数字，或调节「间距」后再试。",
                )
            return
        try:
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
            "① 框选截屏 F2 截数字区域 → 用「整行蓝框」圈住一行",
            "② 点「加入行待审」→ ② 审核「行待审」填整串确认（约 30～50 条更好）",
            "③ 「训练行模型」→ 蓝框「识别」试读；要给 Studio 再「导出行 ONNX」",
        ]
        step = max(0, min(self._guide_step, len(steps) - 1))
        self.guide_label.setText(f"快速上手·行模型（{step + 1}/3）：{steps[step]}")

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
