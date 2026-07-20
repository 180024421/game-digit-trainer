"""② 审核标注页：待审/已标/难例/行待审。"""
from __future__ import annotations

from game_digit_trainer.gui.deps import *  # noqa: F403


class ReviewTabMixin:
    def _build_review(self) -> None:
        layout = QHBoxLayout(self.tab_review)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        gallery = QVBoxLayout()
        mode_row = QHBoxLayout()
        self.btn_rev_pending = QPushButton('待审')
        self.btn_rev_labeled = QPushButton('已标注（可改）')
        self.btn_rev_hard = QPushButton('难例')
        self.btn_rev_line = QPushButton('行待审')
        self.btn_rev_line.setToolTip('整行框选后的待标样本：填整串文字确认即可')
        self.btn_rev_pending.setObjectName('primaryBtn')
        self.btn_rev_pending.clicked.connect(lambda: self._set_review_mode('pending'))
        self.btn_rev_labeled.clicked.connect(lambda: self._set_review_mode('labeled'))
        self.btn_rev_hard.clicked.connect(lambda: self._set_review_mode('hard'))
        self.btn_rev_line.clicked.connect(lambda: self._set_review_mode('line'))
        mode_row.addWidget(self.btn_rev_pending)
        mode_row.addWidget(self.btn_rev_line)
        mode_row.addWidget(self.btn_rev_labeled)
        mode_row.addWidget(self.btn_rev_hard)
        gallery.addLayout(mode_row)

        gal_head = QHBoxLayout()
        self.gallery_title = QLabel('待审预览（点击选择）')
        btn_reload = QPushButton('刷新')
        btn_reload.clicked.connect(self._reload_review_lists)
        btn_clear_pending = QPushButton('清空列表')
        btn_clear_pending.setObjectName('dangerBtn')
        btn_clear_pending.setToolTip('清空当前列表：待审全部 / 难例全部（已标注请用右侧样本库删除）')
        btn_clear_pending.clicked.connect(self._clear_review_gallery)
        gal_head.addWidget(self.gallery_title)
        gal_head.addWidget(btn_reload)
        gal_head.addWidget(btn_clear_pending)
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

        self.line_label_edit = QLineEdit()
        self.line_label_edit.setPlaceholderText("行待审：填写整串，例如 3920万 或 1.9亿")
        self.line_label_edit.setVisible(False)
        self.line_label_edit.returnPressed.connect(self._confirm_line_pending)
        right.addWidget(self.line_label_edit)
        self.btn_line_confirm = QPushButton("确认行标注 Enter")
        self.btn_line_confirm.setObjectName("successBtn")
        self.btn_line_confirm.setVisible(False)
        self.btn_line_confirm.clicked.connect(self._confirm_line_pending)
        right.addWidget(self.btn_line_confirm)

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
        for text, lab in [
            ("万 W", "万"),
            ("亿 Y", "亿"),
            (".", "."),
            (",", ","),
            ("/", "/"),
            ("%", "%"),
            (":", ":"),
        ]:
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
        sc_dot = QShortcut(QKeySequence(Qt.Key.Key_Period), self)
        sc_dot.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_dot.activated.connect(lambda: self._review_only(lambda: self._assign(".")))
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

    def _goto_review(self) -> None:
        self.tabs.setCurrentIndex(self.TAB_REVIEW)
        if self.project and list_line_pending(self.project):
            self._set_review_mode("line")
        elif (
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

    def _set_review_mode(self, mode: str) -> None:
        self._review_mode = mode if mode in ("pending", "labeled", "hard", "line") else "pending"
        self.btn_rev_pending.setObjectName("primaryBtn" if self._review_mode == "pending" else "")
        self.btn_rev_labeled.setObjectName("primaryBtn" if self._review_mode == "labeled" else "")
        if hasattr(self, "btn_rev_hard"):
            self.btn_rev_hard.setObjectName("primaryBtn" if self._review_mode == "hard" else "")
        if hasattr(self, "btn_rev_line"):
            self.btn_rev_line.setObjectName("primaryBtn" if self._review_mode == "line" else "")
        titles = {
            "pending": "待审预览（点击选择）",
            "labeled": "已标注（点选后可改标/退回）",
            "hard": "难例（低置信/改标/金标失败）",
            "line": "行待审（填整串文字后确认）",
        }
        self.gallery_title.setText(titles.get(self._review_mode, "预览"))
        for b in (
            self.btn_rev_pending,
            self.btn_rev_labeled,
            getattr(self, "btn_rev_hard", None),
            getattr(self, "btn_rev_line", None),
        ):
            if b is None:
                continue
            b.style().unpolish(b)
            b.style().polish(b)
        self._sync_line_review_ui()
        self._reload_review_lists()

    def _sync_line_review_ui(self) -> None:
        is_line = self._review_mode == "line"
        if hasattr(self, "line_label_edit"):
            self.line_label_edit.setVisible(is_line)
            self.btn_line_confirm.setVisible(is_line)
        if hasattr(self, "btn_confirm"):
            self.btn_confirm.setVisible(not is_line)
        # 行待审时隐藏逐字按钮区的使用引导，仍可保留按钮但不强制
        for b in getattr(self, "_digit_btns", {}).values():
            b.setEnabled(not is_line)
        for b in getattr(self, "_unit_btns", {}).values():
            b.setEnabled(not is_line)

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
        self._line_pending = list_line_pending(self.project)
        self._rebuild_gallery()
        self._show_current()
        self._update_header()
        self._update_review_progress()

    def _reload_pending(self) -> None:
        self._reload_review_lists()

    def _clear_review_gallery(self) -> None:
        """清空审核页当前列表：待审或难例。"""
        proj = self._require_project()
        if not proj:
            return
        mode = self._review_mode
        if mode == "labeled":
            QMessageBox.information(
                self,
                "提示",
                "已标注样本请到 ③ 训练页右侧「样本库」按类删除，或单张点「删除」。\n"
                "避免误清空整库。",
            )
            return
        if mode == "pending":
            files = list(proj.pending_files())
            if not files:
                QMessageBox.information(self, "提示", "待审列表已空")
                return
            reply = QMessageBox.question(
                self,
                "清空待审",
                f"将删除全部 {len(files)} 个待审单字，不可恢复。\n已标注的 dataset 不受影响。\n\n确定清空？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for p in files:
                p.unlink(missing_ok=True)
            self._pending_scores.clear()
            self._idx = 0
            self._status.showMessage(f"已清空待审 {len(files)} 个")
        elif mode == "hard":
            files = list_hard_files(proj)
            if not files:
                QMessageBox.information(self, "提示", "难例列表已空")
                return
            reply = QMessageBox.question(
                self,
                "清空难例",
                f"将删除全部 {len(files)} 个难例文件，不可恢复。确定？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for p in files:
                remove_hard_file(proj, p)
            self._hard_idx = 0
            self._status.showMessage(f"已清空难例 {len(files)} 个")
        elif mode == "line":
            files = list_line_pending(proj)
            if not files:
                QMessageBox.information(self, "提示", "行待审已空")
                return
            reply = QMessageBox.question(
                self,
                "清空行待审",
                f"将删除全部 {len(files)} 个行待审图，不可恢复。确定？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for p in files:
                p.unlink(missing_ok=True)
            self._line_idx = 0
            self._status.showMessage(f"已清空行待审 {len(files)} 个")
        self._reload_review_lists()
        self._refresh_project_info()

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
        elif self._review_mode == "line":
            items = [(p, "行") for p in self._line_pending]
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
        elif self._review_mode == "line":
            if row >= len(self._line_pending):
                return
            self._line_idx = row
            if hasattr(self, "line_label_edit"):
                self.line_label_edit.clear()
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
        if self._review_mode == "line":
            if not self._line_pending:
                return None
            self._line_idx = max(0, min(self._line_idx, len(self._line_pending) - 1))
            return self._line_pending[self._line_idx]
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
            elif self._review_mode == "line":
                self.char_view.setText("没有行待审\n① 整行蓝框 →「加入行待审」")
                self.review_meta.setText("无行待审")
                if hasattr(self, "line_label_edit"):
                    self.line_label_edit.clear()
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
            elif self._review_mode == "line":
                row = self._line_idx
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
        # 行图横向展示更宽
        max_side = 520 if self._review_mode == "line" else 340
        pix = numpy_to_pixmap(img)
        self.char_view.setPixmap(
            pix.scaled(
                max_side,
                340,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

        if self._review_mode == "pending":
            self.review_meta.setText(f"待审 {self._idx + 1} / {len(self._pending)}  ·  {path.name}")
            self._update_prediction(path)
        elif self._review_mode == "hard":
            self.review_meta.setText(f"难例 {self._hard_idx + 1} / {len(self._hard)}  ·  {path.name}")
            self._update_prediction(path)
            self.btn_confirm.setText("难例：点数字标注后移入数据集")
        elif self._review_mode == "line":
            self.review_meta.setText(
                f"行待审 {self._line_idx + 1} / {len(self._line_pending)}  ·  {path.name}"
            )
            self.pred_label_ui.setText("请填写整串文字后点「确认行标注」")
            self.pred_label_ui.setStyleSheet("font-size:18px; font-weight:700; color:#059669;")
            self.btn_confirm.setEnabled(False)
            if hasattr(self, "line_label_edit"):
                self.line_label_edit.setFocus()
                # 可选：用行模型预填
                if self.project:
                    line_ckpt = latest_line_checkpoint(self.project)
                    if line_ckpt and not self.line_label_edit.text().strip():
                        try:
                            from game_digit_trainer.predict_line import predict_line_gray
                            from game_digit_trainer.predict_line import load_line_checkpoint

                            model, classes, _h, max_w = load_line_checkpoint(line_ckpt)
                            text, _parts, _c = predict_line_gray(model, classes, img, max_w=max_w)
                            if text:
                                self.line_label_edit.setText(text)
                                self.line_label_edit.selectAll()
                        except Exception:
                            pass
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
        if self._review_mode != "line":
            self._update_context_view(path)
        else:
            self.context_view.setText("行待审：看大图填整串（如 3920万），不必逐字点")
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
        if self._review_mode in ("labeled", "hard", "line"):
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
        elif self._review_mode == "line":
            path.unlink(missing_ok=True)
            self._line_pending = [p for p in self._line_pending if p != path]
            if self._line_idx >= len(self._line_pending) and self._line_pending:
                self._line_idx = len(self._line_pending) - 1
            if hasattr(self, "line_label_edit"):
                self.line_label_edit.clear()
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
        elif self._review_mode == "line":
            if self._line_pending:
                self._line_idx = min(self._line_idx + 1, len(self._line_pending) - 1)
                if hasattr(self, "line_label_edit"):
                    self.line_label_edit.clear()
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
        elif self._review_mode == "line":
            if self._line_pending:
                self._line_idx = max(self._line_idx - 1, 0)
                if hasattr(self, "line_label_edit"):
                    self.line_label_edit.clear()
                self._show_current()
        elif self._labeled:
            self._labeled_idx = max(self._labeled_idx - 1, 0)
            self._show_current()

    # ---------- dataset ----------
    def _persist_confirm_threshold(self, value: float) -> None:
        if not self.project:
            return
        self.project.config.confirm_threshold = float(value)
        self.project.save_config()
        update_prefs(confirm_threshold=float(value))

    def _confirm_line_pending(self) -> None:
        proj = self._require_project()
        if not proj or self._review_mode != "line":
            return
        path = self._current_review_path()
        if not path:
            return
        text = (self.line_label_edit.text() or "").strip()
        if not text:
            QMessageBox.information(self, "提示", "请填写整串文字，例如 3920万")
            return
        try:
            dest = confirm_line_pending(proj, path, text)
        except Exception as exc:
            QMessageBox.warning(self, "标注失败", str(exc))
            return
        self.line_label_edit.clear()
        self._line_pending = list_line_pending(proj)
        if self._line_idx >= len(self._line_pending) and self._line_pending:
            self._line_idx = len(self._line_pending) - 1
        self._rebuild_gallery()
        self._show_current(sync_list=True)
        self._update_header()
        self._refresh_project_info()
        self._status.showMessage(f"行已标注：{text} → {dest.name}")
        # 方便立刻去样本库核对
        if self.tabs.currentIndex() == self.TAB_TRAIN:
            self._reload_dataset_browser()

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
            if self._review_mode == "line":
                self._confirm_line_pending()
            else:
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

