"""③ 训练导出页：训练、样本库、导出、回归。"""
from __future__ import annotations

from game_digit_trainer.gui.deps import *  # noqa: F403


class TrainTabMixin:
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
        btn_train = QPushButton("训练单字（兼容）")
        btn_train.setToolTip("按 dataset 单字目录训练 CNN（切字审核 / Studio 旧包）")
        btn_train.clicked.connect(self._start_train)
        form.addWidget(btn_train)
        btn_train_line = QPushButton("训练行模型（推荐）")
        btn_train_line.setObjectName("successBtn")
        btn_train_line.setToolTip(
            "主路径：用行样本训练 CRNN；无单字库也可。"
            "识别时蓝框一次出整串。"
        )
        btn_train_line.clicked.connect(self._start_train_line)
        form.addWidget(btn_train_line)
        self.btn_stop_train = QPushButton("停止训练")
        self.btn_stop_train.setObjectName("dangerBtn")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.setToolTip("尽快结束当前训练并保留已写出的最佳权重")
        self.btn_stop_train.clicked.connect(self._stop_train)
        form.addWidget(self.btn_stop_train)
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
        btn_export = QPushButton("导出单字 ONNX")
        btn_export.clicked.connect(self._export)
        btn_export_line = QPushButton("导出行 ONNX")
        btn_export_line.setObjectName("successBtn")
        btn_export_line.setToolTip("导出 exports/line/digits_line.onnx，可供 Studio recognizeDigits")
        btn_export_line.clicked.connect(self._export_line)
        btn_open_export = QPushButton("打开导出目录")
        btn_open_export.clicked.connect(self._open_export_dir)
        btn_copy_export = QPushButton("复制导出路径")
        btn_copy_export.clicked.connect(self._copy_export_path)
        btn_studio = QPushButton("拷单字到 Studio")
        btn_studio.setToolTip("拷贝 digits.onnx 到脚本工程 models/")
        btn_studio.clicked.connect(self._copy_to_studio)
        btn_studio_line = QPushButton("拷行模型到 Studio")
        btn_studio_line.setToolTip("拷贝 exports/line/ 到脚本工程 models/line/")
        btn_studio_line.clicked.connect(self._copy_line_to_studio)
        self.export_dep_label = QLabel("")
        self.export_dep_label.setObjectName("hintLabel")
        exp.addWidget(btn_export_line)
        exp.addWidget(btn_export)
        exp.addWidget(btn_open_export)
        exp.addWidget(btn_copy_export)
        exp.addWidget(btn_studio_line)
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
        btn_reg_run.setToolTip("优先用最新行模型；无行模型时用单字 checkpoint")
        btn_reg_run.clicked.connect(self._run_regression)
        bal_row.addWidget(btn_boost)
        bal_row.addWidget(btn_scarce)
        bal_row.addWidget(btn_reg_run)
        btn_conf = QPushButton("混淆矩阵")
        btn_conf.clicked.connect(self._show_confusion)
        btn_backup = QPushButton("备份项目")
        btn_backup.setToolTip("打包 dataset/lines/exports 等整包到 backups/")
        btn_backup.clicked.connect(self._backup_project)
        btn_export_labels = QPushButton("导出标注包")
        btn_export_labels.setObjectName("successBtn")
        btn_export_labels.setToolTip(
            "导出单字库 + 行样本/行待审 + 难例（不含模型），便于换机或分享标注"
        )
        btn_export_labels.clicked.connect(self._export_labels_pack)
        btn_import_labels = QPushButton("导入标注包")
        btn_import_labels.setToolTip("从 zip 合并或替换导入标注数据")
        btn_import_labels.clicked.connect(self._import_labels_pack)
        btn_cmp = QPushButton("多项目对比")
        btn_cmp.clicked.connect(self._compare_projects)
        bal_row.addWidget(btn_conf)
        bal_row.addWidget(btn_backup)
        bal_row.addWidget(btn_export_labels)
        bal_row.addWidget(btn_import_labels)
        bal_row.addWidget(btn_cmp)
        bal_row.addStretch()
        left.addLayout(bal_row)
        layout.addLayout(left, 2)

        # dataset browser embedded（含【行样本】）
        right = QGroupBox("样本库（改错/删除）")
        rl = QVBoxLayout(right)
        hint = QLabel(
            "行模型主用「行样本」。左侧选【行样本】可改金标/删除；"
            "没有单字也能训行模型。单字类仅用于可选合成增强，或旧单字模型。"
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        rl.addWidget(hint)
        row = QHBoxLayout()
        self.ds_class_list = QListWidget()
        self.ds_class_list.setMaximumWidth(140)
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

        line_edit_row = QHBoxLayout()
        self.ds_line_edit = QLineEdit()
        self.ds_line_edit.setPlaceholderText("行样本金标，例如 3920万")
        self.ds_line_edit.setVisible(False)
        self.ds_line_edit.returnPressed.connect(self._ds_update_line_label)
        btn_line_relabel = QPushButton("改金标")
        btn_line_relabel.setVisible(False)
        btn_line_relabel.clicked.connect(self._ds_update_line_label)
        self.btn_ds_line_relabel = btn_line_relabel
        line_edit_row.addWidget(self.ds_line_edit, 1)
        line_edit_row.addWidget(btn_line_relabel)
        rl.addLayout(line_edit_row)

        move_row = QHBoxLayout()
        self.ds_move_combo = QComboBox()
        self.btn_ds_move = QPushButton("改到此类")
        btn_del = QPushButton("删除")
        btn_del.setObjectName("dangerBtn")
        btn_clear_cls = QPushButton("清空本类")
        btn_clear_cls.setObjectName("dangerBtn")
        btn_clear_cls.setToolTip("单字：清空当前类；行样本：清空全部已标行图")
        self.btn_ds_move.clicked.connect(self._ds_move)
        btn_del.clicked.connect(self._ds_delete)
        btn_clear_cls.clicked.connect(self._ds_clear_class)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._reload_dataset_browser)
        move_row.addWidget(self.ds_move_combo, 1)
        move_row.addWidget(self.btn_ds_move)
        move_row.addWidget(btn_del)
        move_row.addWidget(btn_clear_cls)
        move_row.addWidget(btn_refresh)
        rl.addLayout(move_row)
        layout.addWidget(right, 2)

    def _refresh_project_info(self) -> None:
        if not self.project:
            self.project_info.setPlainText("")
            return
        counts = self.project.class_counts()
        total = sum(counts.values())
        line_n = count_line_labeled(self.project)
        line_pending_n = len(list_line_pending(self.project))
        lines = [
            f"路径: {self.project.root}",
            f"类别: {' '.join(display_label(c) for c in self.project.config.classes)}",
            f"单字已标: {total} · 单字待审: {len(self.project.pending_files())}",
            f"行样本已标: {line_n} · 行待审: {line_pending_n}",
            "",
        ]
        for k, v in counts.items():
            if v:
                lines.append(f"  {display_label(k)}: {v}")
        if line_n:
            lines.append(f"  【行样本】: {line_n}（右侧样本库可改/删）")
        ckpt = latest_checkpoint(self.project)
        line_ckpt = latest_line_checkpoint(self.project)
        lines.append(f"单字模型: {ckpt.parent.name if ckpt else '尚未训练'}")
        lines.append(f"行模型: {line_ckpt.parent.name if line_ckpt else '尚未训练（③点「训练行模型」）'}")
        if line_n < 30:
            lines.append("")
            lines.append(f"建议：行样本现 {line_n} 条，标到约 30～50 条再训通常更稳。")
        elif line_n < 80:
            lines.append("")
            lines.append(f"行样本 {line_n} 条：同 HUD 一般够用；换皮肤可继续补到 80+。")
        lines.append("")
        lines.append(format_balance_text(counts))
        self.project_info.setPlainText("\n".join(lines))
        self._update_header()
        self._reload_roi_preset_combo()
        self._update_export_dep_hint()

    def _ds_is_line_mode(self) -> bool:
        item = self.ds_class_list.currentItem()
        if not item:
            return False
        return item.data(Qt.ItemDataRole.UserRole) == LINE_DATASET_KEY

    def _set_ds_line_controls(self, line_mode: bool) -> None:
        self.ds_line_edit.setVisible(line_mode)
        self.btn_ds_line_relabel.setVisible(line_mode)
        self.ds_move_combo.setVisible(not line_mode)
        self.btn_ds_move.setVisible(not line_mode)
        if not line_mode:
            self.ds_line_edit.clear()

    def _reload_dataset_browser(self) -> None:
        prev_key = None
        cur = self.ds_class_list.currentItem()
        if cur:
            prev_key = cur.data(Qt.ItemDataRole.UserRole)
        self.ds_class_list.clear()
        self.ds_file_list.clear()
        self.ds_move_combo.clear()
        self.ds_preview.setText("选样本")
        self._set_ds_line_controls(False)
        if not self.project:
            return
        line_n = count_line_labeled(self.project)
        line_item = QListWidgetItem(f"【行样本】 ({line_n})")
        line_item.setData(Qt.ItemDataRole.UserRole, LINE_DATASET_KEY)
        line_item.setToolTip("审核「行待审」确认后的整行图 + 金标，供训练行模型")
        self.ds_class_list.addItem(line_item)
        counts = self.project.class_counts()
        for name in self.project.config.classes:
            item = QListWidgetItem(f"{display_label(name)} ({counts.get(name, 0)})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.ds_class_list.addItem(item)
            self.ds_move_combo.addItem(display_label(name), name)
        # 尽量保持上次选中；有行样本时默认点开行样本便于管理
        target_row = 0
        if prev_key is not None:
            for i in range(self.ds_class_list.count()):
                if self.ds_class_list.item(i).data(Qt.ItemDataRole.UserRole) == prev_key:
                    target_row = i
                    break
        elif line_n == 0 and self.ds_class_list.count() > 1:
            target_row = 1
        self.ds_class_list.setCurrentRow(target_row)

    def _on_ds_class(self, _text: str) -> None:
        self.ds_file_list.clear()
        self.ds_preview.setText("选样本")
        if not self.project:
            return
        item = self.ds_class_list.currentItem()
        if not item:
            return
        label = item.data(Qt.ItemDataRole.UserRole)
        if label == LINE_DATASET_KEY:
            self._set_ds_line_controls(True)
            for path, text in list_line_labeled(self.project):
                title = f"{text or '（无金标）'}  ·  {path.name}"
                it = QListWidgetItem(title)
                it.setData(Qt.ItemDataRole.UserRole, str(path))
                it.setData(Qt.ItemDataRole.UserRole + 1, text)
                self.ds_file_list.addItem(it)
            return
        self._set_ds_line_controls(False)
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
        max_w, max_h = (280, 120) if self._ds_is_line_mode() else (120, 120)
        self.ds_preview.setPixmap(
            numpy_to_pixmap(img).scaled(
                max_w,
                max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )
        if self._ds_is_line_mode():
            text = item.data(Qt.ItemDataRole.UserRole + 1) or ""
            self.ds_line_edit.setText(str(text))
            self.ds_line_edit.setFocus()

    def _current_ds_path(self) -> Path | None:
        item = self.ds_file_list.currentItem()
        if not item:
            return None
        return Path(item.data(Qt.ItemDataRole.UserRole))

    def _ds_update_line_label(self) -> None:
        proj = self._require_project()
        path = self._current_ds_path()
        if not proj or not path or not self._ds_is_line_mode():
            return
        text = (self.ds_line_edit.text() or "").strip()
        if not text:
            QMessageBox.information(self, "提示", "请填写金标，例如 3920万")
            return
        try:
            display = update_line_label(proj, path, text)
        except Exception as exc:
            QMessageBox.warning(self, "改金标失败", str(exc))
            return
        row = self.ds_file_list.currentRow()
        self._on_ds_class("")
        if 0 <= row < self.ds_file_list.count():
            self.ds_file_list.setCurrentRow(row)
        self._refresh_project_info()
        self._status.showMessage(f"行金标已改为：{display}")

    def _ds_move(self) -> None:
        if self._ds_is_line_mode():
            self._ds_update_line_label()
            return
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
        proj = self._require_project()
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
        if self._ds_is_line_mode() and proj:
            delete_line_sample(proj, path)
        else:
            path.unlink(missing_ok=True)
        self._on_ds_class("")
        self._refresh_project_info()
        # 刷新左侧计数
        self._reload_dataset_browser()

    def _ds_clear_class(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        item = self.ds_class_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在左侧选中一个类别")
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if name == LINE_DATASET_KEY:
            n = count_line_labeled(proj)
            if n <= 0:
                QMessageBox.information(self, "提示", "没有已标行样本")
                return
            reply = QMessageBox.question(
                self,
                "清空行样本",
                f"将删除全部 {n} 个已标行样本及金标，不可恢复。确定？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            deleted = clear_line_samples(proj)
            self._reload_dataset_browser()
            self._refresh_project_info()
            self._status.showMessage(f"已清空行样本 {deleted} 个")
            return
        try:
            name = normalize_label(str(name))
        except Exception:
            name = str(name)
        files = list_dataset_files(proj, name)
        if not files:
            QMessageBox.information(self, "提示", f"类别「{display_label(name)}」已无样本")
            return
        reply = QMessageBox.question(
            self,
            "清空本类",
            f"将删除类别「{display_label(name)}」下全部 {len(files)} 个样本，不可恢复。确定？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for p in files:
            p.unlink(missing_ok=True)
        self._reload_dataset_browser()
        self._refresh_project_info()
        self._status.showMessage(f"已清空类别 {display_label(name)}（{len(files)}）")

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
        self._worker.failed.connect(self._on_train_failed)
        if hasattr(self, "btn_stop_train"):
            self.btn_stop_train.setEnabled(True)
        self._worker.start()

    def _start_train_line(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "提示", "训练进行中")
            return
        from game_digit_trainer.line_data import count_line_labeled, load_char_pools

        line_n = count_line_labeled(proj)
        char_n = sum(proj.class_counts().values())
        has_pools = bool(load_char_pools(proj))
        if line_n <= 0 and not has_pools:
            QMessageBox.warning(
                self,
                "提示",
                "还没有行样本，也无法用单字合成。\n"
                "请先：① 整行蓝框 →「加入行待审」→ ② 填金标确认，再点「训练行模型」。",
            )
            return
        if line_n <= 0 and char_n < 10:
            QMessageBox.warning(
                self,
                "提示",
                f"没有行样本，且单字太少（{char_n}）。\n"
                "推荐直接标几条「行样本」再训；或先多标一些单字用于合成。",
            )
            return
        self.train_log.clear()
        if line_n and not has_pools:
            self.train_log.append(
                f"仅用真实行样本训练（{line_n} 条，会增强/重复采样）。"
                "无需单字库。样本越多越准；识别仍是蓝框一次前向。"
            )
        elif line_n and has_pools:
            self.train_log.append(
                f"真实行 {line_n} 条 + 单字合成行图。训练可能较久；识别很快。"
            )
        else:
            self.train_log.append(
                "当前无真实行样本，将用单字拼成行图训练（精度通常不如真实 HUD）。"
                "建议之后多标行样本再训。"
            )
        # 精度优先：至少 20 轮，尊重用户调高的轮数
        epochs = max(20, int(self.epochs_spin.value()))
        self._worker = TrainWorker(proj, epochs, augment=False, line_mode=True)
        self._worker.log.connect(lambda m: self.train_log.append(m))
        self._worker.done.connect(self._train_line_done)
        self._worker.failed.connect(self._on_train_failed)
        if hasattr(self, "btn_stop_train"):
            self.btn_stop_train.setEnabled(True)
        self._worker.start()

    def _stop_train(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._status.showMessage("正在停止训练…")
            if hasattr(self, "btn_stop_train"):
                self.btn_stop_train.setEnabled(False)

    def _on_train_failed(self, err: str) -> None:
        if hasattr(self, "btn_stop_train"):
            self.btn_stop_train.setEnabled(False)
        QMessageBox.critical(self, "训练失败", err)

    def _train_line_done(self, path: str) -> None:
        if hasattr(self, "btn_stop_train"):
            self.btn_stop_train.setEnabled(False)
        self.train_log.append(f"行模型完成: {path}")
        self._refresh_project_info()
        self._reload_verify_models()
        self._status.showMessage(f"行模型完成: {path} — 可用蓝框「识别」试读，或「导出行 ONNX」")
        line_n = count_line_labeled(self.project) if self.project else 0
        tip = ""
        if line_n < 30:
            tip = f"\n提示：当前行样本约 {line_n} 条，建议标到 30～50 条再训效果更好。"
        QMessageBox.information(
            self,
            "行模型完成",
            f"已保存：{path}\n\n切字页用「整行蓝框」后点「识别」即可出整串。{tip}",
        )

    def _train_done(self, path: str) -> None:
        if hasattr(self, "btn_stop_train"):
            self.btn_stop_train.setEnabled(False)
        self.train_log.append(f"完成: {path}")
        self._refresh_project_info()
        self._refresh_train_curve()
        self._status.showMessage(f"单字训练完成: {path}")
        QMessageBox.information(
            self,
            "完成",
            "单字训练完成。主路径仍推荐行模型；单字包用于 Studio 旧 recognizeDigits。",
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
        self._status.showMessage(f"已导出单字 ONNX：{out.parent}")

    def _export_line(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        ok, msg = check_onnx_dependency()
        self._update_export_dep_hint()
        if not ok:
            QMessageBox.warning(self, "缺少依赖", msg)
            return
        try:
            out = export_line_onnx(proj)
        except Exception as exc:
            QMessageBox.critical(self, "导出行模型失败", f"{exc}\n{traceback.format_exc()}")
            return
        self._reload_verify_models(prefer_key=f"onnx:{out.resolve()}")
        self._status.showMessage(f"已导出行 ONNX：{out}")
        QMessageBox.information(
            self,
            "已导出行模型",
            f"{out.parent}\n\n可用「拷行模型到 Studio」写入 models/line/\n"
            "Lua: bot.recognizeDigits({{ roi=..., model='models/line/digits_line' }})",
        )

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

    def _persist_augment(self, _state: int = 0) -> None:
        if not self.project:
            return
        self.project.config.augment = self.chk_augment.isChecked()
        self.project.save_config()

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
        self._status.showMessage(f"已拷单字包到 {models}")
        QMessageBox.information(
            self,
            "已拷到 Studio",
            f"目标：{models}\n文件：{', '.join(copied)}\n详见 docs/studio-recognize-digits.md",
        )

    def _copy_line_to_studio(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        line_dir = proj.exports_dir / "line"
        if not (line_dir / "digits_line.onnx").is_file():
            QMessageBox.information(self, "提示", "请先点「导出行 ONNX」")
            return
        path = QFileDialog.getExistingDirectory(self, "选择脚本工程目录（将写入 models/line/）")
        if not path:
            return
        models = Path(path) / "models"
        try:
            copied = copy_line_exports_to_studio(line_dir, models)
        except Exception as exc:
            QMessageBox.warning(self, "拷贝失败", str(exc))
            return
        self._status.showMessage(f"已拷行模型到 {models / 'line'}")
        QMessageBox.information(
            self,
            "已拷行模型到 Studio",
            f"目标：{models / 'line'}\n文件：{', '.join(copied)}\n"
            "Lua: model = \"models/line/digits_line\"",
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

    def _run_regression(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        cases = load_cases(proj)
        if not cases:
            QMessageBox.information(self, "提示", "回归集为空：在切字页填金标后点「加入回归集」")
            return
        line_ckpt = latest_line_checkpoint(proj)
        if line_ckpt:
            report = run_line_regression(proj, line_ckpt)
            mode = f"行模型 {line_ckpt.parent.name}"
        else:
            ckpt = latest_checkpoint(proj)
            if not ckpt:
                QMessageBox.information(self, "提示", "请先训练行模型或单字模型")
                return
            report = run_regression(proj, ckpt)
            mode = f"单字 {ckpt.parent.name}"
        lines = [f"[{mode}] 通过 {report['passed']}/{report['total']}"]
        for r in report.get("results") or []:
            if r.get("ok"):
                lines.append(f"✓ {r.get('name')}: {r.get('got')}")
            else:
                lines.append(
                    f"✗ {r.get('name')}: expect={r.get('expected')} got={r.get('got') or r.get('error')}"
                )
        self._status.showMessage(lines[0])
        QMessageBox.information(self, "回归结果", "\n".join(lines[:40]))

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

    def _export_labels_pack(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "导出标注包",
            str(proj.root / "backups" / f"{proj.config.game_id}_labels.zip"),
            "ZIP (*.zip)",
        )
        if not dest:
            return
        try:
            path = export_labels_pack(proj, dest=Path(dest))
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self._status.showMessage(f"已导出标注包: {path}")
        QMessageBox.information(
            self,
            "已导出标注包",
            f"{path}\n\n含：单字 dataset、行样本 lines、待审 pending、难例 hard、config.json",
        )

    def _import_labels_pack(self) -> None:
        proj = self._require_project()
        if not proj:
            return
        src, _ = QFileDialog.getOpenFileName(self, "导入标注包", "", "ZIP (*.zip)")
        if not src:
            return
        box = QMessageBox(self)
        box.setWindowTitle("导入标注包")
        box.setText(
            "如何导入？\n\n"
            "· 合并：保留现有样本，同名文件自动改名\n"
            "· 替换：先清空本项目的单字/行样本/待审/难例，再导入"
        )
        merge_btn = box.addButton("合并导入", QMessageBox.ButtonRole.AcceptRole)
        replace_btn = box.addButton("替换导入", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or clicked not in (merge_btn, replace_btn):
            return
        mode = "replace" if clicked is replace_btn else "merge"
        if mode == "replace":
            confirm = QMessageBox.question(
                self,
                "确认替换",
                "将清空本项目的 dataset / lines / pending / hard，再写入标注包内容。\n继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        try:
            result = import_labels_pack(proj, Path(src), mode=mode)
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        self._refresh_project_info()
        if hasattr(self, "_reload_dataset_browser"):
            self._reload_dataset_browser()
        self._status.showMessage(f"已导入标注包: {result.summary()}")
        QMessageBox.information(self, "导入完成", result.summary())

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

