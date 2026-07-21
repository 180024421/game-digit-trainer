"""行评估 / 导出门禁 / 覆盖建议 / 待审预填。"""
from __future__ import annotations

import numpy as np

from game_digit_trainer.line_data import (
    clear_line_pending_hint,
    confirm_line_pending,
    coverage_fill_suggestions,
    get_line_pending_hint,
    line_coverage_report,
    save_line_pending,
    save_line_sample,
    set_line_pending_hint,
)
from game_digit_trainer.line_eval import evaluate_line_samples, format_line_eval_report
from game_digit_trainer.project import create_project
from game_digit_trainer.quality import line_export_quality_report
from game_digit_trainer.train_line import train_line_project


def test_line_pending_hints(tmp_path):
    proj = create_project("hint_t", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((36, 120, 3), dtype=np.uint8)
    bgr[8:28, 10:100] = 200
    path = save_line_pending(proj, bgr, (10, 8, 90, 20), source_name="x", pred="12", conf=0.9)
    h = get_line_pending_hint(proj, path.name)
    assert h.get("pred") == "12"
    assert float(h.get("conf") or 0) >= 0.9
    confirm_line_pending(proj, path, "12")
    assert not get_line_pending_hint(proj, path.name)


def test_coverage_suggestions(tmp_path):
    proj = create_project("cov_t", tmp_path, with_units=True, with_symbols=True)
    tips = coverage_fill_suggestions(proj)
    assert any("整行" in t or "差" in t for t in tips)
    bgr = np.zeros((36, 120, 3), dtype=np.uint8)
    bgr[8:28, 10:100] = 200
    save_line_sample(proj, bgr, (10, 8, 90, 20), "12")
    cov = line_coverage_report(proj)
    assert int(cov["total"]) >= 1
    assert int(cov["plain"]) >= 1


def test_line_export_quality_gate(tmp_path):
    proj = create_project("gate_t", tmp_path, with_units=True, with_symbols=True)
    fake = proj.runs_dir / "x" / "line_best.pt"
    fake.parent.mkdir(parents=True)
    # empty file → load fails → checkpoint_acc may error or skip
    import torch

    torch.save({"val_acc": 0.5, "classes": proj.config.classes, "model_state": {}}, fake)
    errors, warnings = line_export_quality_report(proj, fake, min_lines=10, min_val_acc=0.7)
    assert any("行样本过少" in e for e in errors)
    assert any("准确率" in w for w in warnings)


def test_evaluate_line_samples_runs(tmp_path):
    proj = create_project("eval_t", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((36, 160, 3), dtype=np.uint8)
    bgr[8:28, 10:150] = 220
    save_line_sample(proj, bgr, (10, 8, 140, 20), "12万")
    save_line_sample(proj, bgr, (10, 8, 140, 20), "3.9亿")
    path = train_line_project(proj, epochs=1, batch_size=2, synthetic=0, log=lambda _m: None)
    report = evaluate_line_samples(proj, path)
    assert int(report["total"]) >= 2
    text = format_line_eval_report(report)
    assert "行样本评估" in text


def test_clear_hint_helper(tmp_path):
    proj = create_project("clear_h", tmp_path, with_units=True)
    set_line_pending_hint(proj, "a.png", pred="1", conf=0.5)
    clear_line_pending_hint(proj, "a.png")
    assert not get_line_pending_hint(proj, "a.png")
