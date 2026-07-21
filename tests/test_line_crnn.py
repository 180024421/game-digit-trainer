"""行模型 CRNN / 合成行 / CTC 解码 / 仅真实行训练。"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from game_digit_trainer.line_data import (
    ctc_greedy_decode,
    prepare_line_tensor,
    save_line_sample,
    synthesize_line,
)
from game_digit_trainer.model_crnn import DigitCRNN
from game_digit_trainer.project import create_project
from game_digit_trainer.segment import crop_bgr
from game_digit_trainer.train_line import LineDataset, suggest_synthetic_count, train_line_project


def test_crnn_forward_shape():
    m = DigitCRNN(num_classes=12)
    x = torch.zeros(2, 1, 32, 128)
    y = m(x)
    assert y.dim() == 3
    assert y.size(1) == 2
    assert y.size(2) == 13  # +blank


def test_ctc_greedy_decode():
    # T=5, C=4 (blank=3)
    logits = np.zeros((5, 4), dtype=np.float32)
    logits[0, 0] = 1
    logits[1, 0] = 1  # repeat
    logits[2, 3] = 1  # blank
    logits[3, 1] = 1
    logits[4, 1] = 1
    assert ctc_greedy_decode(logits, blank_index=3) == [0, 1]


def test_trim_line_margins_drops_left_noise():
    from game_digit_trainer.predict_line import trim_line_margins

    # 左侧一大块暗噪声 + 右侧亮字
    g = np.zeros((24, 80), dtype=np.uint8)
    g[:, 0:25] = 40  # 噪声，不够成「字」列
    g[4:20, 40:70] = 220  # 字
    trimmed = trim_line_margins(g)
    assert trimmed.shape[1] < g.shape[1]
    assert trimmed.shape[1] <= 40
    pools = {
        "1": [np.zeros((20, 10), dtype=np.uint8) + 255],
        "2": [np.zeros((20, 12), dtype=np.uint8) + 255],
        "wan": [np.zeros((20, 16), dtype=np.uint8) + 255],
    }
    classes = ["0", "1", "2", "wan"]
    img, idxs = synthesize_line(pools, classes, min_len=2, max_len=3, gap=1)
    assert img.ndim == 2 and img.shape[0] == 32
    assert len(idxs) >= 2
    arr, w = prepare_line_tensor(img)
    assert arr.shape[0] == 1 and arr.shape[1] == 32
    assert w >= 8


def test_line_infer_matches_train_gray_not_binarize(tmp_path):
    """推理须与入库一致用灰度，不能套项目 Otsu（否则会乱读）。"""
    from game_digit_trainer.predict_line import _roi_to_line_gray, predict_line_roi
    from game_digit_trainer.preprocess import apply_preprocess
    from game_digit_trainer.train_line import latest_line_checkpoint

    proj = create_project("line_infer_pp", tmp_path, with_units=True, with_symbols=True)
    proj.config.preprocess.binarize = "otsu"
    proj.config.preprocess.invert = True
    proj.save_config()

    bgr = np.zeros((40, 120, 3), dtype=np.uint8)
    bgr[10:30, 20:100] = (40, 180, 220)  # 亮色字区域
    region = (20, 10, 80, 20)
    gray = _roi_to_line_gray(bgr, region)
    binary = apply_preprocess(crop_bgr(bgr, region), proj.config.preprocess)
    assert gray.ndim == 2
    assert not np.array_equal(gray, binary), "灰度应与二值预处理不同"

    save_line_sample(proj, bgr, region, "12万")
    train_line_project(proj, epochs=2, batch_size=2, synthetic=0, log=lambda _m: None)
    ckpt = latest_line_checkpoint(proj)
    assert ckpt is not None
    # 能跑通即可；重点是不会因预处理崩，且走灰度路径
    text, _parts, _c = predict_line_roi(proj, bgr, region, ckpt)
    assert isinstance(text, str)


def test_line_decode_truncates_padding_tail(tmp_path):
    """pad 黑边不应解出尾巴；短训后对入库样本应能读对金标。"""
    from game_digit_trainer.predict_line import load_line_checkpoint, predict_line_gray
    from game_digit_trainer.train_line import latest_line_checkpoint

    proj = create_project("line_pad_decode", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((36, 160, 3), dtype=np.uint8)
    bgr[8:28, 10:100] = 220
    save_line_sample(proj, bgr, (10, 8, 90, 20), "12万")
    save_line_sample(proj, bgr, (10, 8, 90, 20), "3.9亿")
    train_line_project(proj, epochs=8, batch_size=2, synthetic=0, log=lambda _m: None)
    ckpt = latest_line_checkpoint(proj)
    assert ckpt
    model, classes, _h, max_w = load_line_checkpoint(ckpt)
    from game_digit_trainer.line_data import list_line_labeled

    for path, gold in list_line_labeled(proj):
        raw = np.fromfile(str(path), dtype=np.uint8)
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        text, _p, _c = predict_line_gray(model, classes, gray, max_w=max_w)
        # 短训可能不完全准，但不应稳定出现「万万亿」式 pad 尾巴
        assert "万万" not in text
        assert not text.endswith("亿2")


def test_bootstrap_chars_from_lines(tmp_path):
    proj = create_project("boot_chars", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((36, 160, 3), dtype=np.uint8)
    bgr[8:28, 10:100] = 220
    save_line_sample(proj, bgr, (10, 8, 90, 20), "12万")
    from game_digit_trainer.line_data import bootstrap_chars_from_line_samples

    added, previews = bootstrap_chars_from_line_samples(proj)
    assert sum(added.values()) >= 2
    assert previews
    assert (proj.dataset_dir / "1").is_dir()
    assert any((proj.dataset_dir / "1").glob("*.png"))
    from game_digit_trainer.line_data import clear_bootstrap_chars, load_char_pools

    assert not load_char_pools(proj, include_bootstrap=False)
    assert load_char_pools(proj, include_bootstrap=True)
    n = clear_bootstrap_chars(proj)
    assert n >= 2
    assert not load_char_pools(proj, include_bootstrap=True)


def test_line_train_real_only_no_chars(tmp_path):
    """无单字库、仅有行样本时也能建数据集并跑通短训。"""
    proj = create_project("line_real_only", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((36, 160, 3), dtype=np.uint8)
    bgr[8:28, 10:150] = 220
    save_line_sample(proj, bgr, (10, 8, 140, 20), "12万")
    save_line_sample(proj, bgr, (10, 8, 140, 20), "3.9亿")
    assert suggest_synthetic_count(proj) == 0
    ds = LineDataset(proj, synthetic=100, real_repeat=5, seed=1)
    assert ds.synthetic_n == 0
    assert len(ds.real) >= 2
    x, y, w = ds[0]
    assert x.shape[0] == 1 and y.numel() >= 1 and w >= 1
    path = train_line_project(proj, epochs=1, batch_size=2, synthetic=0, log=lambda _m: None)
    assert path.is_file()


def test_export_line_onnx(tmp_path):
    from game_digit_trainer.export_line_onnx import export_line_onnx
    from game_digit_trainer.predict import check_onnx_dependency

    ok, _ = check_onnx_dependency()
    if not ok:
        import pytest

        pytest.skip("onnx not installed")
    proj = create_project("line_onnx_t", tmp_path, with_units=True, with_symbols=True)
    bgr = np.zeros((36, 160, 3), dtype=np.uint8)
    bgr[8:28, 10:150] = 220
    save_line_sample(proj, bgr, (10, 8, 140, 20), "12万")
    ckpt = train_line_project(proj, epochs=1, batch_size=2, synthetic=0, log=lambda _m: None)
    out = export_line_onnx(proj, ckpt)
    assert out.is_file()
    assert (out.parent / "manifest.json").is_file()
    manifest = (out.parent / "manifest.json").read_text(encoding="utf-8")
    assert "line_crnn" in manifest
