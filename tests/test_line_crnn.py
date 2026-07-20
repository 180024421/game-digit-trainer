"""行模型 CRNN / 合成行 / CTC 解码 / 仅真实行训练。"""
from __future__ import annotations

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


def test_synthesize_line_runs():
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
