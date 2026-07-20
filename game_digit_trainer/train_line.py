"""行模型 CRNN + CTC 训练。

数据来源（可只其一）：
- 真实行样本 lines/（推荐；无单字库也可训）
- 由单字 dataset 合成的行图（无真实行时兜底）
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from game_digit_trainer.line_data import (
    LINE_MAX_WIDTH,
    load_char_pools,
    load_real_line_samples,
    prepare_line_tensor,
    synthesize_line,
)
from game_digit_trainer.model_crnn import DigitCRNN
from game_digit_trainer.project import GameProject


def _augment_line_gray(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """轻量增强，便于「仅真实行」时靠重复采样仍有变化。"""
    out = img.astype(np.float32)
    if rng.random() < 0.6:
        out *= rng.uniform(0.75, 1.25)
    if rng.random() < 0.4:
        out += rng.uniform(-12.0, 12.0)
    if rng.random() < 0.35 and out.shape[1] > 8:
        shift = rng.randint(-3, 3)
        out = np.roll(out, shift, axis=1)
    if rng.random() < 0.25:
        h, w = out.shape[:2]
        scale = rng.uniform(0.9, 1.1)
        nh = max(8, int(round(h * scale)))
        resized = cv2.resize(out, (w, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((h, w), dtype=np.float32)
        y0 = max(0, (h - nh) // 2)
        y1 = min(h, y0 + nh)
        canvas[y0:y1] = resized[: y1 - y0]
        out = canvas
    return np.clip(out, 0, 255).astype(np.uint8)


class LineDataset(Dataset):
    def __init__(
        self,
        project: GameProject,
        *,
        synthetic: int = 4000,
        real_repeat: int = 20,
        seed: int = 42,
        augment_real: bool = True,
    ) -> None:
        self.project = project
        self.classes = list(project.config.classes)
        self.pools = load_char_pools(project)
        real = load_real_line_samples(project)
        if not real and not self.pools:
            raise ValueError(
                "没有可用训练数据：请先在 ②「行待审」标注一些整行，"
                "或准备单字样本用于合成行图。"
            )
        # 无单字库时不能合成
        syn_n = max(0, int(synthetic)) if self.pools else 0
        if not real and syn_n <= 0:
            raise ValueError("单字数据集为空且无真实行样本，无法训练行模型")
        # 真实 HUD 行样本更重要：重复采样提高权重；仅真实时多重复
        base_repeat = max(1, int(real_repeat))
        if real and not self.pools:
            base_repeat = max(base_repeat, 80)
        self.real = real * base_repeat if real else []
        self.synthetic_n = syn_n
        self.augment_real = bool(augment_real)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.real) + self.synthetic_n

    def __getitem__(self, index: int):
        if index < len(self.real):
            img, indices = self.real[index]
            if self.augment_real:
                img = _augment_line_gray(img, self._rng)
        else:
            img, indices = synthesize_line(self.pools, self.classes)
        arr, valid_w = prepare_line_tensor(img, max_w=LINE_MAX_WIDTH)
        x = torch.from_numpy(arr)  # 1,H,W
        y = torch.tensor(indices, dtype=torch.long)
        return x, y, int(valid_w)


def _collate(batch):
    xs, ys, widths = zip(*batch)
    x = torch.stack(list(xs), dim=0)
    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    y_cat = torch.cat(list(ys), dim=0)
    in_lens = torch.tensor([max(2, w // 4) for w in widths], dtype=torch.long)
    return x, y_cat, in_lens, y_lens


def suggest_synthetic_count(project: GameProject) -> int:
    """有单字库时合成足够多行图；无单字则返回 0（只训真实行）。"""
    pools = load_char_pools(project)
    if not pools:
        return 0
    n = sum(project.class_counts().values())
    # 约每字 25～40 条合成行，上限 5000
    return int(min(5000, max(2000, n * 30)))


def train_line_project(
    project: GameProject,
    *,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    synthetic: int | None = None,
    device: str | None = None,
    log=None,
) -> Path:
    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    real_n = len(load_real_line_samples(project))
    char_n = sum(project.class_counts().values())
    has_pools = bool(load_char_pools(project))
    if synthetic is None:
        syn = suggest_synthetic_count(project)
    else:
        syn = int(synthetic) if has_pools else 0

    if real_n == 0 and not has_pools:
        raise ValueError(
            "没有行样本也没有单字样本。请先框选整行加入「行待审」并确认金标，再训练行模型。"
        )

    # 仅真实行：提高重复；有合成时真实仍加权
    real_repeat = 80 if real_n and not has_pools else (40 if real_n else 1)
    ds = LineDataset(
        project,
        synthetic=syn,
        real_repeat=real_repeat,
        augment_real=True,
    )
    mode = "仅真实行样本" if not has_pools else ("真实+合成" if real_n else "仅合成（单字拼行）")
    _log(
        f"行训练模式：{mode} · 真实原始 {real_n}（加权后 {len(ds.real)}）"
        f" + 合成 {ds.synthetic_n} = {len(ds)}（单字库 {char_n}）"
    )
    if real_n and real_n < 5:
        _log(f"提示：真实行仅 {real_n} 条，建议多标几条 HUD 再训，准确率会更好。")

    loader = DataLoader(
        ds,
        batch_size=min(batch_size, max(1, len(ds))),
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
    )
    # 仅少量真实行时缩短 epoch，避免无意义空转；用户仍可通过 GUI 调高
    if real_n and not has_pools and epochs > 40 and real_n < 10:
        _log(f"仅真实行且样本较少：保持用户设定轮数 {epochs}（可适当加标后再训）")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _log(f"设备: {dev} · 轮数: {epochs}")
    model = DigitCRNN(num_classes=len(project.config.classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctc = torch.nn.CTCLoss(blank=model.blank_index, zero_infinity=True)

    run_dir = project.runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "line_best.pt"
    best_loss = float("inf")
    history: list[dict] = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, y_cat, in_lens, y_lens in loader:
            xb = xb.to(dev)
            y_cat = y_cat.to(dev)
            in_lens = in_lens.to(dev)
            y_lens = y_lens.to(dev)
            opt.zero_grad()
            logits = model(xb)  # T,N,C
            t = logits.size(0)
            in_lens = torch.clamp(in_lens, max=t)
            log_probs = logits.log_softmax(2)
            loss = ctc(log_probs, y_cat, in_lens, y_lens)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        avg = total / max(n, 1)
        _log(f"line epoch {epoch}/{epochs}  ctc_loss={avg:.4f}")
        history.append({"epoch": epoch, "loss": avg})
        if avg < best_loss - 1e-4:
            best_loss = avg
            bad_epochs = 0
            torch.save(
                {
                    "kind": "line_crnn",
                    "model_state": model.state_dict(),
                    "classes": project.config.classes,
                    "input_height": 32,
                    "input_max_width": LINE_MAX_WIDTH,
                    "blank_index": model.blank_index,
                    "ctc_loss": best_loss,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        # 充分训练后再早停：连续多轮无提升
        if epoch >= 15 and bad_epochs >= 6 and best_loss < 0.15:
            _log(f"已充分收敛（best={best_loss:.4f}），提前结束")
            break

    (run_dir / "line_metrics.json").write_text(
        json.dumps({"history": history, "best_ctc_loss": best_loss}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"最佳行模型: {best_path}  ctc_loss={best_loss:.4f}")
    return best_path


def latest_line_checkpoint(project: GameProject) -> Path | None:
    if not project.runs_dir.is_dir():
        return None
    bests = sorted(
        project.runs_dir.glob("*/line_best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return bests[0] if bests else None
