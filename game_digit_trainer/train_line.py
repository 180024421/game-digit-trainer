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
        real_samples: list[tuple[np.ndarray, list[int]]] | None = None,
        max_w: int = LINE_MAX_WIDTH,
        include_bootstrap: bool = False,
    ) -> None:
        self.project = project
        self.classes = list(project.config.classes)
        self.pools = load_char_pools(project, include_bootstrap=include_bootstrap)
        self.max_w = int(max_w)
        real = list(real_samples) if real_samples is not None else load_real_line_samples(project)
        if not real and not self.pools:
            raise ValueError(
                "没有可用训练数据：请先在 ②「行待审」标注一些整行，"
                "或准备单字样本用于合成行图。"
            )
        # 无单字库时不能合成
        syn_n = max(0, int(synthetic)) if self.pools else 0
        if not real and syn_n <= 0:
            raise ValueError("单字数据集为空且无真实行样本，无法训练行模型")
        base_repeat = max(1, int(real_repeat))
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
        arr, valid_w = prepare_line_tensor(img, max_w=self.max_w)
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


def suggest_synthetic_count(project: GameProject, *, include_bootstrap: bool = False) -> int:
    """有单字库时合成足够多行图；无单字则返回 0（只训真实行）。"""
    pools = load_char_pools(project, include_bootstrap=include_bootstrap)
    if not pools:
        return 0
    n = sum(project.class_counts().values())
    # CPU 友好：约 800～2500，覆盖未见过的数字组合即可
    return int(min(2500, max(800, n * 12)))


def _split_train_val(
    samples: list[tuple[np.ndarray, list[int]]],
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, list[int]]], list[tuple[np.ndarray, list[int]]]]:
    if len(samples) < 5:
        return samples, []
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, min(len(samples) // 5, int(round(len(samples) * val_ratio))))
    val_i = set(idx[:n_val])
    train = [samples[i] for i in idx if i not in val_i]
    val = [samples[i] for i in idx if i in val_i]
    return train, val


def _eval_exact_match(
    model: DigitCRNN,
    classes: list[str],
    samples: list[tuple[np.ndarray, list[int]]],
    *,
    device: torch.device,
    max_w: int = LINE_MAX_WIDTH,
) -> float:
    """验证集整串完全匹配率。"""
    from game_digit_trainer.labels import display_label
    from game_digit_trainer.line_data import ctc_greedy_decode

    if not samples:
        return 0.0
    model.eval()
    ok = 0
    blank = model.blank_index
    with torch.no_grad():
        for img, indices in samples:
            arr, valid_w = prepare_line_tensor(img, max_w=max_w)
            x = torch.from_numpy(arr).unsqueeze(0).to(device)
            logits = model(x)
            t_use = int(max(2, min(logits.size(0), valid_w // 4)))
            log_prob = torch.log_softmax(logits[:t_use, 0], dim=-1).cpu().numpy()
            idxs = ctc_greedy_decode(log_prob, blank)
            text = "".join(display_label(classes[i]) for i in idxs if 0 <= i < len(classes))
            gold = "".join(display_label(classes[i]) for i in indices)
            if text == gold:
                ok += 1
    model.train()
    return ok / len(samples)


def _suggest_real_repeat(real_n: int, *, has_pools: bool) -> int:
    """控制每轮有效样本量，避免「样本一多 ×80」把 CPU 拖到数分钟/轮。"""
    if real_n <= 0:
        return 1
    if has_pools:
        # 另有合成行，真实样本适度加权即可
        return max(5, min(30, (1000 + real_n - 1) // real_n))
    # 仅真实行：每轮目标约 1200～1800 次前向（含增强），够学又不太慢
    target = 1500
    return max(8, min(40, (target + real_n - 1) // real_n))


def train_line_project(
    project: GameProject,
    *,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    synthetic: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    log=None,
    should_stop=None,
    augment_real: bool = True,
    finetune: bool = True,
    auto_bootstrap: bool = False,
) -> Path:
    import time

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    from game_digit_trainer.line_data import (
        bootstrap_chars_from_line_samples,
        suggest_line_max_width,
    )

    real_all = load_real_line_samples(project)
    real_n = len(real_all)
    use_boot = bool(auto_bootstrap)
    # 默认不含 from_line_*；仅勾选「自动粗切合成」时才用粗切图做合成
    has_hand_pools = bool(load_char_pools(project, include_bootstrap=False))

    if use_boot and real_n > 0 and not has_hand_pools:
        added, _previews = bootstrap_chars_from_line_samples(project)
        n_added = sum(added.values())
        if n_added:
            _log(
                f"已从行样本粗切单字入库 {n_added} 张（{', '.join(f'{k}:{v}' for k, v in sorted(added.items())[:8])}…），"
                "将混入合成行以覆盖更多数值组合（可在 ③ 点「行→单字粗切」预览效果）"
            )
    elif not use_boot and not has_hand_pools:
        _log("已关闭「自动粗切合成」，仅用真实行样本训练（忽略 from_line_* 粗切图）")

    has_pools = bool(load_char_pools(project, include_bootstrap=use_boot))
    char_n = sum(len(v) for v in load_char_pools(project, include_bootstrap=use_boot).values())

    if synthetic is None:
        syn = suggest_synthetic_count(project, include_bootstrap=use_boot)
    else:
        syn = int(synthetic) if has_pools else 0

    if real_n == 0 and not has_pools:
        raise ValueError(
            "没有行样本也没有单字样本。请先框选整行加入「行待审」并确认金标，再训练行模型。"
        )

    train_real, val_real = _split_train_val(real_all)
    real_repeat = _suggest_real_repeat(len(train_real) or real_n, has_pools=has_pools)
    on_cpu = (device == "cpu") if device else (not torch.cuda.is_available())
    if has_pools and syn > 0:
        real_repeat = max(5, min(real_repeat, 15))
        if on_cpu:
            syn = min(syn, 1200)

    max_w = suggest_line_max_width(train_real if train_real else real_all)
    _log(f"动态行宽 pad：max_w={max_w}（短数字少算空白，加快训练）")

    ds = LineDataset(
        project,
        synthetic=syn,
        real_repeat=real_repeat,
        augment_real=bool(augment_real),
        real_samples=train_real if train_real else real_all,
        max_w=max_w,
        include_bootstrap=use_boot,
    )
    mode = "仅真实行样本" if not has_pools else ("真实+合成" if real_n else "仅合成（单字拼行）")
    _log(
        f"行训练模式：{mode} · 训练真实 {len(train_real) or real_n}（×{real_repeat} → {len(ds.real)}）"
        f" + 合成 {ds.synthetic_n} = {len(ds)} · 验证 {len(val_real)}（单字库 {char_n}）"
    )
    if real_n and real_n < 5:
        _log(f"提示：真实行仅 {real_n} 条，建议多标几条 HUD 再训，准确率会更好。")

    bs = max(1, min(int(batch_size), len(ds)))
    workers = 0 if num_workers is None else max(0, int(num_workers))
    if num_workers is None:
        # Windows 默认 0 更稳；有 CUDA 时可试 2
        workers = 2 if (device == "cuda" or (device is None and torch.cuda.is_available())) else 0
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=_collate,
        num_workers=workers,
        persistent_workers=bool(workers > 0),
    )

    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            _log("请求 CUDA 但不可用，回退 CPU")
            dev = torch.device("cpu")
        else:
            dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = max(1, (len(ds) + bs - 1) // bs)
    _log(f"设备: {dev} · workers={workers} · 轮数: {epochs} · batch={bs} · 每轮约 {steps} step")
    if str(dev) == "cpu":
        _log("提示：当前为 CPU 训练。有 NVIDIA 显卡时可装 CUDA 版 PyTorch，通常能快数倍～十几倍。")

    classes = list(project.config.classes)
    model = DigitCRNN(num_classes=len(classes)).to(dev)
    use_lr = float(lr)
    resumed = False
    prev = latest_line_checkpoint(project) if finetune else None
    if prev and prev.is_file():
        try:
            ckpt = torch.load(prev, map_location=dev, weights_only=False)
            prev_classes = list(ckpt.get("classes") or [])
            if prev_classes == classes and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                use_lr = min(use_lr, 3e-4)  # 续训用更小学习率，更快适配新样本
                resumed = True
                _log(f"续训：加载 {prev.parent.name}/line_best.pt · lr={use_lr:g}")
            else:
                _log("类别与上次不一致，改为从头训练")
        except Exception as exc:
            _log(f"续训加载失败，改为从头训练：{exc}")

    opt = torch.optim.Adam(model.parameters(), lr=use_lr)
    ctc = torch.nn.CTCLoss(blank=model.blank_index, zero_infinity=True)

    run_dir = project.runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "line_best.pt"
    best_loss = float("inf")
    best_val_acc = -1.0
    history: list[dict] = []
    bad_epochs = 0  # 验证集准确率无提升轮数
    epoch_times: list[float] = []

    for epoch in range(1, epochs + 1):
        if should_stop and should_stop():
            _log("用户停止训练，保存当前最佳后退出")
            break
        t0 = time.perf_counter()
        model.train()
        total = 0.0
        n = 0
        for xb, y_cat, in_lens, y_lens in loader:
            if should_stop and should_stop():
                break
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
        val_acc = (
            _eval_exact_match(model, classes, val_real, device=dev, max_w=max_w)
            if val_real
            else -1.0
        )
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        remain = epochs - epoch
        eta = ""
        if epoch_times:
            avg_t = sum(epoch_times[-3:]) / min(3, len(epoch_times))
            eta = f" · 本轮 {dt:.0f}s · 预计剩余 ~{avg_t * remain / 60:.1f} 分钟"
        val_txt = f"  val_acc={val_acc:.0%}" if val_acc >= 0 else ""
        _log(f"line epoch {epoch}/{epochs}  ctc_loss={avg:.4f}{val_txt}{eta}")
        history.append(
            {
                "epoch": epoch,
                "loss": avg,
                "val_acc": val_acc if val_acc >= 0 else None,
                "seconds": round(dt, 2),
            }
        )

        improved = False
        if val_real:
            if val_acc > best_val_acc + 1e-6:
                best_val_acc = val_acc
                best_loss = avg
                bad_epochs = 0
                improved = True
            else:
                bad_epochs += 1
        elif avg < best_loss - 1e-4:
            best_loss = avg
            bad_epochs = 0
            improved = True
        else:
            bad_epochs += 1

        if improved:
            torch.save(
                {
                    "kind": "line_crnn",
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "input_height": 32,
                    "input_max_width": max_w,
                    "blank_index": model.blank_index,
                    "ctc_loss": best_loss,
                    "val_acc": best_val_acc if best_val_acc >= 0 else None,
                },
                best_path,
            )

        # 验证集准确率早停（优先）；无验证集时回退 loss 早停
        min_ep = 3 if resumed else 5
        if val_real:
            if best_val_acc >= 0.999 and epoch >= 2:
                _log(f"验证集已满分（val_acc={best_val_acc:.0%}），提前结束")
                break
            if epoch >= min_ep and bad_epochs >= 3 and best_val_acc >= 0.85:
                _log(f"验证集准确率已稳定（best val_acc={best_val_acc:.0%}），提前结束")
                break
            if epoch >= 10 and bad_epochs >= 5:
                _log(f"验证集连续无提升（best val_acc={best_val_acc:.0%}），提前结束")
                break
        else:
            if epoch >= min_ep and bad_epochs >= 2 and best_loss < 0.02:
                _log(f"已收敛（best={best_loss:.4f}），提前结束")
                break
            if epoch >= 8 and bad_epochs >= 4 and best_loss < 0.15:
                _log(f"已充分收敛（best={best_loss:.4f}），提前结束")
                break

    (run_dir / "line_metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_ctc_loss": best_loss,
                "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if best_val_acc >= 0:
        _log(f"最佳行模型: {best_path}  val_acc={best_val_acc:.0%}  ctc_loss={best_loss:.4f}")
    else:
        _log(f"最佳行模型: {best_path}  ctc_loss={best_loss:.4f}")
    if not best_path.is_file():
        torch.save(
            {
                "kind": "line_crnn",
                "model_state": model.state_dict(),
                "classes": classes,
                "input_height": 32,
                "input_max_width": max_w,
                "blank_index": model.blank_index,
                "ctc_loss": best_loss,
            },
            best_path,
        )
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
