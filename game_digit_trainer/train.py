from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from game_digit_trainer.model import DigitCNN
from game_digit_trainer.project import GameProject
from game_digit_trainer.segment import prepare_tensor_image


class CharFolderDataset(Dataset):
    def __init__(self, project: GameProject, *, augment: bool = False) -> None:
        self.project = project
        self.augment = augment
        self.classes = list(project.config.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.items: list[tuple[Path, int]] = []
        for name in self.classes:
            folder = project.dataset_dir / name
            if not folder.is_dir():
                continue
            for p in folder.iterdir():
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.items.append((p, self.class_to_idx[name]))
        if not self.items:
            raise ValueError("数据集为空，请先审核修正若干样本")

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        # slight brightness / contrast
        if np.random.rand() < 0.7:
            alpha = float(np.random.uniform(0.75, 1.25))
            beta = float(np.random.uniform(-25, 25))
            out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        # small shift
        if np.random.rand() < 0.6:
            h, w = out.shape[:2]
            tx = int(np.random.randint(-max(1, w // 8), max(2, w // 8 + 1)))
            ty = int(np.random.randint(-max(1, h // 8), max(2, h // 8 + 1)))
            m = np.float32([[1, 0, tx], [0, 1, ty]])
            out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # mild scale
        if np.random.rand() < 0.5:
            h, w = out.shape[:2]
            scale = float(np.random.uniform(0.85, 1.15))
            nh, nw = max(4, int(h * scale)), max(4, int(w * scale))
            scaled = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros_like(out)
            y0 = max(0, (h - nh) // 2)
            x0 = max(0, (w - nw) // 2)
            y1 = min(h, y0 + nh)
            x1 = min(w, x0 + nw)
            canvas[y0:y1, x0:x1] = scaled[: y1 - y0, : x1 - x0]
            out = canvas
        # light noise
        if np.random.rand() < 0.4:
            noise = np.random.normal(0, 8, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    def __getitem__(self, index: int):
        path, label = self.items[index]
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取: {path}")
        if self.augment:
            img = self._augment(img)
        cfg = self.project.config
        arr = prepare_tensor_image(img, cfg.input_width, cfg.input_height)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1xHxW
        return tensor, label


def train_project(
    project: GameProject,
    *,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    augment: bool | None = None,
    log=None,
) -> Path:
    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    use_aug = project.config.augment if augment is None else augment
    ds = CharFolderDataset(project, augment=False)
    n = len(ds)
    val_n = max(1, int(n * 0.15)) if n >= 20 else 0
    if val_n:
        train_base, val_ds = torch.utils.data.random_split(
            ds, [n - val_n, val_n], generator=torch.Generator().manual_seed(42)
        )
    else:
        train_base, val_ds = ds, None

    # Augment only on training indices via wrapper
    if use_aug:
        train_ds = CharFolderDataset(project, augment=True)
        # keep same train indices as split when possible
        if val_n and hasattr(train_base, "indices"):
            train_ds = torch.utils.data.Subset(train_ds, list(train_base.indices))  # type: ignore[attr-defined]
        else:
            train_ds = train_ds
        _log("数据增强：开（位移/对比度/缩放/噪声）")
    else:
        train_ds = train_base
        _log("数据增强：关")

    loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)
        if val_ds is not None
        else None
    )

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DigitCNN(num_classes=len(project.config.classes), in_channels=1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    from datetime import datetime

    run_dir = project.runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_acc = -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)
        train_acc = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)

        val_acc = train_acc
        if val_loader is not None:
            model.eval()
            vc = vt = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    pred = model(xb).argmax(dim=1)
                    vc += int((pred == yb).sum().item())
                    vt += xb.size(0)
            val_acc = vc / max(vt, 1)

        _log(f"epoch {epoch}/{epochs}  loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        history.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": project.config.classes,
                    "input_width": project.config.input_width,
                    "input_height": project.config.input_height,
                    "channels": 1,
                    "val_acc": best_acc,
                },
                best_path,
            )

    import json

    (run_dir / "metrics.json").write_text(
        json.dumps({"history": history, "best_val_acc": best_acc}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        from game_digit_trainer.confusion import compute_confusion

        if best_path.is_file():
            conf_report = compute_confusion(project, best_path)
            (run_dir / "confusion.json").write_text(
                json.dumps(conf_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pairs = conf_report.get("pairs") or []
            if pairs:
                tip = "、".join(
                    f"{p['true_display']}→{p['pred_display']}×{p['count']}" for p in pairs[:5]
                )
                _log(f"易错对: {tip}")
    except Exception as exc:
        _log(f"混淆矩阵跳过: {exc}")
    _log(f"最佳模型: {best_path}  val_acc={best_acc:.3f}")
    return best_path
