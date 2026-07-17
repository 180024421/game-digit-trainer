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
    def __init__(self, project: GameProject) -> None:
        self.project = project
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

    def __getitem__(self, index: int):
        path, label = self.items[index]
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取: {path}")
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
    log=None,
) -> Path:
    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    ds = CharFolderDataset(project)
    n = len(ds)
    val_n = max(1, int(n * 0.15)) if n >= 20 else 0
    if val_n:
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [n - val_n, val_n], generator=torch.Generator().manual_seed(42)
        )
    else:
        train_ds, val_ds = ds, None

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

    _log(f"最佳模型: {best_path}  val_acc={best_acc:.3f}")
    return best_path
