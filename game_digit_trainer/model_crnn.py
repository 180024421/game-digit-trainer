"""行识别 CRNN（CTC）：整行一次前向，不依赖切字。"""
from __future__ import annotations

import torch
from torch import nn


class DigitCRNN(nn.Module):
    """输入 NCHW 1×H×W（H 建议 32），输出 (T, N, C) logits，C = num_classes + 1（blank 在最后）。"""

    def __init__(self, num_classes: int, *, hidden: int = 128) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes 无效")
        self.num_classes = num_classes
        self.blank_index = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # H/8, W/4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # H/16, W/4
            nn.Conv2d(128, 128, kernel_size=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # ~1 x W/4
        )
        self.rnn = nn.LSTM(128, hidden, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden * 2, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N,1,H,W
        feat = self.cnn(x)
        # N,C,H,W → expect H==1
        if feat.size(2) != 1:
            feat = nn.functional.adaptive_avg_pool2d(feat, (1, feat.size(3)))
        feat = feat.squeeze(2)  # N,C,W
        feat = feat.permute(2, 0, 1)  # W,N,C  → T,N,C
        out, _ = self.rnn(feat)
        return self.fc(out)  # T,N,C_out
