"""Background training thread."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from game_digit_trainer.project import GameProject
from game_digit_trainer.train import train_project
from game_digit_trainer.train_line import train_line_project


class TrainWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        project: GameProject,
        epochs: int,
        *,
        augment: bool = True,
        line_mode: bool = False,
        finetune: bool = True,
        auto_bootstrap: bool = False,
        device: str | None = None,
        num_workers: int | None = None,
    ) -> None:
        super().__init__()
        self.project = project
        self.epochs = epochs
        self.augment = augment
        self.line_mode = line_mode
        self.finetune = finetune
        self.auto_bootstrap = auto_bootstrap
        self.device = device
        self.num_workers = num_workers
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            stop = lambda: self._stop
            if self.line_mode:
                path = train_line_project(
                    self.project,
                    epochs=self.epochs,
                    augment_real=self.augment,
                    finetune=self.finetune,
                    auto_bootstrap=self.auto_bootstrap,
                    device=self.device,
                    num_workers=self.num_workers,
                    log=lambda m: self.log.emit(m),
                    should_stop=stop,
                )
            else:
                path = train_project(
                    self.project,
                    epochs=self.epochs,
                    augment=self.augment,
                    log=lambda m: self.log.emit(m),
                    should_stop=stop,
                )
            self.done.emit(str(path))
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
