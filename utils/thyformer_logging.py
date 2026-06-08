"""
ThyFormer — Logging Utilities
Console + CSV logging, optional Weights & Biases.
"""
import csv
from pathlib import Path
from typing import Dict

import torch


class MetricLogger:
    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        project: str = "thyformer",
        name: str = "run",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self.use_wandb = use_wandb
        self._file = None
        self._writer = None
        if use_wandb:
            try:
                import wandb

                wandb.init(project=project, name=name)
                self._wandb = wandb
            except ImportError:
                print("wandb not installed — disabling W&B logging")
                self.use_wandb = False

    def log_step(self, epoch, step, total, losses, lr):
        ls = " | ".join(
            f"{k.replace('loss_','')}=" f"{v.item() if isinstance(v, torch.Tensor) else v:.4f}"
            for k, v in losses.items()
            if "loss" in k
        )
        pct = 100 * step / max(total, 1)
        print(f"  E{epoch:03d} [{step:4d}/{total}] ({pct:5.1f}%)  {ls}  lr={lr:.2e}")

    def log_epoch(self, metrics: Dict):
        num = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if self._writer is None and num:
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=sorted(num))
            self._writer.writeheader()
        if self._writer:
            try:
                self._writer.writerow({k: num.get(k, "") for k in self._writer.fieldnames})
                self._file.flush()
            except Exception:
                pass
        if self.use_wandb:
            self._wandb.log(num)

    def close(self):
        if self._file:
            self._file.close()
        if self.use_wandb:
            try:
                self._wandb.finish()
            except Exception:
                pass
