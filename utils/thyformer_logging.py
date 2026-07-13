"""
ThyFormer — Logging Utilities
Console + CSV logging, optional Weights & Biases.

The CSV (metrics.csv) holds one row per training epoch plus a final test row.
It is rebuilt from an in-memory buffer and swapped in atomically on every write,
so that:
  • columns that only show up later (e.g. the test_* metrics) are added without
    losing or misaligning earlier rows,
  • resuming a run reloads the existing rows and appends instead of wiping them,
  • re-running an epoch (resume from an older checkpoint) overwrites that epoch's
    row rather than duplicating it,
  • a crash mid-write can't corrupt the file (temp file + atomic replace).
"""
import csv
from pathlib import Path
from typing import Dict, List

import torch


def _coerce(v):
    """Parse a CSV cell back into int/float when possible (used on resume)."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() else f


class MetricLogger:
    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        project: str = "thyformer",
        name: str = "run",
        resume: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self.use_wandb = use_wandb
        self._rows: List[Dict] = []
        self._fields: List[str] = []
        if resume and self.csv_path.exists():
            self._load_existing()
        if use_wandb:
            try:
                import wandb

                wandb.init(project=project, name=name)
                self._wandb = wandb
            except ImportError:
                print("wandb not installed — disabling W&B logging")
                self.use_wandb = False

    def _load_existing(self):
        """Reload prior rows so a resumed run appends instead of overwriting."""
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self._fields = list(reader.fieldnames or [])
            for raw in reader:
                row = {k: _coerce(v) for k, v in raw.items()}
                self._rows.append({k: v for k, v in row.items() if v is not None})

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
        if not num:
            return
        # Idempotent per epoch: a resumed / re-run epoch replaces its old row.
        if "epoch" in num:
            self._rows = [r for r in self._rows if r.get("epoch") != num["epoch"]]
        self._rows.append(num)
        for k in num:
            if k not in self._fields:
                self._fields.append(k)
        self._flush_csv()
        if self.use_wandb:
            self._wandb.log(num)

    def _ordered_fields(self) -> List[str]:
        rest = sorted(f for f in self._fields if f != "epoch")
        return (["epoch"] + rest) if "epoch" in self._fields else rest

    def _flush_csv(self):
        fields = self._ordered_fields()
        tmp = self.csv_path.with_name(self.csv_path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        tmp.replace(self.csv_path)  # atomic swap on the same filesystem

    def close(self):
        if self.use_wandb:
            try:
                self._wandb.finish()
            except Exception:
                pass
