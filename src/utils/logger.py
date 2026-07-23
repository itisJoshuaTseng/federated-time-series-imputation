"""
Logging Utilities.

Supports TensorBoard and Weights & Biases for experiment tracking.
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, Optional

import numpy as np


class Logger:
    """
    Unified logger for federated training experiments.

    Supports:
      - Console logging
      - TensorBoard
      - Weights & Biases (wandb)
      - JSON file logging
    """

    def __init__(
        self,
        config: dict,
        experiment_name: Optional[str] = None,
    ):
        log_cfg = config.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "./logs")
        self.verbose = log_cfg.get("verbose", True)
        self.log_every = log_cfg.get("log_every", 10)

        # Generate experiment name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fed_cfg = config.get("federation", {})
        self.experiment_name = experiment_name or (
            f"fed_{fed_cfg.get('aggregation', 'fedavg')}"
            f"_{fed_cfg.get('num_clients', 5)}clients"
            f"_{timestamp}"
        )

        self.exp_dir = os.path.join(self.log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # TensorBoard
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=self.exp_dir)
        except ImportError:
            pass

        # Weights & Biases
        self.wandb_run = None
        if log_cfg.get("use_wandb", False):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=log_cfg.get("wandb_project", "fed-marl-imputation"),
                    entity=log_cfg.get("wandb_entity", None),
                    name=self.experiment_name,
                    config=config,
                )
            except Exception as e:
                print(f"[WARNING] Could not initialize wandb: {e}")

        # JSON log buffer
        self._log_buffer = []

    def log_round(
        self,
        round_idx: int,
        metrics: Dict[str, float],
        prefix: str = "train",
    ):
        """
        Log metrics for a federation round.

        Args:
            round_idx: current round number
            metrics: dict of metric_name → value
            prefix: 'train' or 'eval'
        """
        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}/{k}", v, round_idx)

        # Wandb
        if self.wandb_run is not None:
            import wandb
            wandb.log(
                {f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=round_idx,
            )

        # Buffer for JSON
        record = {"round": round_idx, "prefix": prefix, **metrics}
        self._log_buffer.append(record)

        # Periodic flush
        if round_idx % self.log_every == 0:
            self._flush_json()

    def log_client(
        self,
        round_idx: int,
        client_id: int,
        metrics: Dict[str, float],
    ):
        """Log per-client metrics."""
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(
                        f"client_{client_id}/{k}", v, round_idx
                    )

    def log_text(self, message: str):
        """Log a text message."""
        if self.verbose:
            print(message)

    def _flush_json(self):
        """Write buffered logs to JSON file."""
        if not self._log_buffer:
            return

        path = os.path.join(self.exp_dir, "metrics.jsonl")
        with open(path, "a") as f:
            for record in self._log_buffer:
                f.write(json.dumps(record, default=_json_default) + "\n")
        self._log_buffer = []

    def close(self):
        """Clean up logging resources."""
        self._flush_json()
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
