"""
Hospital Client for Federated Learning.

Each hospital client:
  1. Receives global model parameters from the server
  2. Trains a local Multi-Agent RL system on its private data
  3. Uploads updated parameters back to the server

Supports FedProx proximal regularization for heterogeneous data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from copy import deepcopy

from ..agents.multi_agent import MultiAgentSystem
from ..environment.imputation_env import ImputationEnv
from ..data.dataset import TimeSeriesDataset


class HospitalClient:
    """
    A hospital client in the federated learning system.

    Manages:
      - Local dataset (private patient data)
      - Local MultiAgentSystem (MAPPO agents)
      - Local training loop
      - Communication with federated server (param upload/download)

    Args:
        client_id:      unique hospital identifier
        dataset:        local TimeSeriesDataset
        config:         full configuration dict
        feature_groups: per-agent feature assignment
        device:         torch device
    """

    def __init__(
        self,
        client_id: int,
        dataset: TimeSeriesDataset,
        config: dict,
        feature_groups: List[List[int]],
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.device = device
        self.feature_groups = feature_groups

        fed_cfg = config.get("federation", {})
        self.local_epochs = fed_cfg.get("local_epochs", 5)
        self.aggregation = fed_cfg.get("aggregation", "fedavg")
        self.fedprox_mu = fed_cfg.get("fedprox_mu", 0.01)

        # Initialize local Multi-Agent System
        self.marl_system = MultiAgentSystem(
            config=config,
            num_features=dataset.num_features,
            feature_groups=feature_groups,
            device=device,
        )

        # Store global parameters for FedProx
        self._global_params: Optional[Dict] = None

        # Training history
        self.history: List[Dict[str, float]] = []

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    # ----------------------------------------------------------------
    # Federation Protocol
    # ----------------------------------------------------------------

    def download_global_model(self, global_params: Dict[str, Dict]):
        """Receive and load global model parameters from server."""
        self._global_params = deepcopy(global_params)
        self.marl_system.set_all_parameters(global_params)

    def upload_local_model(self) -> Dict[str, Dict]:
        """Return local model parameters for server aggregation."""
        return self.marl_system.get_all_parameters()

    # ----------------------------------------------------------------
    # Local Training
    # ----------------------------------------------------------------

    def local_train(
        self,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Perform local training for `local_epochs` epochs on private data.

        Returns:
            metrics: average training metrics over local epochs
        """
        self.marl_system.train_mode()

        epoch_metrics = []

        for epoch in range(self.local_epochs):
            # Sample a batch of patients
            batch_indices = np.random.choice(
                len(self.dataset),
                size=min(self.config["training"].get("batch_size", 64), len(self.dataset)),
                replace=False,
            )

            episode_metrics = []

            for idx in batch_indices:
                sample = self.dataset[idx]

                # Create environment for this patient
                env = ImputationEnv(
                    data=sample["observed"].numpy(),
                    ground_truth=sample["ground_truth"].numpy(),
                    mask=sample["mask"].numpy(),
                    feature_groups=self.feature_groups,
                    config=self.config,
                )

                # Collect rollout and train
                metrics = self.marl_system.train_episode(env)
                episode_metrics.append(metrics)

            # Average metrics for this epoch
            avg_metrics = {
                k: np.mean([m[k] for m in episode_metrics])
                for k in episode_metrics[0].keys()
            }

            # Apply FedProx proximal term if needed
            if self.aggregation == "fedprox" and self._global_params is not None:
                self._apply_fedprox_penalty()

            epoch_metrics.append(avg_metrics)

            if verbose:
                print(
                    f"  [Client {self.client_id}] Epoch {epoch + 1}/{self.local_epochs} "
                    f"| reward={avg_metrics['mean_reward']:.4f} "
                    f"| actor_loss={avg_metrics['actor_loss']:.4f}"
                )

        # Aggregate over epochs
        final_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }
        final_metrics["client_id"] = self.client_id
        final_metrics["num_samples"] = self.num_samples

        self.history.append(final_metrics)
        return final_metrics

    def _apply_fedprox_penalty(self):
        """
        Apply FedProx proximal term to gradients.

        Adds μ/2 * ||w - w_global||^2 penalty to prevent local models
        from diverging too far from the global model.
        """
        if self._global_params is None:
            return

        local_params = self.marl_system.get_all_parameters()

        for component_key in local_params:
            if component_key not in self._global_params:
                continue
            for param_key in local_params[component_key]:
                if param_key not in self._global_params[component_key]:
                    continue
                local_p = local_params[component_key][param_key]
                global_p = self._global_params[component_key][param_key].to(self.device)

                # Proximal gradient: move toward global model
                local_params[component_key][param_key] = (
                    local_p - self.fedprox_mu * (local_p - global_p)
                )

        self.marl_system.set_all_parameters(local_params)

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def evaluate(
        self,
        test_dataset: Optional[TimeSeriesDataset] = None,
        num_samples: int = 50,
    ) -> Dict[str, float]:
        """
        Evaluate the local model on test data.

        Args:
            test_dataset: dataset to evaluate on (uses local data if None)
            num_samples: number of samples to evaluate

        Returns:
            metrics: evaluation metrics (MAE, RMSE, etc.)
        """
        self.marl_system.eval_mode()
        dataset = test_dataset or self.dataset

        indices = np.random.choice(
            len(dataset),
            size=min(num_samples, len(dataset)),
            replace=False,
        )

        all_mae = []
        all_rmse = []

        with torch.no_grad():
            for idx in indices:
                sample = dataset[idx]

                env = ImputationEnv(
                    data=sample["observed"].numpy(),
                    ground_truth=sample["ground_truth"].numpy(),
                    mask=sample["mask"].numpy(),
                    feature_groups=self.feature_groups,
                    config=self.config,
                )

                # Run full episode (greedy)
                observations, _ = env.reset()
                done = False
                while not done:
                    comm_messages = self.marl_system._communicate(observations)
                    actions = []
                    for i, agent in enumerate(self.marl_system.agents):
                        comm_msg = comm_messages[i] if comm_messages else None
                        action, _, _ = agent.select_action(
                            observations[i], comm_msg, deterministic=True
                        )
                        actions.append(action)

                    observations, _, terminated, truncated, info = env.step(actions)
                    done = terminated or truncated

                # Compute metrics on imputed series
                imputed = env.get_imputed_series()
                gt = sample["ground_truth"].numpy()
                mask = sample["mask"].numpy()

                missing = mask < 0.5
                if missing.sum() > 0:
                    mae = np.abs(imputed[missing] - gt[missing]).mean()
                    rmse = np.sqrt(((imputed[missing] - gt[missing]) ** 2).mean())
                    all_mae.append(mae)
                    all_rmse.append(rmse)

        self.marl_system.train_mode()

        return {
            "mae": float(np.mean(all_mae)) if all_mae else 0.0,
            "rmse": float(np.mean(all_rmse)) if all_rmse else 0.0,
            "num_evaluated": len(all_mae),
        }
