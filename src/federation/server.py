"""
Federated Learning Server.

Orchestrates the federated training process:
  1. Initialize global model
  2. Each round: select clients → distribute model → local training → aggregate
  3. Evaluate global model periodically
  4. Early stopping & checkpointing
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch

from .client import HospitalClient
from .aggregation import fedavg, fedprox, fedadam, FedAdamState
from ..data.dataset import TimeSeriesDataset
from ..environment.imputation_env import ImputationEnv


class FederatedServer:
    """
    Central server for Federated Multi-Agent RL Imputation.

    Coordinates multiple hospital clients, manages global model,
    and performs federated aggregation.

    Args:
        clients:      list of HospitalClient instances
        test_dataset: global test dataset for evaluation
        config:       full configuration dictionary
        device:       torch device
    """

    def __init__(
        self,
        clients: List[HospitalClient],
        test_dataset: TimeSeriesDataset,
        config: dict,
        device: str = "cpu",
    ):
        self.clients = clients
        self.test_dataset = test_dataset
        self.config = config
        self.device = device
        self.num_clients = len(clients)

        fed_cfg = config.get("federation", {})
        self.num_rounds = fed_cfg.get("rounds", 100)
        self.fraction_fit = fed_cfg.get("fraction_fit", 1.0)
        self.aggregation_strategy = fed_cfg.get("aggregation", "fedavg")

        eval_cfg = config.get("evaluation", {})
        self.eval_every = eval_cfg.get("eval_every", 5)
        self.patience = eval_cfg.get("patience", 20)
        self.save_best = eval_cfg.get("save_best", True)

        log_cfg = config.get("logging", {})
        self.save_dir = log_cfg.get("save_dir", "./checkpoints")
        self.verbose = log_cfg.get("verbose", True)

        # --- Initialize global model from first client ---
        self.global_params = clients[0].marl_system.get_all_parameters()

        # FedAdam state (if using adaptive aggregation)
        self.fedadam_state = FedAdamState() if self.aggregation_strategy == "fedadam" else None

        # Training state
        self.current_round = 0
        self.best_mae = float("inf")
        self.rounds_without_improvement = 0
        self.history: List[Dict] = []

    # ----------------------------------------------------------------
    # Main Training Loop
    # ----------------------------------------------------------------

    def train(self) -> Dict[str, List]:
        """
        Run the full federated training process.

        Returns:
            history: dict of metric lists over rounds
        """
        os.makedirs(self.save_dir, exist_ok=True)

        print("=" * 70)
        print("  Federated Multi-Agent RL for Time Series Imputation")
        print(f"  Clients: {self.num_clients} hospitals")
        print(f"  Rounds:  {self.num_rounds}")
        print(f"  Strategy: {self.aggregation_strategy}")
        print("=" * 70)

        for round_idx in range(self.num_rounds):
            self.current_round = round_idx + 1
            round_start = time.time()

            # --- 1. Select clients for this round ---
            selected_clients = self._select_clients()

            # --- 2. Distribute global model ---
            self._distribute_global_model(selected_clients)

            # --- 3. Local training ---
            client_results = self._run_local_training(selected_clients)

            # --- 4. Collect and aggregate ---
            self._aggregate_models(selected_clients)

            round_time = time.time() - round_start

            # --- 5. Evaluate ---
            eval_metrics = None
            if self.current_round % self.eval_every == 0:
                eval_metrics = self.evaluate()

                # Early stopping check
                if eval_metrics["mae"] < self.best_mae:
                    self.best_mae = eval_metrics["mae"]
                    self.rounds_without_improvement = 0
                    if self.save_best:
                        self._save_checkpoint("best_model.pt")
                else:
                    self.rounds_without_improvement += self.eval_every

            # --- 6. Log ---
            round_info = {
                "round": self.current_round,
                "time": round_time,
                "client_metrics": client_results,
                "eval_metrics": eval_metrics,
                "best_mae": self.best_mae,
            }
            self.history.append(round_info)

            if self.verbose:
                self._print_round_summary(round_info)

            # Early stopping
            if self.rounds_without_improvement >= self.patience:
                print(f"\n[EARLY STOPPING] No improvement for {self.patience} rounds.")
                break

        # Save final model
        self._save_checkpoint("final_model.pt")

        print(f"\n{'=' * 70}")
        print(f"  Training complete. Best MAE: {self.best_mae:.6f}")
        print(f"{'=' * 70}")

        return self._compile_history()

    # ----------------------------------------------------------------
    # Round Steps
    # ----------------------------------------------------------------

    def _select_clients(self) -> List[HospitalClient]:
        """Select a subset of clients for the current round."""
        num_selected = max(1, int(self.num_clients * self.fraction_fit))

        if num_selected >= self.num_clients:
            return self.clients

        selected_idx = np.random.choice(
            self.num_clients, size=num_selected, replace=False
        )
        return [self.clients[i] for i in selected_idx]

    def _distribute_global_model(self, clients: List[HospitalClient]):
        """Send current global model to selected clients."""
        for client in clients:
            client.download_global_model(self.global_params)

    def _run_local_training(
        self,
        clients: List[HospitalClient],
    ) -> List[Dict[str, float]]:
        """Run local training on selected clients."""
        results = []
        for client in clients:
            if self.verbose:
                print(
                    f"  Round {self.current_round} | "
                    f"Training Client {client.client_id} "
                    f"({client.num_samples} samples)..."
                )
            metrics = client.local_train(verbose=False)
            results.append(metrics)
        return results

    def _aggregate_models(self, clients: List[HospitalClient]):
        """Collect local models and aggregate into new global model."""
        # Collect parameters and weights
        client_params = [client.upload_local_model() for client in clients]
        weights = [float(client.num_samples) for client in clients]

        # Aggregate
        if self.aggregation_strategy == "fedavg":
            self.global_params = fedavg(client_params, weights)

        elif self.aggregation_strategy == "fedprox":
            self.global_params = fedprox(client_params, weights)

        elif self.aggregation_strategy == "fedadam":
            self.global_params, self.fedadam_state = fedadam(
                client_params, self.global_params, self.fedadam_state, weights
            )
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation_strategy}")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def evaluate(
        self,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate the global model on the test dataset.

        Uses the first client's MARL system with global parameters.
        """
        # Load global params into evaluation client
        eval_client = self.clients[0]
        eval_client.download_global_model(self.global_params)

        metrics = eval_client.evaluate(
            test_dataset=self.test_dataset,
            num_samples=num_samples,
        )

        return metrics

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_checkpoint(self, filename: str):
        """Save global model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "round": self.current_round,
            "global_params": self.global_params,
            "best_mae": self.best_mae,
            "config": self.config,
        }, path)

        if self.verbose:
            print(f"  [SAVED] {path}")

    def load_checkpoint(self, path: str):
        """Load global model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.global_params = checkpoint["global_params"]
        self.current_round = checkpoint.get("round", 0)
        self.best_mae = checkpoint.get("best_mae", float("inf"))
        print(f"[LOADED] Checkpoint from round {self.current_round}, MAE={self.best_mae:.6f}")

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------

    def _print_round_summary(self, info: Dict):
        """Print round summary to console."""
        avg_reward = np.mean([
            m.get("mean_reward", 0) for m in info["client_metrics"]
        ])
        avg_actor_loss = np.mean([
            m.get("actor_loss", 0) for m in info["client_metrics"]
        ])

        line = (
            f"  Round {info['round']:4d}/{self.num_rounds} "
            f"| time={info['time']:.1f}s "
            f"| avg_reward={avg_reward:.4f} "
            f"| actor_loss={avg_actor_loss:.4f}"
        )

        if info["eval_metrics"] is not None:
            line += (
                f" | eval_MAE={info['eval_metrics']['mae']:.6f}"
                f" | eval_RMSE={info['eval_metrics']['rmse']:.6f}"
                f" | best={self.best_mae:.6f}"
            )

        print(line)

    def _compile_history(self) -> Dict[str, list]:
        """Compile training history into metric lists."""
        compiled = {
            "rounds": [],
            "times": [],
            "avg_rewards": [],
            "avg_actor_losses": [],
            "eval_mae": [],
            "eval_rmse": [],
        }

        for info in self.history:
            compiled["rounds"].append(info["round"])
            compiled["times"].append(info["time"])

            avg_r = np.mean([m.get("mean_reward", 0) for m in info["client_metrics"]])
            avg_l = np.mean([m.get("actor_loss", 0) for m in info["client_metrics"]])
            compiled["avg_rewards"].append(avg_r)
            compiled["avg_actor_losses"].append(avg_l)

            if info["eval_metrics"] is not None:
                compiled["eval_mae"].append(info["eval_metrics"]["mae"])
                compiled["eval_rmse"].append(info["eval_metrics"]["rmse"])

        return compiled
