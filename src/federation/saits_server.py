"""
SAITS Federated Server.

Orchestrates the federated learning process for SAITS imputation.
Each round: select clients → distribute global model → local training
→ aggregate → evaluate → checkpoint.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from src.federation.saits_client import SAITSClient
from src.federation.aggregation import (
    fedavg, fedprox, fedadam, FedAdamState,
    complementarity_aware_aggregation,
    compute_complementarity_matrix,
)


class SAITSFederatedServer:
    """
    Central server for federated SAITS training.

    Supports three aggregation strategies:
      - FedAvg  (weighted average)
      - FedProx (same server-side, proximal term handled client-side)
      - FedAdam (server-side adaptive optimizer)

    Args:
        clients:        list of SAITSClient instances
        test_data:      TimeSeriesDataset for global testing
        config:         configuration dict
        device:         torch device string
    """

    def __init__(
        self,
        clients: List[SAITSClient],
        test_data=None,
        config: dict = None,
        device: str = "cpu",
    ):
        self.clients = clients
        self.test_data = test_data
        self.config = config or {}
        self.device = device

        fed_cfg = self.config.get("federation", {})
        self.num_rounds = fed_cfg.get("rounds", 100)
        self.frac_clients = fed_cfg.get("frac_clients", 1.0)
        self.aggregation_method = fed_cfg.get("aggregation", "fedavg")

        train_cfg = self.config.get("training", {})
        self.patience = train_cfg.get("early_stop_patience", 20)
        self.checkpoint_dir = train_cfg.get(
            "checkpoint_dir", "checkpoints/saits"
        )
        self.eval_every = train_cfg.get("eval_every", 1)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Global model parameters (initialized from first client)
        self.global_params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        # FedAdam state
        if self.aggregation_method == "fedadam":
            self._fedadam_state = FedAdamState(
                beta1=fed_cfg.get("beta1", 0.9),
                beta2=fed_cfg.get("beta2", 0.999),
                tau=fed_cfg.get("tau", 1e-3),
                server_lr=fed_cfg.get("server_lr", 0.01),
            )

        # Fed-SAITS-CA / PD / CA+PD flags
        self._use_ca = self.aggregation_method in ("fed_ca", "fed_ca_pd")
        self._use_pd = self.aggregation_method in ("fed_pd", "fed_ca_pd")

        if self._use_ca:
            self._ca_tau = fed_cfg.get("ca_tau", 1.0)
            self._ca_gamma = fed_cfg.get("ca_gamma", 0.02)
            self._ca_scale_factor = fed_cfg.get("ca_scale_factor", 4)
            self._ca_alpha = fed_cfg.get("ca_alpha", 0.95)
            self._mechanism_coefs = None
            self._personalized_params = {}   # client_id -> params

        if self._use_pd:
            self._personalized_params = getattr(
                self, "_personalized_params", {}
            )

        # Tracking
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.history: List[dict] = []

    # ----------------------------------------------------------------
    # Client Selection
    # ----------------------------------------------------------------

    def _select_clients(self) -> List[SAITSClient]:
        """Randomly select a subset of clients for this round."""
        num_selected = max(1, int(len(self.clients) * self.frac_clients))
        if num_selected >= len(self.clients):
            return self.clients
        indices = np.random.choice(
            len(self.clients), num_selected, replace=False
        )
        return [self.clients[i] for i in indices]

    # ----------------------------------------------------------------
    # Aggregation
    # ----------------------------------------------------------------

    def _aggregate(
        self,
        client_params: List[Dict[str, Dict[str, torch.Tensor]]],
        client_weights: List[float],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Aggregate client parameters using the configured strategy."""
        if self.aggregation_method == "fedavg":
            return fedavg(client_params, client_weights)
        elif self.aggregation_method == "fedprox":
            return fedprox(client_params, client_weights)
        elif self.aggregation_method == "fedadam":
            new_params, self._fedadam_state = fedadam(
                client_params,
                self.global_params,
                self._fedadam_state,
                client_weights,
            )
            return new_params
        else:
            raise ValueError(
                f"Unknown aggregation: {self.aggregation_method}"
            )

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate global model on the held-out test set.

        Uses the first client's model (loaded with global params) to impute.
        """
        if self.test_data is None:
            return {}

        # Load global params into a client for inference
        eval_client = self.clients[0]
        if self.global_params is not None:
            eval_client.download_global_model(self.global_params)

        observed = self.test_data.data.numpy()
        masks = self.test_data.masks.numpy()
        ground_truth = self.test_data.ground_truth.numpy()
        eval_masks = self.test_data.eval_masks.numpy()

        return eval_client.model.evaluate(observed, masks, ground_truth, eval_masks)

    def evaluate_per_client(self) -> Dict[str, object]:
        """
        Evaluate the global model on EACH client's local data separately.

        For CA aggregation, each client is evaluated with its own
        personalized model. For other methods, a single global model
        is used for all clients.

        Returns:
            dict with:
                "global":      global test metrics (MAE, RMSE, MRE)
                "per_client":  list of per-client metric dicts
                "fairness":    fairness summary (std, gap, worst, best)
        """
        if self.global_params is None:
            return {}

        per_client_metrics = []
        for client in self.clients:
            cid = client.client_id

            if self._use_pd:
                # PD modes: each client has its own local layers.
                # Load the latest global layers into the client's model,
                # then evaluate using the client itself.
                if self._use_ca and cid in self._personalized_params:
                    client.download_global_layers(
                        self._personalized_params[cid]
                    )
                else:
                    client.download_global_layers(self.global_params)
                eval_model = client.model
            elif self._use_ca and cid in self._personalized_params:
                # CA without PD: use eval_client with personalized params
                eval_model = self.clients[0].model
                self.clients[0].download_global_model(
                    self._personalized_params[cid]
                )
            else:
                # Standard: single global model
                eval_model = self.clients[0].model
                self.clients[0].download_global_model(self.global_params)

            obs = client.train_observed
            msk = client.train_masks
            gt  = client.train_ground_truth
            em  = client.train_eval_masks
            metrics = eval_model.evaluate(obs, msk, gt, em)
            metrics["client_id"] = cid
            metrics["num_samples"] = client.num_samples
            metrics["local_missing_rate"] = float(1.0 - msk.mean())
            per_client_metrics.append(metrics)

        # Global test metrics
        global_metrics = self.evaluate()

        # Fairness analysis
        maes = [m.get("mae", 0) for m in per_client_metrics]
        fairness = {
            "mae_mean": float(np.mean(maes)),
            "mae_std":  float(np.std(maes)),
            "mae_best": float(np.min(maes)),
            "mae_worst": float(np.max(maes)),
            "mae_gap":  float(np.max(maes) - np.min(maes)),
            "mae_cv":   float(np.std(maes) / (np.mean(maes) + 1e-9)),
        }

        return {
            "global": global_metrics,
            "per_client": per_client_metrics,
            "fairness": fairness,
        }

    # ----------------------------------------------------------------
    # Main Training Loop
    # ----------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """
        Run the full federated training process.

        Returns:
            history dict with per-round metrics
        """
        print("=" * 60)
        print(f"Federated SAITS Training")
        print(f"  Clients:     {len(self.clients)}")
        print(f"  Rounds:      {self.num_rounds}")
        print(f"  Aggregation: {self.aggregation_method}")
        print(f"  Device:      {self.device}")
        print("=" * 60)

        # Initialize global model from first client
        if self._use_pd:
            self.global_params = self.clients[0].upload_global_layers()
        else:
            self.global_params = self.clients[0].upload_local_model()

        # Collect mechanism coefficient vectors for CA
        if self._use_ca:
            self._mechanism_coefs = {
                c.client_id: c.get_mechanism_coefs()
                for c in self.clients
            }
            comp = compute_complementarity_matrix(
                [self._mechanism_coefs[c.client_id] for c in self.clients]
            )
            print(f"  CA complementarity matrix (alpha={self._ca_alpha}, "
                  f"gamma={self._ca_gamma}, scale_factor={self._ca_scale_factor}):")
            for i, ci in enumerate(self.clients):
                row = ", ".join(f"{comp[i,j]:.3f}" for j, _ in enumerate(self.clients))
                print(f"    Client {ci.client_id}: [{row}]")

        if self._use_pd:
            n_global = sum(
                v.numel()
                for v in self.global_params.get("saits", {}).values()
            )
            n_total = sum(
                p.numel()
                for p in self.clients[0].model.model.model.parameters()
            )
            print(f"  PD split: GLOBAL={n_global:,} params ({n_global/n_total*100:.1f}%), "
                  f"LOCAL={n_total-n_global:,} params ({(n_total-n_global)/n_total*100:.1f}%)")

        for round_idx in range(1, self.num_rounds + 1):
            round_start = time.time()
            print(f"\n--- Round {round_idx}/{self.num_rounds} ---")

            # 1. Select clients
            selected = self._select_clients()
            print(f"  Selected {len(selected)} clients: "
                  f"{[c.client_id for c in selected]}")

            # 2. Distribute model
            for client in selected:
                cid = client.client_id
                if self._use_ca and cid in self._personalized_params:
                    # CA (with or without PD): per-client personalized params
                    if self._use_pd:
                        client.download_global_layers(
                            self._personalized_params[cid]
                        )
                    else:
                        client.download_global_model(
                            self._personalized_params[cid]
                        )
                elif self._use_pd:
                    # PD only: distribute only global layers
                    client.download_global_layers(self.global_params)
                else:
                    # Standard: distribute full model
                    client.download_global_model(self.global_params)

            # 3. Local training
            local_metrics = []
            client_params = []
            client_weights = []

            for client in selected:
                metrics = client.local_train()
                local_metrics.append(metrics)
                # PD: only upload global layers for aggregation
                if self._use_pd:
                    client_params.append(client.upload_global_layers())
                else:
                    client_params.append(client.upload_local_model())
                client_weights.append(float(client.get_num_samples()))
                print(f"  Client {client.client_id}: "
                      f"val_mae={metrics.get('val_mae', 'N/A'):.6f}")

            # 4. Aggregate
            if self._use_ca:
                # Complementarity-aware per-client aggregation (CAFÉ-style)
                selected_coefs = [
                    self._mechanism_coefs[c.client_id] for c in selected
                ]
                selected_sizes = [
                    float(c.get_num_samples()) for c in selected
                ]
                personalized_list = complementarity_aware_aggregation(
                    client_params,
                    selected_coefs,
                    tau=self._ca_tau,
                    gamma=self._ca_gamma,
                    scale_factor=self._ca_scale_factor,
                    alpha=self._ca_alpha,
                    sample_sizes=selected_sizes,
                )
                for idx, client in enumerate(selected):
                    self._personalized_params[client.client_id] = (
                        personalized_list[idx]
                    )
                # Keep simple average as global_params for checkpointing
                self.global_params = fedavg(client_params, client_weights)
            elif self._use_pd:
                # PD only: standard FedAvg on global layers
                self.global_params = fedavg(client_params, client_weights)
            else:
                self.global_params = self._aggregate(
                    client_params, client_weights
                )

            # 5. Evaluate global model
            round_metrics = {"round": round_idx}

            avg_val_mae = np.mean([
                m.get("val_mae", 0) for m in local_metrics
            ])
            avg_val_rmse = np.mean([
                m.get("val_rmse", 0) for m in local_metrics
            ])
            round_metrics["avg_local_mae"] = avg_val_mae
            round_metrics["avg_local_rmse"] = avg_val_rmse

            if round_idx % self.eval_every == 0 and self.test_data is not None:
                test_metrics = self.evaluate()
                round_metrics.update(
                    {f"test_{k}": v for k, v in test_metrics.items()}
                )
                test_mae = test_metrics.get("mae", float("inf"))
                print(f"  Global test MAE:  {test_mae:.6f}")
                print(f"  Global test RMSE: {test_metrics.get('rmse', 0):.6f}")

                # Early stopping & checkpointing
                if test_mae < self.best_metric:
                    self.best_metric = test_mae
                    self.patience_counter = 0
                    self._save_checkpoint(round_idx, is_best=True)
                    print(f"  ★ New best! MAE={test_mae:.6f}")
                else:
                    self.patience_counter += 1
            else:
                test_mae = None

            round_time = time.time() - round_start
            round_metrics["time"] = round_time
            self.history.append(round_metrics)
            print(f"  Round time: {round_time:.1f}s")

            # Periodic checkpoint
            if round_idx % 10 == 0:
                self._save_checkpoint(round_idx)

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n  Early stopping at round {round_idx} "
                      f"(patience={self.patience})")
                break

        print("\n" + "=" * 60)
        print(f"Training complete. Best test MAE: {self.best_metric:.6f}")
        print("=" * 60)

        # Final per-client evaluation
        final_eval = self.evaluate_per_client()
        if final_eval:
            print("\n--- Per-Client Evaluation (Global Model) ---")
            for m in final_eval.get("per_client", []):
                print(f"  Client {m['client_id']:2d}: "
                      f"n={m['num_samples']:4d}, "
                      f"miss={m['local_missing_rate']:.1%}, "
                      f"MAE={m.get('mae', 0):.6f}, "
                      f"RMSE={m.get('rmse', 0):.6f}")
            f = final_eval.get("fairness", {})
            print(f"  Fairness: MAE mean={f.get('mae_mean',0):.6f}, "
                  f"std={f.get('mae_std',0):.6f}, "
                  f"gap={f.get('mae_gap',0):.6f}, "
                  f"CV={f.get('mae_cv',0):.3f}")

        return {
            "history": self.history,
            "best_mae": self.best_metric,
            "final_eval": final_eval,
        }

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_checkpoint(self, round_idx: int, is_best: bool = False):
        """Save model checkpoint."""
        path = os.path.join(
            self.checkpoint_dir,
            f"global_round_{round_idx}.pt",
        )
        torch.save({
            "round": round_idx,
            "global_params": {
                group: {k: v.cpu() for k, v in params.items()}
                for group, params in self.global_params.items()
            },
            "best_metric": self.best_metric,
            "history": self.history,
        }, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save({
                "round": round_idx,
                "global_params": {
                    group: {k: v.cpu() for k, v in params.items()}
                    for group, params in self.global_params.items()
                },
                "best_metric": self.best_metric,
            }, best_path)

    def load_checkpoint(self, path: str):
        """Load a checkpoint and distribute to all clients."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.global_params = ckpt["global_params"]
        self.best_metric = ckpt.get("best_metric", float("inf"))
        for client in self.clients:
            client.download_global_model(self.global_params)
        print(f"Loaded checkpoint from {path} "
              f"(round {ckpt.get('round', '?')})")
