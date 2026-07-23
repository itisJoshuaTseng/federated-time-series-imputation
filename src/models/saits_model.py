"""
SAITS Model Wrapper for Federated Learning.

Wraps PyPOTS's SAITS (Self-Attention Imputation for Time Series) model
to expose a clean interface for federated parameter exchange.

SAITS uses two diagonally-masked self-attention blocks to learn temporal
dependencies from observed values and jointly optimize imputation.

Reference:
    Du, W., Côté, D., & Liu, Y. (2023). SAITS: Self-Attention-based
    Imputation for Time Series. Expert Systems with Applications.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


# Parameter prefixes for Partial Decoupling (Fed-SAITS-PD).
# LOCAL: feature-mask embedding + output projections (client-specific).
# GLOBAL: temporal self-attention + FFN (cross-hospital universal).
_LOCAL_PREFIXES = (
    "encoder.embedding_1.",
    "encoder.embedding_2.",
    "encoder.reduce_dim_z.",
    "encoder.reduce_dim_beta.",
    "encoder.reduce_dim_gamma.",
    "encoder.weight_combine.",
)
_GLOBAL_PREFIXES = (
    "encoder.layer_stack_for_first_block.",
    "encoder.layer_stack_for_second_block.",
)


class FederatedSAITS:
    """
    Federated-learning-ready wrapper around PyPOTS SAITS.

    Provides:
      - Local training on a hospital's private data
      - Parameter get/set for federated aggregation
      - Evaluation with standard imputation metrics

    Args:
        num_features:   number of input features (D)
        seq_length:     time series length (T)
        n_layers:       number of transformer encoder layers
        d_model:        model dimension
        n_heads:        number of attention heads
        d_ffn:          feed-forward dimension
        d_k:            key dimension per head
        d_v:            value dimension per head
        dropout:        dropout rate
        attn_dropout:   attention dropout rate
        epochs:         local training epochs per FL round
        batch_size:     training batch size
        learning_rate:  optimizer learning rate
        patience:       early stopping patience
        device:         torch device string
    """

    def __init__(
        self,
        num_features: int,
        seq_length: int,
        n_layers: int = 2,
        d_model: int = 256,
        n_heads: int = 4,
        d_ffn: int = 128,
        d_k: int = 64,
        d_v: int = 64,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        diagonal_attention_mask: bool = True,
        ORT_weight: float = 1.0,
        MIT_weight: float = 1.0,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 5,
        device: str = "cpu",
    ):
        self.num_features = num_features
        self.seq_length = seq_length
        self.device_str = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience

        # Ensure patience < epochs (PyPOTS requirement)
        effective_patience = min(patience, max(epochs - 1, 1))

        # Store config for re-creation
        self._model_config = dict(
            n_steps=seq_length,
            n_features=num_features,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_dropout=attn_dropout,
            diagonal_attention_mask=diagonal_attention_mask,
            ORT_weight=ORT_weight,
            MIT_weight=MIT_weight,
            epochs=epochs,
            batch_size=batch_size,
            patience=effective_patience,
            # NOTE: we set saving_path to None; we handle checkpointing ourselves
            saving_path=None,
            device=self._resolve_device(device),
        )

        # Create the SAITS model
        self.model = self._create_model()

    @staticmethod
    def _resolve_device(device_str: str):
        """Resolve device string for PyPOTS."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def _create_model(self):
        """Create a fresh SAITS model from PyPOTS."""
        from pypots.imputation import SAITS
        return SAITS(**self._model_config)

    # ================================================================
    # Training Parameter Update (for centralized / local modes)
    # ================================================================

    def set_training_params(self, epochs: int, patience: Optional[int] = None):
        """
        Update training epochs and patience on the internal PyPOTS model.

        This is needed because centralized / local-only modes train for
        more epochs than a single FL round, but the PyPOTS model was
        already created with the per-round epoch count.

        Args:
            epochs:   new total number of training epochs
            patience: new early-stopping patience (default: keep current)
        """
        self.epochs = epochs
        self._model_config["epochs"] = epochs
        # Propagate to the internal PyPOTS SAITS model
        if hasattr(self.model, "epochs"):
            self.model.epochs = epochs

        if patience is not None:
            eff = min(patience, max(epochs - 1, 1))
            self._model_config["patience"] = eff
            if hasattr(self.model, "patience"):
                self.model.patience = eff

    # ================================================================
    # Data Conversion (our format → PyPOTS format)
    # ================================================================

    @staticmethod
    def prepare_pypots_data(
        observed: np.ndarray,
        masks: np.ndarray,
    ) -> dict:
        """
        Convert our (N, T, D) data format to PyPOTS input format.

        PyPOTS expects a dict with key "X" containing (N, T, D) array
        where missing values are np.nan.

        Args:
            observed: (N, T, D) observed values (0 where missing)
            masks:    (N, T, D) binary masks (1=observed, 0=missing)

        Returns:
            dict with "X" key for PyPOTS
        """
        X = observed.copy().astype(np.float64)
        X[masks < 0.5] = np.nan
        return {"X": X}

    @staticmethod
    def prepare_pypots_test_data(
        observed: np.ndarray,
        masks: np.ndarray,
        ground_truth: np.ndarray,
    ) -> dict:
        """
        Prepare test/validation data with ground truth for evaluation.

        Args:
            observed:     (N, T, D) observed values
            masks:        (N, T, D) binary masks
            ground_truth: (N, T, D) complete ground truth

        Returns:
            dict with "X" and "X_ori" keys for PyPOTS
        """
        X = observed.copy().astype(np.float64)
        X[masks < 0.5] = np.nan
        return {
            "X": X,
            "X_ori": ground_truth.copy().astype(np.float64),
        }

    # ================================================================
    # FedProx Proximal Term (gradient hooks)
    # ================================================================

    def enable_fedprox(
        self,
        global_params: Dict[str, torch.Tensor],
        mu: float = 0.01,
    ):
        """
        Enable FedProx proximal regularisation via gradient hooks.

        Registers a hook on every trainable parameter so that during
        back-propagation the gradient is augmented with:

            ∇_prox = μ · (w − w_global)

        which is equivalent to adding  μ/2 · ‖w − w_global‖²  to the loss.

        Reference:
            Li et al., "Federated Optimization in Heterogeneous Networks",
            MLSys 2020.

        Args:
            global_params: flat dict  {param_name: tensor}  of the global
                           model received at the start of this FL round.
            mu:            proximal penalty weight (default 0.01).
        """
        self._fedprox_hooks: list = []
        device = next(self.model.model.parameters()).device

        for name, param in self.model.model.named_parameters():
            if not param.requires_grad:
                continue
            # Keep a detached copy on the same device
            global_w = global_params[name].detach().clone().to(device)

            # The hook receives the gradient tensor and returns a modified one
            def _hook(grad, gw=global_w, p=param):
                return grad + mu * (p.data - gw)

            handle = param.register_hook(_hook)
            self._fedprox_hooks.append(handle)

    def disable_fedprox(self):
        """Remove all FedProx gradient hooks."""
        for h in getattr(self, "_fedprox_hooks", []):
            h.remove()
        self._fedprox_hooks = []

    # ================================================================
    # Training
    # ================================================================

    def fit(
        self,
        observed: np.ndarray,
        masks: np.ndarray,
        val_observed: Optional[np.ndarray] = None,
        val_masks: Optional[np.ndarray] = None,
        val_ground_truth: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train SAITS on local hospital data.

        Args:
            observed:       (N, T, D) training observed values
            masks:          (N, T, D) training masks
            val_observed:   optional validation observed values
            val_masks:      optional validation masks
            val_ground_truth: optional validation ground truth

        Returns:
            training metrics dict
        """
        train_data = self.prepare_pypots_data(observed, masks)

        val_data = None
        if val_observed is not None and val_masks is not None:
            if val_ground_truth is not None:
                val_data = self.prepare_pypots_test_data(
                    val_observed, val_masks, val_ground_truth
                )
            else:
                val_data = self.prepare_pypots_data(val_observed, val_masks)

        self.model.fit(train_set=train_data, val_set=val_data)

        return {"status": "trained"}

    # ================================================================
    # Imputation (Inference)
    # ================================================================

    def impute(
        self,
        observed: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Perform imputation on given data.

        Args:
            observed: (N, T, D) observed values (0 where missing)
            masks:    (N, T, D) binary masks

        Returns:
            imputed: (N, T, D) fully imputed time series
        """
        test_data = self.prepare_pypots_data(observed, masks)
        result = self.model.predict(test_data)
        return result["imputation"]

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(
        self,
        observed: np.ndarray,
        masks: np.ndarray,
        ground_truth: np.ndarray,
        eval_masks: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate imputation quality.

        Two evaluation modes:
          1. eval_masks provided: evaluate ONLY on artificially hidden
             positions (originally observed → reliable ground truth).
             The input is modified to also hide eval positions.
          2. eval_masks=None: fall back to all naturally missing positions
             (ground truth from ffill/bfill — less reliable).

        Args:
            observed:     (N, T, D) observed values
            masks:        (N, T, D) binary masks (1=observed, 0=missing)
            ground_truth: (N, T, D) complete ground truth
            eval_masks:   (N, T, D) optional; 1=artificially hidden for eval

        Returns:
            dict with MAE, RMSE, MRE metrics
        """
        if eval_masks is not None and eval_masks.sum() > 0:
            # Hide eval positions from the model input so it must predict them
            eval_input_obs = observed * (1.0 - eval_masks)
            eval_input_masks = masks * (1.0 - eval_masks)
            imputed = self.impute(eval_input_obs, eval_input_masks)

            eval_pos = eval_masks > 0.5
            errors = imputed[eval_pos] - ground_truth[eval_pos]
        else:
            # Fallback: evaluate on all naturally missing positions
            imputed = self.impute(observed, masks)
            eval_pos = masks < 0.5
            if eval_pos.sum() == 0:
                return {"mae": 0.0, "rmse": 0.0, "mre": 0.0}
            errors = imputed[eval_pos] - ground_truth[eval_pos]

        abs_errors = np.abs(errors)
        mae = float(abs_errors.mean())
        rmse = float(np.sqrt((errors ** 2).mean()))

        denom = np.abs(ground_truth[eval_pos]).sum()
        mre = float(abs_errors.sum() / denom) if denom > 1e-8 else 0.0

        return {"mae": mae, "rmse": rmse, "mre": mre}

    # ================================================================
    # Federated Parameter Interface
    # ================================================================

    def get_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract model parameters for federated upload.

        Returns nested dict structure compatible with our aggregation functions:
            {"saits": {"param_name": tensor, ...}}
        """
        state_dict = self.model.model.state_dict()
        return {
            "saits": {
                k: v.cpu().clone() for k, v in state_dict.items()
            }
        }

    def set_parameters(self, params: Dict[str, Dict[str, torch.Tensor]]):
        """
        Load model parameters from federated download.

        Args:
            params: {"saits": {"param_name": tensor, ...}}
        """
        if "saits" in params:
            device = next(self.model.model.parameters()).device
            state_dict = {
                k: v.to(device) for k, v in params["saits"].items()
            }
            self.model.model.load_state_dict(state_dict)

    def get_flat_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters as a flat dict (for simple aggregation)."""
        return {
            k: v.cpu().clone()
            for k, v in self.model.model.state_dict().items()
        }

    def set_flat_parameters(self, state_dict: Dict[str, torch.Tensor]):
        """Set parameters from a flat dict."""
        device = next(self.model.model.parameters()).device
        self.model.model.load_state_dict(
            {k: v.to(device) for k, v in state_dict.items()}
        )

    # ================================================================
    # Partial Decoupling: Global / Local Parameter Split
    # ================================================================

    @staticmethod
    def _is_global_param(name: str) -> bool:
        return any(name.startswith(p) for p in _GLOBAL_PREFIXES)

    @staticmethod
    def _is_local_param(name: str) -> bool:
        return any(name.startswith(p) for p in _LOCAL_PREFIXES)

    def get_global_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract only GLOBAL (attention + FFN) parameters for aggregation.

        Returns:
            {"saits": {param_name: tensor, ...}} containing only global layers.
        """
        state_dict = self.model.model.state_dict()
        return {
            "saits": {
                k: v.cpu().clone()
                for k, v in state_dict.items()
                if self._is_global_param(k)
            }
        }

    def set_global_parameters(
        self, params: Dict[str, Dict[str, torch.Tensor]]
    ):
        """
        Load only GLOBAL parameters from server. Local layers are untouched.
        """
        if "saits" not in params:
            return
        device = next(self.model.model.parameters()).device
        current = self.model.model.state_dict()
        for k, v in params["saits"].items():
            if self._is_global_param(k):
                current[k] = v.to(device)
        self.model.model.load_state_dict(current)

    def get_local_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract only LOCAL (embedding + reduce_dim) parameters.
        """
        state_dict = self.model.model.state_dict()
        return {
            "saits": {
                k: v.cpu().clone()
                for k, v in state_dict.items()
                if self._is_local_param(k)
            }
        }

    # ================================================================
    # Save / Load
    # ================================================================

    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.model.state_dict(),
            "model_config": self._model_config,
            "num_features": self.num_features,
            "seq_length": self.seq_length,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.model.load_state_dict(checkpoint["model_state_dict"])
