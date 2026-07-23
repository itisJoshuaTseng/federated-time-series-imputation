"""
SAITS Federated Client.

A hospital client that trains a SAITS imputation model on local data
and participates in federated aggregation. Each client holds a private
partition of the VitalDB dataset.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.models.saits_model import FederatedSAITS


class SAITSClient:
    """
    Hospital client for federated SAITS training.

    Each client:
      1. Downloads global model parameters from the server.
      2. Trains SAITS locally on its private data.
      3. Uploads updated parameters to the server.

    Args:
        client_id:    unique identifier for this hospital
        train_data:   TimeSeriesDataset for training
        val_data:     TimeSeriesDataset or None for validation
        config:       configuration dict
        device:       torch device string
    """

    def __init__(
        self,
        client_id: int,
        train_data,
        val_data=None,
        config: dict = None,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.val_data = val_data
        self.config = config or {}
        self.device = device

        # Extract data arrays from TimeSeriesDataset
        self.train_observed = train_data.data.numpy()
        self.train_masks = train_data.masks.numpy()
        self.train_ground_truth = train_data.ground_truth.numpy()
        self.train_eval_masks = train_data.eval_masks.numpy()

        if val_data is not None:
            self.val_observed = val_data.data.numpy()
            self.val_masks = val_data.masks.numpy()
            self.val_ground_truth = val_data.ground_truth.numpy()
            self.val_eval_masks = val_data.eval_masks.numpy()
        else:
            # Use a portion of training data for validation
            n = len(self.train_observed)
            val_size = max(1, int(n * 0.1))
            self.val_observed = self.train_observed[-val_size:]
            self.val_masks = self.train_masks[-val_size:]
            self.val_ground_truth = self.train_ground_truth[-val_size:]
            self.val_eval_masks = self.train_eval_masks[-val_size:]

        # SAITS configuration
        saits_cfg = self.config.get("saits", {})
        data_cfg = self.config.get("data", {})

        num_features = data_cfg.get("num_features", self.train_observed.shape[-1])
        seq_length = data_cfg.get("seq_length", self.train_observed.shape[1])

        self.model = FederatedSAITS(
            num_features=num_features,
            seq_length=seq_length,
            n_layers=saits_cfg.get("n_layers", 2),
            d_model=saits_cfg.get("d_model", 256),
            n_heads=saits_cfg.get("n_heads", 4),
            d_ffn=saits_cfg.get("d_ffn", 128),
            d_k=saits_cfg.get("d_k", 64),
            d_v=saits_cfg.get("d_v", 64),
            dropout=saits_cfg.get("dropout", 0.1),
            attn_dropout=saits_cfg.get("attn_dropout", 0.0),
            diagonal_attention_mask=saits_cfg.get("diagonal_attention_mask", True),
            ORT_weight=saits_cfg.get("ORT_weight", 1.0),
            MIT_weight=saits_cfg.get("MIT_weight", 1.0),
            epochs=saits_cfg.get("local_epochs", 10),
            batch_size=saits_cfg.get("batch_size", 32),
            learning_rate=saits_cfg.get("learning_rate", 1e-3),
            patience=saits_cfg.get("patience", 5),
            device=device,
        )

        self.num_samples = len(self.train_observed)
        self._training_history: list = []

        # FedProx config
        fed_cfg = self.config.get("federation", {})
        self._use_fedprox = fed_cfg.get("aggregation", "fedavg") == "fedprox"
        self._mu = fed_cfg.get("mu", 0.01)
        self._global_flat_params = None  # set before local_train when FedProx

    # ----------------------------------------------------------------
    # Federated Interface
    # ----------------------------------------------------------------

    def download_global_model(self, global_params: Dict[str, Dict[str, torch.Tensor]]):
        """Receive global model parameters from server."""
        self.model.set_parameters(global_params)
        # Save a flat copy for FedProx proximal term
        if self._use_fedprox:
            self._global_flat_params = self.model.get_flat_parameters()

    def upload_local_model(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Send local model parameters to server."""
        return self.model.get_parameters()

    # --- Partial Decoupling interface ---

    def download_global_layers(self, global_params: Dict[str, Dict[str, torch.Tensor]]):
        """
        Receive only GLOBAL layer parameters from server.
        Local layers (embedding, reduce_dim) are preserved.
        """
        self.model.set_global_parameters(global_params)

    def upload_global_layers(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Send only GLOBAL layer parameters to server for aggregation."""
        return self.model.get_global_parameters()

    def local_train(self) -> Dict[str, float]:
        """
        Train SAITS model on local private data for one federated round.

        When FedProx is active, a proximal term μ/2·‖w − w_global‖² is
        added to the loss via gradient hooks (see FederatedSAITS).

        Returns:
            dict with training status and metrics
        """
        # --- Enable FedProx proximal regularisation if configured ---
        if self._use_fedprox and self._global_flat_params is not None:
            self.model.enable_fedprox(self._global_flat_params, mu=self._mu)

        metrics = self.model.fit(
            observed=self.train_observed,
            masks=self.train_masks,
            val_observed=self.val_observed,
            val_masks=self.val_masks,
            val_ground_truth=self.val_ground_truth,
        )

        # --- Disable hooks after training ---
        if self._use_fedprox:
            self.model.disable_fedprox()

        # Evaluate on local validation set
        eval_metrics = self.evaluate()
        metrics.update({f"val_{k}": v for k, v in eval_metrics.items()})

        self._training_history.append(metrics)
        return metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on local validation data using eval_masks.

        Returns:
            dict with MAE, RMSE, MRE metrics
        """
        return self.model.evaluate(
            observed=self.val_observed,
            masks=self.val_masks,
            ground_truth=self.val_ground_truth,
            eval_masks=self.val_eval_masks,
        )

    def impute(
        self,
        observed: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Run imputation inference."""
        return self.model.impute(observed, masks)

    def get_num_samples(self) -> int:
        """Return number of local training samples (for weighted aggregation)."""
        return self.num_samples

    def get_mechanism_coefs(self) -> np.ndarray:
        """
        Estimate the MNAR mechanism "fingerprint" of this client following the
        CAFÉ approach (Min et al., IEEE TKDE 2025, Section 4).

        For EACH feature d we fit a multivariate logistic-regression model
        that predicts whether feature d is missing, using **all D features
        (including x_d itself)** as inputs:

            P(mask_d = 0 | x_1 .. x_D) = sigmoid(W_d · x + b_d)

        Including x_d in the regressors is critical: for self-driven MNAR
        (e.g. quantile-based, where P(missing_d) depends on x_d's own value)
        the direct coefficient on x_d carries the strongest directional
        signal — negative → MNAR-Left, positive → MNAR-Right.  This matches
        CAFÉ's ``fit_one_feature`` which feeds ``X_filled`` (all features)
        to the logistic regression.

        Data substrate:  we fit on ``train_ground_truth`` — the client's
        underlying clean time series before the MNAR mask was applied in
        the experimental protocol.  This matches what CAFÉ's ICE procedure
        converges to after a few imputation rounds on each client's local
        data, and represents what a real-world client would have from its
        own historical records prior to joining the federation.  Using
        mean-imputed observed data at round 0 fails for MNAR because both
        classes (missing / observed) collapse to the same feature mean,
        destroying the directional signal.

        The concatenation of (W_d, b_d) across all D features forms a
        D × (D+1) fingerprint that captures the direction and structure
        of the client's missing mechanism.  Clients with opposite mechanisms
        produce fingerprints with cosine similarity ≈ −1, giving maximal
        complementarity after the (1 − cos) / 2 transform.

        Returns:
            (D*(D+1),) flattened fingerprint array.  Each length-(D+1)
            segment d is [coef_{d,0} .. coef_{d,D-1}, intercept_d].
        """
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold

        gt = self.train_ground_truth           # (N, T, D)  — clean data
        msk = self.train_masks                 # (N, T, D)  — 1=observed, 0=missing
        N, T, D = gt.shape

        X = gt.reshape(-1, D).astype(float)    # (N*T, D), no NaN
        M = msk.reshape(-1, D).astype(int)     # 1 = observed

        # --- Fit per-feature multivariate logistic regression ---
        # Segment length = D coefficients (all features, including the target
        # one) + 1 intercept.
        seg_len = D + 1
        fingerprint = np.zeros(D * seg_len, dtype=np.float64)
        for d in range(D):
            y_d = (1 - M[:, d]).astype(int)     # 1 = missing
            n_pos = int(y_d.sum())
            n_neg = len(y_d) - n_pos
            if n_pos < 5 or n_neg < 5:
                # Trivial feature (fully observed or fully missing) → zero row
                continue

            try:
                lr = LogisticRegressionCV(
                    Cs=[1.0],
                    cv=StratifiedKFold(3),
                    random_state=0,
                    max_iter=500,
                    n_jobs=1,
                    class_weight="balanced",
                )
                lr.fit(X, y_d)                   # all D features as inputs
                # lr.coef_: (1, D), lr.intercept_: (1,)
                segment = np.concatenate(
                    [lr.coef_[0], lr.intercept_]
                )                                # length D+1
                fingerprint[d * seg_len : (d + 1) * seg_len] = segment
            except Exception:
                pass

        return fingerprint

    @property
    def training_history(self) -> list:
        return self._training_history
