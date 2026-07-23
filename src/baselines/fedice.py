"""
Federated ICE baseline for MNAR reconstruction experiments.

This adapts CAFE's ICE-style baseline to the current VitalDB tensor setup by
flattening each client's (N, T, D) time series into tabular rows (N*T, D), then
running chained linear-regression imputation with federated coefficient
aggregation.  The baseline intentionally does not model temporal structure; it
is the linear/tabular counterpart to Fed-SAITS.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class _LocalFeatureModel:
    weights: np.ndarray
    sample_size: int


class FedICEImputer:
    """
    Iterative chained-equations imputer with federated linear-model averaging.

    Args:
        n_rounds: Number of ICE passes over all features. CAFE defaults to 20.
        ridge_alpha: L2 regularization strength for per-feature regressors.
        use_ca: If True, produce personalized CAFE-style aggregations
            (FedICE-CA). If False, use one FedAvg coefficient vector per feature
            (FedICE).
        ca_alpha: CAFE mechanism-vs-size mixing coefficient.
        ca_gamma: CAFE self-preservation coefficient.
        ca_scale_factor: CAFE power-law sharpening exponent.
        clip: Clip imputed values to the observed global feature range.
        seed: Random seed used by logistic fingerprint estimation.
        min_fit_rows: Minimum observed target rows needed to fit a local model.
    """

    def __init__(
        self,
        n_rounds: int = 20,
        ridge_alpha: float = 1.0,
        use_ca: bool = False,
        ca_alpha: float = 0.95,
        ca_gamma: float = 0.02,
        ca_scale_factor: float = 4.0,
        clip: bool = True,
        seed: int = 0,
        min_fit_rows: int = 20,
    ):
        self.n_rounds = int(n_rounds)
        self.ridge_alpha = float(ridge_alpha)
        self.use_ca = bool(use_ca)
        self.ca_alpha = float(ca_alpha)
        self.ca_gamma = float(ca_gamma)
        self.ca_scale_factor = float(ca_scale_factor)
        self.clip = bool(clip)
        self.seed = int(seed)
        self.min_fit_rows = int(min_fit_rows)

        self.ca_weights_: List[np.ndarray] = []
        self.mechanism_fingerprints_: List[np.ndarray] = []

    def fit_transform(
        self,
        client_ground_truths: List[np.ndarray],
        client_masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Fit the federated ICE baseline and return imputed client tensors.

        ``client_ground_truths`` are the complete reference tensors used by the
        existing MNAR simulator.  ``client_masks`` are the post-MNAR training
        masks, where 1 means observed and 0 means missing.
        """
        if len(client_ground_truths) != len(client_masks):
            raise ValueError("client_ground_truths and client_masks must align")
        if not client_ground_truths:
            return []

        client_shapes = [x.shape for x in client_ground_truths]
        n_features = client_ground_truths[0].shape[-1]

        x_missing = [
            self._to_missing_matrix(x, m) for x, m in zip(client_ground_truths, client_masks)
        ]
        x_complete = [np.asarray(x, dtype=np.float64).reshape(-1, n_features)
                      for x in client_ground_truths]
        flat_masks = [np.asarray(m, dtype=bool).reshape(-1, n_features)
                      for m in client_masks]

        feature_means = self._global_feature_means(x_missing, n_features)
        feature_min, feature_max = self._global_feature_ranges(x_missing, feature_means)
        x_filled = [
            self._initial_fill(x, feature_means) for x in x_missing
        ]

        if self.use_ca:
            self.mechanism_fingerprints_ = [
                self._estimate_fingerprint(xc, m, n_features)
                for xc, m in zip(x_complete, flat_masks)
            ]
        else:
            self.mechanism_fingerprints_ = []

        self.ca_weights_ = []
        for _ in range(self.n_rounds):
            round_weights = []
            for feature_idx in range(n_features):
                local_models = [
                    self._fit_feature_model(xf, xm, feature_idx, feature_means[feature_idx])
                    for xf, xm in zip(x_filled, x_missing)
                ]

                if self.use_ca:
                    personalized = self._aggregate_ca(local_models)
                    round_weights.append(self._last_ca_weight_matrix)
                else:
                    global_weights = self._aggregate_fedavg(local_models)
                    personalized = [global_weights for _ in local_models]

                for client_idx, weights in enumerate(personalized):
                    self._impute_feature(
                        x_filled[client_idx],
                        x_missing[client_idx],
                        feature_idx,
                        weights,
                        feature_min[feature_idx],
                        feature_max[feature_idx],
                    )

            if self.use_ca:
                self.ca_weights_.append(np.stack(round_weights, axis=0))

        return [
            xf.reshape(shape).astype(np.float32)
            for xf, shape in zip(x_filled, client_shapes)
        ]

    @staticmethod
    def _to_missing_matrix(ground_truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = np.asarray(ground_truth, dtype=np.float64).reshape(-1, ground_truth.shape[-1])
        m = np.asarray(mask, dtype=bool).reshape(-1, ground_truth.shape[-1])
        out = x.copy()
        out[~m] = np.nan
        return out

    @staticmethod
    def _global_feature_means(
        client_matrices: List[np.ndarray],
        n_features: int,
    ) -> np.ndarray:
        means = np.zeros(n_features, dtype=np.float64)
        for d in range(n_features):
            vals = [x[:, d][np.isfinite(x[:, d])] for x in client_matrices]
            vals = [v for v in vals if v.size > 0]
            means[d] = float(np.concatenate(vals).mean()) if vals else 0.0
        return means

    @staticmethod
    def _global_feature_ranges(
        client_matrices: List[np.ndarray],
        fallback: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_features = fallback.shape[0]
        mins = fallback.copy()
        maxs = fallback.copy()
        for d in range(n_features):
            vals = [x[:, d][np.isfinite(x[:, d])] for x in client_matrices]
            vals = [v for v in vals if v.size > 0]
            if vals:
                joined = np.concatenate(vals)
                mins[d] = float(joined.min())
                maxs[d] = float(joined.max())
        return mins, maxs

    @staticmethod
    def _initial_fill(x_missing: np.ndarray, feature_means: np.ndarray) -> np.ndarray:
        x = x_missing.copy()
        rows, cols = np.where(~np.isfinite(x))
        if rows.size:
            x[rows, cols] = feature_means[cols]
        return x

    def _fit_feature_model(
        self,
        x_filled: np.ndarray,
        x_missing: np.ndarray,
        feature_idx: int,
        fallback_mean: float,
    ) -> _LocalFeatureModel:
        observed = np.isfinite(x_missing[:, feature_idx])
        sample_size = int(observed.sum())
        n_features = x_filled.shape[1]
        pred_idx = [i for i in range(n_features) if i != feature_idx]

        if sample_size < self.min_fit_rows:
            return _LocalFeatureModel(
                weights=self._fallback_weights(n_features, fallback_mean),
                sample_size=sample_size,
            )

        x_train = x_filled[observed][:, pred_idx]
        y_train = x_missing[observed, feature_idx]

        if np.nanstd(y_train) < 1e-10:
            return _LocalFeatureModel(
                weights=self._fallback_weights(n_features, float(np.nanmean(y_train))),
                sample_size=sample_size,
            )

        model = Ridge(alpha=self.ridge_alpha)
        model.fit(x_train, y_train)
        weights = np.concatenate([
            np.asarray(model.coef_, dtype=np.float64),
            np.array([float(model.intercept_)], dtype=np.float64),
        ])
        return _LocalFeatureModel(weights=weights, sample_size=sample_size)

    @staticmethod
    def _fallback_weights(n_features: int, intercept: float) -> np.ndarray:
        # One coefficient for every non-target feature, plus intercept.
        return np.concatenate([
            np.zeros(n_features - 1, dtype=np.float64),
            np.array([intercept], dtype=np.float64),
        ])

    @staticmethod
    def _aggregate_fedavg(local_models: List[_LocalFeatureModel]) -> np.ndarray:
        weights = np.stack([m.weights for m in local_models], axis=0)
        counts = np.asarray([max(m.sample_size, 1) for m in local_models], dtype=np.float64)
        counts = counts / counts.sum()
        return np.average(weights, axis=0, weights=counts)

    def _aggregate_ca(self, local_models: List[_LocalFeatureModel]) -> List[np.ndarray]:
        weights = np.stack([m.weights for m in local_models], axis=0)
        sample_sizes = np.asarray([max(m.sample_size, 1) for m in local_models], dtype=np.float64)
        comp = self._complementarity_matrix(self.mechanism_fingerprints_)
        max_size = float(sample_sizes.max()) if sample_sizes.max() > 0 else 1.0

        personalized = []
        ca_weight_rows = []
        for i in range(len(local_models)):
            other_idx = [j for j in range(len(local_models)) if j != i]
            mech_w = np.asarray([comp[i, j] for j in other_idx], dtype=np.float64) + 1e-5
            size_w = sample_sizes[other_idx] / max_size
            raw_w = (self.ca_alpha * mech_w + (1.0 - self.ca_alpha) * size_w) ** self.ca_scale_factor
            final_w = raw_w / raw_w.sum()
            peer_avg = np.average(weights[other_idx], axis=0, weights=final_w)
            personalized.append(self.ca_gamma * weights[i] + (1.0 - self.ca_gamma) * peer_avg)

            row = np.zeros(len(local_models), dtype=np.float64)
            row[other_idx] = final_w
            ca_weight_rows.append(row)

        self._last_ca_weight_matrix = np.stack(ca_weight_rows, axis=0)
        return personalized

    def _impute_feature(
        self,
        x_filled: np.ndarray,
        x_missing: np.ndarray,
        feature_idx: int,
        weights: np.ndarray,
        min_value: float,
        max_value: float,
    ) -> None:
        missing = ~np.isfinite(x_missing[:, feature_idx])
        if not missing.any():
            return

        n_features = x_filled.shape[1]
        pred_idx = [i for i in range(n_features) if i != feature_idx]
        preds = x_filled[missing][:, pred_idx] @ weights[:-1] + weights[-1]
        if self.clip:
            preds = np.clip(preds, min_value, max_value)
        x_filled[missing, feature_idx] = preds

    def _estimate_fingerprint(
        self,
        x_complete: np.ndarray,
        mask: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        seg_len = n_features + 1
        fingerprint = np.zeros(n_features * seg_len, dtype=np.float64)
        x = np.nan_to_num(x_complete, nan=0.0, posinf=0.0, neginf=0.0)

        for d in range(n_features):
            y = (~mask[:, d]).astype(int)
            if np.unique(y).size < 2:
                continue

            try:
                lr = LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=self.seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    lr.fit(x, y)
                segment = np.concatenate([
                    lr.coef_.reshape(-1).astype(np.float64),
                    np.array([float(lr.intercept_[0])], dtype=np.float64),
                ])
                fingerprint[d * seg_len:(d + 1) * seg_len] = segment
            except Exception:
                continue

        return fingerprint

    @staticmethod
    def _complementarity_matrix(fingerprints: List[np.ndarray]) -> np.ndarray:
        p = np.stack(fingerprints, axis=0)
        norms = np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
        p = p / norms
        cos = p @ p.T
        return (1.0 - cos) / 2.0
