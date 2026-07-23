"""
Evaluation Metrics for Time Series Imputation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean Absolute Error (only on masked/missing entries)."""
    if mask is not None:
        missing = mask < 0.5
        if missing.sum() == 0:
            return 0.0
        return float(np.abs(predictions[missing] - targets[missing]).mean())
    return float(np.abs(predictions - targets).mean())


def rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Root Mean Squared Error."""
    if mask is not None:
        missing = mask < 0.5
        if missing.sum() == 0:
            return 0.0
        return float(np.sqrt(((predictions[missing] - targets[missing]) ** 2).mean()))
    return float(np.sqrt(((predictions - targets) ** 2).mean()))


def mre(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean Relative Error (normalized by true value magnitude)."""
    if mask is not None:
        missing = mask < 0.5
        if missing.sum() == 0:
            return 0.0
        pred_m = predictions[missing]
        tgt_m = targets[missing]
    else:
        pred_m = predictions.ravel()
        tgt_m = targets.ravel()

    denom = np.abs(tgt_m).sum()
    if denom < 1e-8:
        return 0.0
    return float(np.abs(pred_m - tgt_m).sum() / denom)


def mape(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error."""
    if mask is not None:
        missing = mask < 0.5
        if missing.sum() == 0:
            return 0.0
        pred_m = predictions[missing]
        tgt_m = targets[missing]
    else:
        pred_m = predictions.ravel()
        tgt_m = targets.ravel()

    return float(np.mean(np.abs((pred_m - tgt_m) / (np.abs(tgt_m) + epsilon))) * 100)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    metric_names: list = ("mae", "rmse", "mre"),
) -> Dict[str, float]:
    """
    Compute multiple imputation metrics.

    Args:
        predictions: (N, T, D) or (T, D) imputed values
        targets:     same shape, ground truth
        mask:        same shape, observation mask (1=observed, 0=missing)
        metric_names: which metrics to compute

    Returns:
        dict of metric_name → value
    """
    _metric_fns = {
        "mae": mae,
        "rmse": rmse,
        "mre": mre,
        "mape": mape,
    }

    results = {}
    for name in metric_names:
        if name in _metric_fns:
            results[name] = _metric_fns[name](predictions, targets, mask)
    return results
