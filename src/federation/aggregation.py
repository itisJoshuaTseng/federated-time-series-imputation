"""
Federated Aggregation Strategies.

Implements various federated aggregation algorithms:
  - FedAvg:  weighted averaging of model parameters
  - FedProx: FedAvg + proximal regularization (handled client-side)
  - FedAdam: server-side adaptive optimization (FedOpt family)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch


def fedavg(
    client_params_list: List[Dict[str, Dict[str, torch.Tensor]]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Federated Averaging (FedAvg).

    Performs weighted average of model parameters from multiple clients.
    McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.

    Args:
        client_params_list: list of parameter dicts from each client
            Each dict has structure: {"encoder": {...}, "agent_0": {...}, ...}
        weights: per-client weights (proportional to dataset size).
            If None, uniform weights.

    Returns:
        aggregated_params: averaged parameter dict with same structure
    """
    num_clients = len(client_params_list)

    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    # Initialize with zeros using first client's structure
    aggregated = deepcopy(client_params_list[0])
    for component_key in aggregated:
        for param_key in aggregated[component_key]:
            aggregated[component_key][param_key] = torch.zeros_like(
                aggregated[component_key][param_key]
            )

    # Weighted sum
    for client_idx, client_params in enumerate(client_params_list):
        w = weights[client_idx]
        for component_key in client_params:
            for param_key in client_params[component_key]:
                aggregated[component_key][param_key] += (
                    w * client_params[component_key][param_key]
                )

    return aggregated


def fedprox(
    client_params_list: List[Dict[str, Dict[str, torch.Tensor]]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    FedProx aggregation (same as FedAvg at server side).

    The proximal term is handled client-side during local training.
    Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
    """
    return fedavg(client_params_list, weights)


class FedAdamState:
    """Server-side optimizer state for FedAdam."""

    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        server_lr: float = 1e-1,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.server_lr = server_lr
        self.m = None  # first moment
        self.v = None  # second moment
        self.t = 0     # step counter


def fedadam(
    client_params_list: List[Dict[str, Dict[str, torch.Tensor]]],
    global_params: Dict[str, Dict[str, torch.Tensor]],
    state: FedAdamState,
    weights: Optional[List[float]] = None,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], FedAdamState]:
    """
    FedAdam: Server-side adaptive optimization.

    Reddi et al., "Adaptive Federated Optimization", ICLR 2021.

    Uses Adam-style momentum on the aggregated pseudo-gradients.

    Args:
        client_params_list: updated params from clients
        global_params: current global model params
        state: FedAdamState with momentum buffers
        weights: per-client weights

    Returns:
        new_global_params: updated global parameters
        state: updated optimizer state
    """
    # First compute the pseudo-gradient (FedAvg result - global)
    avg_params = fedavg(client_params_list, weights)

    state.t += 1

    # Initialize moments if needed
    if state.m is None:
        state.m = deepcopy(global_params)
        state.v = deepcopy(global_params)
        for comp in state.m:
            for key in state.m[comp]:
                state.m[comp][key] = torch.zeros_like(state.m[comp][key])
                state.v[comp][key] = torch.zeros_like(state.v[comp][key])

    new_params = deepcopy(global_params)

    for comp in avg_params:
        for key in avg_params[comp]:
            # Pseudo-gradient
            delta = avg_params[comp][key] - global_params[comp][key]

            # Update moments
            state.m[comp][key] = (
                state.beta1 * state.m[comp][key] + (1 - state.beta1) * delta
            )
            state.v[comp][key] = (
                state.beta2 * state.v[comp][key]
                + (1 - state.beta2) * delta ** 2
            )

            # Bias correction
            m_hat = state.m[comp][key] / (1 - state.beta1 ** state.t)
            v_hat = state.v[comp][key] / (1 - state.beta2 ** state.t)

            # Update global params
            new_params[comp][key] = (
                global_params[comp][key]
                + state.server_lr * m_hat / (torch.sqrt(v_hat) + state.tau)
            )

    return new_params, state


# ================================================================
# Complementarity-Aware Aggregation (Fed-SAITS-CA)
# ================================================================


def compute_complementarity_matrix(
    mechanism_coefs: List[np.ndarray],
) -> np.ndarray:
    """
    Compute pairwise complementarity between clients from their MNAR
    mechanism coefficient vectors (CAFÉ approach).

    complementarity(i, j) = (1 - cosine_similarity(xi_i, xi_j)) / 2

    Normalised to [0, 1]:
        0  →  identical mechanism (same direction, no complementarity)
        1  →  opposite mechanisms (e.g. MNAR-Left vs MNAR-Right, max complementarity)

    Using mechanism *coefficients* (from logistic regression on the missing
    mask) rather than simple missing *rates* ensures that directional MNAR
    differences (Left vs Right) produce non-zero complementarity, whereas
    missing-rate vectors from heterogeneous-magnitude-but-same-direction
    clients would be colinear and yield complementarity ≈ 0.

    Args:
        mechanism_coefs: list of (D,) arrays, per-client logistic-regression
            coefficients from get_mechanism_coefs().

    Returns:
        (N, N) complementarity matrix, values in [0, 1].
    """
    P = np.stack(mechanism_coefs)                          # (N, D)
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    P_normed = P / norms
    cos_sim = P_normed @ P_normed.T                        # (N, N), in [-1, 1]
    # Map to [0, 1]: same direction → 0, opposite direction → 1
    return (1.0 - cos_sim) / 2.0


def complementarity_aware_aggregation(
    client_params_list: List[Dict[str, Dict[str, torch.Tensor]]],
    mechanism_coefs: List[np.ndarray],
    tau: float = 1.0,
    gamma: float = 0.02,
    scale_factor: float = 4.0,
    alpha: float = 0.95,
    sample_sizes: Optional[List[float]] = None,
) -> List[Dict[str, Dict[str, torch.Tensor]]]:
    """
    Complementarity-Aware Aggregation for Fed-SAITS.

    Exact port of CAFÉ's ``fedmechw_new`` (Min et al., IEEE TKDE 2025) to
    the SAITS parameter space:

        mech_w_ij   = complementarity(i, j) + 1e-5               , j ≠ i
        size_w_j    = n_j / max(n)
        raw_w_ij    = (α · mech_w_ij + (1-α) · size_w_j) ^ scale_factor
        w_ij        = raw_w_ij / Σ_k raw_w_ik
        θ_i_others  = Σ_{j≠i} w_ij · θ_j
        θ_i'        = γ · θ_i  +  (1-γ) · θ_i_others

    Notes:
      * Uses CAFÉ's **power-law** weighting, not softmax.  For cos-sim-based
        complementarity ∈ [0, 1], raising to ``scale_factor=4`` sharpens the
        assignment toward the most complementary peers without needing a
        temperature parameter.
      * ``γ=0.02`` matches CAFÉ's default in
        ``conf/config_tmpl/imp_config_tmplate.yaml``.
      * ``alpha=0.95`` mixes in a small sample-size bias (5%) so that larger
        clients still pull slightly more weight — irrelevant when all
        clients have equal sample size, matters for heterogeneous splits.

    Args:
        client_params_list: list of parameter dicts, one per client.
            Each: {"saits": {"param_name": tensor, ...}}.
        mechanism_coefs: list of per-client fingerprint vectors, one entry
            per client (same length/order as ``client_params_list``).
        tau:          kept for CLI back-compat; only used when ``scale_factor``
                      is None (legacy softmax path).
        gamma:        self-preservation weight (CAFÉ default 0.02).
        scale_factor: power-law exponent on the raw weight (CAFÉ default 4).
        alpha:        mixing between mechanism-similarity and sample-size
                      weighting (CAFÉ default 0.95 → 95% mechanism, 5% size).
        sample_sizes: optional per-client sample sizes for the size term;
                      if None, falls back to uniform (no size bias).

    Returns:
        List of personalized aggregated parameter dicts, one per client
        (same order as the input list).
    """
    num_clients = len(client_params_list)
    assert len(mechanism_coefs) == num_clients

    # Complementarity matrix in [0, 1]
    comp_matrix = compute_complementarity_matrix(mechanism_coefs)

    # Normalised sample-size signal (broadcastable to length num_clients-1 later)
    if sample_sizes is None:
        sample_arr = np.ones(num_clients, dtype=float)
    else:
        sample_arr = np.asarray(sample_sizes, dtype=float)
    size_denom = float(sample_arr.max()) if sample_arr.max() > 0 else 1.0

    result = []
    for i in range(num_clients):
        other_idx = [j for j in range(num_clients) if j != i]
        mech_w = np.array(
            [comp_matrix[i, j] for j in other_idx]
        ) + 1e-5                                             # (num_clients-1,)
        size_w = sample_arr[other_idx] / size_denom          # (num_clients-1,)

        # CAFÉ power-law weighting
        raw_w = (alpha * mech_w + (1.0 - alpha) * size_w) ** scale_factor
        final_w = raw_w / raw_w.sum()

        # Weighted average of peer parameters
        other_avg = deepcopy(client_params_list[other_idx[0]])
        for ck in other_avg:
            for pk in other_avg[ck]:
                other_avg[ck][pk] = torch.zeros_like(other_avg[ck][pk])
        for w_idx, j in enumerate(other_idx):
            w = float(final_w[w_idx])
            for ck in client_params_list[j]:
                for pk in client_params_list[j][ck]:
                    other_avg[ck][pk] += (
                        w * client_params_list[j][ck][pk]
                    )

        # γ·self + (1-γ)·other_avg  (self-preservation)
        blended = deepcopy(client_params_list[i])
        for ck in blended:
            for pk in blended[ck]:
                blended[ck][pk] = (
                    gamma * client_params_list[i][ck][pk]
                    + (1.0 - gamma) * other_avg[ck][pk]
                )

        result.append(blended)

    return result
