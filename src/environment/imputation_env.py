"""
Imputation Environment (Gymnasium).

Models time series imputation as a Markov Decision Process for
multi-agent reinforcement learning.

MDP formulation:
  State:   [observed_values, missing_mask, temporal_encoding, previous_imputation]
  Action:  continuous imputation values for assigned features
  Reward:  negative imputation error + temporal consistency bonus
  Done:    after all time steps are processed
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import gymnasium as gym
import numpy as np
import torch


class ImputationEnv(gym.Env):
    """
    Multi-Agent Time Series Imputation Environment.

    Each agent is responsible for imputing a group of features.
    The environment steps through time, and at each step agents
    produce imputation values for their assigned missing entries.

    Args:
        data:           (T, D) observed values (0 where missing)
        ground_truth:   (T, D) true values (for reward computation)
        mask:           (T, D) binary mask (1=observed, 0=missing)
        feature_groups: list of lists, each sublist = feature indices for an agent
        config:         reward & environment configuration
    """

    metadata = {"render_modes": ["none"]}

    def __init__(
        self,
        data: np.ndarray,
        ground_truth: np.ndarray,
        mask: np.ndarray,
        feature_groups: List[List[int]],
        config: Optional[dict] = None,
    ):
        super().__init__()

        self.data = data.copy()                     # (T, D)
        self.ground_truth = ground_truth.copy()     # (T, D)
        self.mask = mask.copy()                     # (T, D)
        self.seq_length, self.num_features = data.shape
        self.feature_groups = feature_groups
        self.num_agents = len(feature_groups)

        # Configuration
        cfg = config or {}
        reward_cfg = cfg.get("reward", {})
        self.temporal_weight = reward_cfg.get("temporal_consistency_weight", 0.1)
        self.cross_var_weight = reward_cfg.get("cross_variable_weight", 0.05)
        self.primary_metric = reward_cfg.get("primary_metric", "mae")

        # Current imputation canvas (filled in over time)
        self.canvas = np.zeros_like(data)
        self.current_step = 0

        # ---- Spaces (per-agent) ----
        # Observation: observed_values + mask + temporal_enc + prev_imputation
        # For each agent: features in its group × 4 channels
        max_group_size = max(len(g) for g in feature_groups)

        # Global observation includes all features for context
        obs_dim = self.num_features * 4 + 2  # 4 channels + step + progress
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: imputed values for the agent's assigned features
        self.action_spaces = [
            gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(len(g),),
                dtype=np.float32,
            )
            for g in feature_groups
        ]

        self._episode_rewards = []

    # ----------------------------------------------------------------
    # Gym Interface
    # ----------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.canvas = self.data.copy()  # start with observed values
        self._episode_rewards = []

        obs = self._get_observations()
        info = {"step": 0, "num_missing": int((self.mask < 0.5).sum())}
        return obs, info

    def step(
        self,
        actions: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """
        Execute one step: agents provide imputation values for current time step.

        Args:
            actions: list of arrays, one per agent (imputed values for its features)

        Returns:
            observations: list of obs arrays (one per agent)
            rewards:      list of reward floats
            terminated:   whether episode is done
            truncated:    always False
            info:         diagnostic info
        """
        t = self.current_step

        # --- Apply actions to canvas ---
        for agent_id, action in enumerate(actions):
            features = self.feature_groups[agent_id]
            for i, f_idx in enumerate(features):
                if self.mask[t, f_idx] < 0.5:  # only impute missing entries
                    # Map from [-1, 1] → [0, 1] (normalized data range)
                    imputed_val = (action[i] + 1.0) / 2.0
                    self.canvas[t, f_idx] = np.clip(imputed_val, 0, 1)

        # --- Compute rewards ---
        rewards = self._compute_rewards(t, actions)
        self._episode_rewards.append(np.mean(rewards))

        # --- Advance time step ---
        self.current_step += 1
        terminated = (self.current_step >= self.seq_length)

        # --- Get next observations ---
        obs = self._get_observations()

        info = {
            "step": self.current_step,
            "step_mae": self._step_mae(t),
            "cumulative_mae": self._cumulative_mae(),
        }

        if terminated:
            info["episode_reward"] = float(np.sum(self._episode_rewards))
            info["final_mae"] = self._cumulative_mae()

        return obs, rewards, terminated, False, info

    # ----------------------------------------------------------------
    # Observations
    # ----------------------------------------------------------------

    def _get_observations(self) -> List[np.ndarray]:
        """
        Construct observation for each agent at current time step.

        Observation channels (for current time step t):
          1. observed values (all D features)
          2. missing mask (all D features)
          3. temporal encoding (sin/cos of normalized time)
          4. current canvas values (previous imputation)
          + scalar: normalized step, progress fraction
        """
        t = min(self.current_step, self.seq_length - 1)

        observed_vals = self.data[t]           # (D,)
        mask_vals = self.mask[t]               # (D,)
        canvas_vals = self.canvas[t]           # (D,)

        # Temporal encoding
        progress = t / max(self.seq_length - 1, 1)
        time_enc = np.array([np.sin(2 * np.pi * progress), np.cos(2 * np.pi * progress)])

        # Context from previous time steps (delta encoding)
        if t > 0:
            prev_canvas = self.canvas[t - 1]
        else:
            prev_canvas = np.zeros(self.num_features)

        # Global observation (shared context)
        global_obs = np.concatenate([
            observed_vals,   # (D,)
            mask_vals,       # (D,)
            canvas_vals,     # (D,)
            prev_canvas,     # (D,)
            time_enc,        # (2,)
        ]).astype(np.float32)

        # Each agent sees the same global observation
        # (agent-specific features are handled by the policy network)
        observations = [global_obs.copy() for _ in range(self.num_agents)]

        return observations

    # ----------------------------------------------------------------
    # Reward Computation
    # ----------------------------------------------------------------

    def _compute_rewards(
        self,
        t: int,
        actions: List[np.ndarray],
    ) -> List[float]:
        """
        Compute per-agent rewards at time step t.

        Reward components:
          1. Imputation accuracy (negative MAE/RMSE for agent's features)
          2. Temporal consistency (smooth transitions between t-1 and t)
          3. Cross-variable consistency (correlation preservation)
        """
        rewards = []

        for agent_id, action in enumerate(actions):
            features = self.feature_groups[agent_id]
            reward = 0.0

            # --- Accuracy reward ---
            for i, f_idx in enumerate(features):
                if self.mask[t, f_idx] < 0.5:  # only for missing entries
                    imputed = self.canvas[t, f_idx]
                    true_val = self.ground_truth[t, f_idx]

                    if self.primary_metric == "mae":
                        error = abs(imputed - true_val)
                    elif self.primary_metric == "rmse":
                        error = (imputed - true_val) ** 2
                    else:
                        error = abs(imputed - true_val)

                    reward -= error

            # --- Temporal consistency ---
            if t > 0 and self.temporal_weight > 0:
                for f_idx in features:
                    delta = abs(self.canvas[t, f_idx] - self.canvas[t - 1, f_idx])
                    reward -= self.temporal_weight * delta

            # --- Cross-variable consistency ---
            if self.cross_var_weight > 0:
                all_imputed = self.canvas[t]
                all_true = self.ground_truth[t]
                # Penalize if imputed correlation pattern differs from ground truth
                cross_err = np.mean(np.abs(all_imputed - all_true))
                reward -= self.cross_var_weight * cross_err

            rewards.append(float(reward))

        return rewards

    # ----------------------------------------------------------------
    # Metrics Helpers
    # ----------------------------------------------------------------

    def _step_mae(self, t: int) -> float:
        """MAE for imputed entries at time step t."""
        missing = self.mask[t] < 0.5
        if missing.sum() == 0:
            return 0.0
        return float(np.abs(self.canvas[t][missing] - self.ground_truth[t][missing]).mean())

    def _cumulative_mae(self) -> float:
        """MAE over all imputed entries so far."""
        t = self.current_step
        region = slice(0, t)
        missing = self.mask[region] < 0.5
        if missing.sum() == 0:
            return 0.0
        return float(np.abs(
            self.canvas[region][missing] - self.ground_truth[region][missing]
        ).mean())

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    def get_imputed_series(self) -> np.ndarray:
        """Return the full imputed time series canvas."""
        return self.canvas.copy()

    @staticmethod
    def create_feature_groups(
        num_features: int,
        num_agents: int,
    ) -> List[List[int]]:
        """
        Evenly divide features among agents.

        Example: 37 features, 6 agents → groups of [7, 7, 7, 6, 5, 5]
        """
        base = num_features // num_agents
        remainder = num_features % num_agents

        groups = []
        idx = 0
        for a in range(num_agents):
            size = base + (1 if a < remainder else 0)
            groups.append(list(range(idx, idx + size)))
            idx += size
        return groups
