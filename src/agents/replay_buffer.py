"""
Rollout Buffer for MAPPO.

Stores experience tuples from environment interactions,
computes GAE advantages, and provides mini-batch iteration.
"""

from __future__ import annotations

from typing import Dict, Generator, Optional

import numpy as np
import torch


class RolloutBuffer:
    """
    On-policy rollout buffer for multi-agent PPO.

    Stores transitions: (obs, action, reward, value, log_prob, done, comm_msg)
    for each agent, and computes GAE returns/advantages.
    """

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        action_dims: list,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        comm_dim: int = 0,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.comm_dim = comm_dim

        self._ptr = 0
        self._full = False

        # Pre-allocate buffers per agent
        self.observations = [
            np.zeros((buffer_size, obs_dim), dtype=np.float32)
            for _ in range(num_agents)
        ]
        self.actions = [
            np.zeros((buffer_size, action_dims[i]), dtype=np.float32)
            for i in range(num_agents)
        ]
        self.rewards = [
            np.zeros(buffer_size, dtype=np.float32)
            for _ in range(num_agents)
        ]
        self.values = [
            np.zeros(buffer_size, dtype=np.float32)
            for _ in range(num_agents)
        ]
        self.log_probs = [
            np.zeros(buffer_size, dtype=np.float32)
            for _ in range(num_agents)
        ]
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Global state for centralized critic
        self.global_states = np.zeros(
            (buffer_size, obs_dim * num_agents), dtype=np.float32
        )

        # Communication messages (optional)
        if comm_dim > 0:
            self.comm_messages = [
                np.zeros((buffer_size, comm_dim), dtype=np.float32)
                for _ in range(num_agents)
            ]
        else:
            self.comm_messages = None

        # Computed after rollout
        self.advantages = [
            np.zeros(buffer_size, dtype=np.float32)
            for _ in range(num_agents)
        ]
        self.returns = [
            np.zeros(buffer_size, dtype=np.float32)
            for _ in range(num_agents)
        ]

    @property
    def size(self) -> int:
        return self.buffer_size if self._full else self._ptr

    def add(
        self,
        observations: list,
        actions: list,
        rewards: list,
        values: list,
        log_probs: list,
        done: bool,
        global_state: np.ndarray,
        comm_messages: Optional[list] = None,
    ):
        """Add a single transition for all agents."""
        idx = self._ptr

        for i in range(self.num_agents):
            self.observations[i][idx] = observations[i]
            self.actions[i][idx] = actions[i]
            self.rewards[i][idx] = rewards[i]
            self.values[i][idx] = values[i]
            self.log_probs[i][idx] = log_probs[i]

        self.dones[idx] = float(done)
        self.global_states[idx] = global_state

        if self.comm_messages is not None and comm_messages is not None:
            for i in range(self.num_agents):
                self.comm_messages[i][idx] = comm_messages[i]

        self._ptr += 1
        if self._ptr >= self.buffer_size:
            self._full = True
            self._ptr = 0

    def compute_returns_and_advantages(
        self,
        last_values: list,
        last_done: bool,
    ):
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_values: list of value estimates at the step after buffer ends
            last_done:   whether the episode terminated
        """
        n = self.size

        for agent_id in range(self.num_agents):
            last_gae = 0.0
            for t in reversed(range(n)):
                if t == n - 1:
                    next_non_terminal = 1.0 - float(last_done)
                    next_value = last_values[agent_id]
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_value = self.values[agent_id][t + 1]

                delta = (
                    self.rewards[agent_id][t]
                    + self.gamma * next_value * next_non_terminal
                    - self.values[agent_id][t]
                )
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                self.advantages[agent_id][t] = last_gae

            self.returns[agent_id][:n] = (
                self.advantages[agent_id][:n] + self.values[agent_id][:n]
            )

    def get_batches(
        self,
        num_mini_batches: int = 4,
        agent_id: int = 0,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield mini-batches for a specific agent.

        Args:
            num_mini_batches: number of mini-batches to split data into
            agent_id: which agent's data to yield
        """
        n = self.size
        batch_size = n // num_mini_batches
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch = {
                "observations": torch.FloatTensor(
                    self.observations[agent_id][idx]
                ).to(self.device),
                "actions": torch.FloatTensor(
                    self.actions[agent_id][idx]
                ).to(self.device),
                "old_log_probs": torch.FloatTensor(
                    self.log_probs[agent_id][idx]
                ).to(self.device),
                "advantages": torch.FloatTensor(
                    self.advantages[agent_id][idx]
                ).to(self.device),
                "returns": torch.FloatTensor(
                    self.returns[agent_id][idx]
                ).to(self.device),
                "global_states": torch.FloatTensor(
                    self.global_states[idx]
                ).to(self.device),
            }

            # Normalize advantages
            adv = batch["advantages"]
            batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

            if self.comm_messages is not None:
                batch["comm_messages"] = torch.FloatTensor(
                    self.comm_messages[agent_id][idx]
                ).to(self.device)

            yield batch

    def reset(self):
        """Clear the buffer."""
        self._ptr = 0
        self._full = False
