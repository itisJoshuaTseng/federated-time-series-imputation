"""
Actor-Critic Neural Networks for MAPPO.

- Actor: per-agent policy network (outputs imputation actions)
- Critic: centralized value network (CTDE paradigm)
- CommunicationNet: inter-agent message passing
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """
    Per-agent actor (policy) network.

    Input:  agent's local observation + communication messages
    Output: continuous imputation actions (Gaussian policy)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
        comm_dim: int = 0,
        activation: str = "relu",
        log_std_range: Tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.comm_dim = comm_dim
        self.log_std_min, self.log_std_max = log_std_range

        # Build MLP
        input_dim = obs_dim + comm_dim
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                self._get_activation(activation),
            ])
            input_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Gaussian policy heads
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        # Policy output: small init for exploration
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "elu": nn.ELU()}[name]

    def forward(
        self,
        obs: torch.Tensor,
        comm_msg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:      (B, obs_dim) local observation
            comm_msg: (B, comm_dim) optional communication message

        Returns:
            mean:    (B, action_dim) action mean
            log_std: (B, action_dim) action log-std (clamped)
        """
        if comm_msg is not None:
            x = torch.cat([obs, comm_msg], dim=-1)
        elif self.comm_dim > 0:
            # Pad with zeros when communication is expected but no message provided
            zeros = torch.zeros(obs.shape[0], self.comm_dim, device=obs.device)
            x = torch.cat([obs, zeros], dim=-1)
        else:
            x = obs

        features = self.backbone(x)
        mean = torch.tanh(self.mean_head(features))  # bound to [-1, 1]
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_distribution(
        self,
        obs: torch.Tensor,
        comm_msg: Optional[torch.Tensor] = None,
    ) -> Normal:
        """Get Gaussian action distribution."""
        mean, log_std = self.forward(obs, comm_msg)
        return Normal(mean, log_std.exp())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        comm_msg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probability and entropy of given actions.

        Returns:
            log_prob: (B,) log probability
            entropy:  (B,) entropy
        """
        dist = self.get_distribution(obs, comm_msg)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # sum over action dims
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralized critic (value function) for CTDE.

    In MAPPO, the critic has access to global state information
    (all agents' observations + global features).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = "relu",
    ):
        super().__init__()

        layers = []
        input_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                ActorNetwork._get_activation(activation),
            ])
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, 1))  # single value output
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) global state

        Returns:
            value: (B, 1) estimated value
        """
        return self.network(state)


class CommunicationNetwork(nn.Module):
    """
    Inter-agent communication module.

    Implements a simple message-passing mechanism where agents
    exchange embeddings to coordinate imputation strategies.

    Uses attention-based aggregation of messages from other agents.
    """

    def __init__(
        self,
        obs_dim: int,
        comm_dim: int = 32,
        num_agents: int = 6,
        num_rounds: int = 2,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.comm_dim = comm_dim
        self.num_rounds = num_rounds

        # Message encoder: obs → message
        self.message_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, comm_dim),
        )

        # Message aggregation (attention-based)
        self.query_proj = nn.Linear(comm_dim, comm_dim)
        self.key_proj = nn.Linear(comm_dim, comm_dim)
        self.value_proj = nn.Linear(comm_dim, comm_dim)

        # Message update (GRU for iterative refinement)
        self.gru = nn.GRUCell(comm_dim, comm_dim)

    def forward(
        self,
        observations: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Multi-round message passing between agents.

        Args:
            observations: list of (B, obs_dim) tensors, one per agent

        Returns:
            messages: list of (B, comm_dim) aggregated messages per agent
        """
        B = observations[0].shape[0]
        device = observations[0].device

        # Initialize messages from observations
        messages = [
            self.message_encoder(obs) for obs in observations
        ]  # list of (B, comm_dim)

        for _ in range(self.num_rounds):
            # Stack all messages: (B, num_agents, comm_dim)
            msg_stack = torch.stack(messages, dim=1)

            new_messages = []
            for i in range(self.num_agents):
                # Attention: agent i attends to all other agents
                query = self.query_proj(messages[i]).unsqueeze(1)    # (B, 1, comm_dim)
                keys = self.key_proj(msg_stack)                       # (B, N, comm_dim)
                values = self.value_proj(msg_stack)                   # (B, N, comm_dim)

                # Scaled dot-product attention
                scale = self.comm_dim ** 0.5
                attn_weights = torch.bmm(query, keys.transpose(1, 2)) / scale  # (B, 1, N)
                attn_weights = F.softmax(attn_weights, dim=-1)
                aggregated = torch.bmm(attn_weights, values).squeeze(1)  # (B, comm_dim)

                # Update message via GRU
                new_msg = self.gru(aggregated, messages[i])
                new_messages.append(new_msg)

            messages = new_messages

        return messages
