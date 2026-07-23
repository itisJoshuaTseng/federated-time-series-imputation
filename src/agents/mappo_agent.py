"""
MAPPO (Multi-Agent Proximal Policy Optimization) Agent.

Implements the per-agent PPO update with clipped surrogate objective
and centralized value function (CTDE paradigm).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import ActorNetwork, CriticNetwork
from .replay_buffer import RolloutBuffer


class MAPPOAgent:
    """
    Single agent in the MAPPO system.

    Each agent has:
      - A private actor (policy) network for its feature group
      - Access to a shared centralized critic

    The actor produces continuous imputation values.
    The critic estimates state values from global information.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        config: dict,
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        marl_cfg = config.get("marl", {})
        net_cfg = config.get("networks", {})
        train_cfg = config.get("training", {})

        # Communication
        comm_cfg = marl_cfg.get("communication", {})
        self.comm_enabled = comm_cfg.get("enabled", True)
        self.comm_dim = comm_cfg.get("comm_dim", 32) if self.comm_enabled else 0

        # --- Actor network ---
        actor_cfg = net_cfg.get("actor", {})
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_cfg.get("hidden_dims", [256, 128]),
            comm_dim=self.comm_dim,
            activation=actor_cfg.get("activation", "relu"),
        ).to(device)

        # --- Critic network (centralized) ---
        critic_cfg = net_cfg.get("critic", {})
        self.critic = CriticNetwork(
            state_dim=global_state_dim,
            hidden_dims=critic_cfg.get("hidden_dims", [256, 128]),
            activation=critic_cfg.get("activation", "relu"),
        ).to(device)

        # --- Optimizers ---
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=train_cfg.get("lr_actor", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=train_cfg.get("lr_critic", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # --- PPO Hyperparameters ---
        self.clip_epsilon = marl_cfg.get("clip_epsilon", 0.2)
        self.entropy_coef = marl_cfg.get("entropy_coef", 0.01)
        self.value_loss_coef = marl_cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm = marl_cfg.get("max_grad_norm", 0.5)
        self.ppo_epochs = marl_cfg.get("ppo_epochs", 4)
        self.num_mini_batches = marl_cfg.get("num_mini_batches", 4)

    # ----------------------------------------------------------------
    # Action Selection
    # ----------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        comm_msg: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action from policy.

        Returns:
            action:   (action_dim,) imputation values
            log_prob: scalar log-probability
            value:    scalar value estimate
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        comm_t = (
            torch.FloatTensor(comm_msg).unsqueeze(0).to(self.device)
            if comm_msg is not None else None
        )

        dist = self.actor.get_distribution(obs_t, comm_t)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        # Clamp to valid range
        action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            0.0,  # value computed separately via critic
        )

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        """Estimate value from global state."""
        state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        value = self.critic(state_t)
        return value.item()

    # ----------------------------------------------------------------
    # PPO Update
    # ----------------------------------------------------------------

    def update(
        self,
        buffer: RolloutBuffer,
    ) -> Dict[str, float]:
        """
        Perform PPO update using data from the rollout buffer.

        Returns:
            losses: dict with actor_loss, critic_loss, entropy, etc.
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in buffer.get_batches(
                num_mini_batches=self.num_mini_batches,
                agent_id=self.agent_id,
            ):
                obs = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                global_states = batch["global_states"]
                comm_msg = batch.get("comm_messages", None)

                # --- Actor loss (PPO clipped objective) ---
                new_log_probs, entropy = self.actor.evaluate_actions(
                    obs, actions, comm_msg
                )

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # --- Critic loss ---
                values = self.critic(global_states).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, returns)

                # --- Total loss ---
                loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                # --- Gradient step ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += (-entropy_loss.item())
                num_updates += 1

        return {
            "actor_loss": total_actor_loss / max(num_updates, 1),
            "critic_loss": total_critic_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    # ----------------------------------------------------------------
    # Model Parameter Access (for Federation)
    # ----------------------------------------------------------------

    def get_parameters(self) -> Dict[str, Dict]:
        """Get all model parameters (for federated aggregation)."""
        return {
            "actor": {k: v.cpu().clone() for k, v in self.actor.state_dict().items()},
            "critic": {k: v.cpu().clone() for k, v in self.critic.state_dict().items()},
        }

    def set_parameters(self, params: Dict[str, Dict]):
        """Set model parameters (from federated server)."""
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])

    def get_actor_parameters(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.actor.state_dict().items()}

    def set_actor_parameters(self, params: Dict[str, torch.Tensor]):
        self.actor.load_state_dict(params)

    def get_critic_parameters(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.critic.state_dict().items()}

    def set_critic_parameters(self, params: Dict[str, torch.Tensor]):
        self.critic.load_state_dict(params)
