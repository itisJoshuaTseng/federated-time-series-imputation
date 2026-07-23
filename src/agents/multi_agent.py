"""
Multi-Agent System Coordinator.

Manages all MAPPO agents, the communication network, and
the shared imputation transformer backbone. Orchestrates
the interaction between agents and the imputation environment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from .mappo_agent import MAPPOAgent
from .networks import CommunicationNetwork
from .replay_buffer import RolloutBuffer
from ..models.imputation_model import ImputationTransformer
from ..environment.imputation_env import ImputationEnv


class MultiAgentSystem:
    """
    Multi-Agent RL System for Time Series Imputation.

    Architecture (CTDE — Centralized Training, Decentralized Execution):
      - Shared Transformer encoder extracts temporal representations
      - Each agent has a private actor for its feature group
      - Shared centralized critic for value estimation
      - Optional communication network for inter-agent coordination

    Args:
        config: full configuration dictionary
        num_features: number of clinical variables
        feature_groups: per-agent feature index assignment
        device: torch device
    """

    def __init__(
        self,
        config: dict,
        num_features: int,
        feature_groups: List[List[int]],
        device: str = "cpu",
    ):
        self.config = config
        self.num_features = num_features
        self.feature_groups = feature_groups
        self.num_agents = len(feature_groups)
        self.device = device

        marl_cfg = config.get("marl", {})
        net_cfg = config.get("networks", {})
        train_cfg = config.get("training", {})

        # --- Shared Transformer Encoder ---
        enc_cfg = net_cfg.get("encoder", {})
        self.encoder = ImputationTransformer(
            num_features=num_features,
            d_model=enc_cfg.get("d_model", 128),
            nhead=enc_cfg.get("nhead", 4),
            num_layers=enc_cfg.get("num_layers", 3),
            dim_feedforward=enc_cfg.get("dim_feedforward", 256),
            dropout=enc_cfg.get("dropout", 0.1),
        ).to(device)

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=train_cfg.get("lr_encoder", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # Observation dim: env obs + encoder output
        d_model = enc_cfg.get("d_model", 128)
        obs_dim = num_features * 4 + 2  # from ImputationEnv
        self.obs_dim = obs_dim

        # Global state dim for centralized critic
        global_state_dim = obs_dim * self.num_agents

        # --- Communication Network ---
        comm_cfg = marl_cfg.get("communication", {})
        self.comm_enabled = comm_cfg.get("enabled", True)
        self.comm_dim = comm_cfg.get("comm_dim", 32) if self.comm_enabled else 0

        if self.comm_enabled:
            self.comm_net = CommunicationNetwork(
                obs_dim=obs_dim,
                comm_dim=self.comm_dim,
                num_agents=self.num_agents,
                num_rounds=comm_cfg.get("num_comm_rounds", 2),
            ).to(device)

            self.comm_optimizer = optim.Adam(
                self.comm_net.parameters(),
                lr=train_cfg.get("lr_actor", 3e-4),
            )
        else:
            self.comm_net = None
            self.comm_optimizer = None

        # --- Create Agents ---
        self.agents: List[MAPPOAgent] = []
        for i in range(self.num_agents):
            agent = MAPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=len(feature_groups[i]),
                global_state_dim=global_state_dim,
                config=config,
                device=device,
            )
            self.agents.append(agent)

        # --- Rollout Buffer ---
        rollout_steps = marl_cfg.get("num_rollout_steps", 64)
        self.buffer = RolloutBuffer(
            buffer_size=rollout_steps,
            num_agents=self.num_agents,
            obs_dim=obs_dim,
            action_dims=[len(g) for g in feature_groups],
            gamma=marl_cfg.get("gamma", 0.99),
            gae_lambda=marl_cfg.get("gae_lambda", 0.95),
            comm_dim=self.comm_dim,
            device=device,
        )

    # ----------------------------------------------------------------
    # Interaction with Environment
    # ----------------------------------------------------------------

    def collect_rollout(
        self,
        env: ImputationEnv,
        num_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Collect a rollout of experience from the environment.

        Args:
            env: ImputationEnv instance (single sample)
            num_steps: override for rollout length

        Returns:
            info: dict with rollout statistics
        """
        steps = num_steps or self.config["marl"].get("num_rollout_steps", 64)
        self.buffer.reset()

        observations, info = env.reset()
        episode_rewards = []
        total_steps = 0

        for step in range(steps):
            # --- Communication phase ---
            comm_messages = self._communicate(observations)

            # --- Action selection ---
            actions = []
            log_probs = []
            values = []

            # Build global state for centralized critic
            global_state = np.concatenate(observations)

            for i, agent in enumerate(self.agents):
                comm_msg = comm_messages[i] if comm_messages else None
                action, log_prob, _ = agent.select_action(
                    observations[i], comm_msg, deterministic=False
                )
                value = agent.get_value(global_state)

                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            # --- Environment step ---
            next_observations, rewards, terminated, truncated, step_info = env.step(actions)
            done = terminated or truncated

            # --- Store transition ---
            comm_np = (
                [m.cpu().numpy() if isinstance(m, torch.Tensor) else m
                 for m in comm_messages]
                if comm_messages else None
            )

            self.buffer.add(
                observations=observations,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                done=done,
                global_state=global_state,
                comm_messages=comm_np,
            )

            episode_rewards.append(np.mean(rewards))
            observations = next_observations
            total_steps += 1

            if done:
                observations, info = env.reset()

        # Compute returns and advantages
        global_state = np.concatenate(observations)
        last_values = [
            agent.get_value(global_state) for agent in self.agents
        ]
        self.buffer.compute_returns_and_advantages(last_values, done)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "total_steps": total_steps,
            "step_mae": step_info.get("step_mae", 0.0),
        }

    def _communicate(
        self,
        observations: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        """Run inter-agent communication if enabled."""
        if not self.comm_enabled or self.comm_net is None:
            return None

        with torch.no_grad():
            obs_tensors = [
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                for obs in observations
            ]
            messages = self.comm_net(obs_tensors)
            return [msg.squeeze(0).cpu().numpy() for msg in messages]

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    def train_step(self) -> Dict[str, float]:
        """
        Perform one PPO update for all agents using collected rollout.

        Returns:
            losses: aggregated loss metrics
        """
        all_losses = {"actor_loss": 0, "critic_loss": 0, "entropy": 0}

        for agent in self.agents:
            agent_losses = agent.update(self.buffer)
            for k, v in agent_losses.items():
                all_losses[k] += v

        # Average across agents
        for k in all_losses:
            all_losses[k] /= self.num_agents

        return all_losses

    def train_episode(
        self,
        env: ImputationEnv,
    ) -> Dict[str, float]:
        """
        Full training episode: collect rollout + update.

        Args:
            env: ImputationEnv instance

        Returns:
            metrics: combined rollout and training metrics
        """
        rollout_info = self.collect_rollout(env)
        train_losses = self.train_step()

        return {**rollout_info, **train_losses}

    # ----------------------------------------------------------------
    # Federation Interface
    # ----------------------------------------------------------------

    def get_all_parameters(self) -> Dict[str, Dict]:
        """Get all model parameters for federated upload."""
        params = {
            "encoder": {
                k: v.cpu().clone()
                for k, v in self.encoder.state_dict().items()
            },
        }

        for i, agent in enumerate(self.agents):
            params[f"agent_{i}"] = agent.get_parameters()

        if self.comm_net is not None:
            params["comm_net"] = {
                k: v.cpu().clone()
                for k, v in self.comm_net.state_dict().items()
            }

        return params

    def set_all_parameters(self, params: Dict[str, Dict]):
        """Set all model parameters from federated download."""
        if "encoder" in params:
            self.encoder.load_state_dict(params["encoder"])

        for i, agent in enumerate(self.agents):
            key = f"agent_{i}"
            if key in params:
                agent.set_parameters(params[key])

        if self.comm_net is not None and "comm_net" in params:
            self.comm_net.load_state_dict(params["comm_net"])

    def eval_mode(self):
        """Set all networks to eval mode."""
        self.encoder.eval()
        for agent in self.agents:
            agent.actor.eval()
            agent.critic.eval()
        if self.comm_net:
            self.comm_net.eval()

    def train_mode(self):
        """Set all networks to train mode."""
        self.encoder.train()
        for agent in self.agents:
            agent.actor.train()
            agent.critic.train()
        if self.comm_net:
            self.comm_net.train()
