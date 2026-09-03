"""Actor-Critic Neural Network Architecture with Orthogonal Initialization and Pre-Softmax Masking for TFT RL."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tft_ai_player.rl.models.distributions import MaskedCategorical
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS


def layer_init(layer: nn.Linear, gain: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize linear layer with orthogonal weights and constant bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ShopBenchFeatureExtractor(nn.Module):
    """Learnable projection modules for shop and bench tokens.

    - Shop (160D -> 64D): Linear(160, 64) -> LayerNorm(64) -> ReLU()
    - Bench DeepSets (64D -> 64D): Linear(64, 64) -> LayerNorm(64) -> ReLU()
    """

    def __init__(self, shop_in: int = 160, bench_in: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.shop_proj = nn.Sequential(
            layer_init(nn.Linear(shop_in, out_dim)),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self.bench_proj = nn.Sequential(
            layer_init(nn.Linear(bench_in, out_dim)),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, shop_tokens: torch.Tensor, bench_pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shop_feat = self.shop_proj(shop_tokens)
        bench_feat = self.bench_proj(bench_pooled)
        return shop_feat, bench_feat


class TFTActorCritic(nn.Module):
    """PPO Actor-Critic Network for Teamfight Tactics (|A|=111, Obs=704D).

    Architecture:
      Input: State observation o_t in R^{704}
      Trunk:
        Linear(704, 512) -> LayerNorm(512) -> GELU()
        Linear(512, 512) -> LayerNorm(512) -> GELU()
      Actor Head:
        Linear(512, 256) -> GELU()
        Linear(256, 111) -> Action Logits -> MaskedCategorical
      Critic Head:
        Linear(512, 256) -> GELU()
        Linear(256, 1) -> Value Estimate V(s)
    """

    def __init__(
        self,
        obs_dim: int = 704,
        action_dim: int = TOTAL_DISCRETE_ACTIONS,
        hidden_dim: int = 512,
        branch_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared representation trunk
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Actor head (policy pi_theta)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, branch_dim), gain=math.sqrt(2)),
            nn.GELU(),
            layer_init(nn.Linear(branch_dim, action_dim), gain=0.01),
        )

        # Critic head (value function V_phi)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, branch_dim), gain=math.sqrt(2)),
            nn.GELU(),
            layer_init(nn.Linear(branch_dim, 1), gain=1.0),
        )

    def forward(
        self,
        obs: torch.Tensor | np.ndarray | dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through trunk, actor, and critic.

        Args:
            obs: (batch_size, 704) observation tensor or array.

        Returns:
            logits: (batch_size, 111) raw action logits
            values: (batch_size, 1) state value estimates
        """
        if isinstance(obs, np.ndarray):
            device = next(self.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        elif isinstance(obs, dict):
            # Compatibility fallback if dict was passed
            device = next(self.parameters()).device
            if "obs" in obs:
                obs_t = torch.as_tensor(obs["obs"], dtype=torch.float32, device=device)
            elif "state" in obs:
                obs_t = torch.as_tensor(obs["state"], dtype=torch.float32, device=device)
            else:
                # Concatenate available components
                comps = [torch.as_tensor(v, dtype=torch.float32, device=device).view(v.shape[0], -1) for k, v in obs.items() if k != "action_mask"]
                obs_t = torch.cat(comps, dim=-1)
        else:
            obs_t = obs.float()

        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        shared_feat = self.trunk(obs_t)
        logits = self.actor(shared_feat)
        values = self.critic(shared_feat)
        return logits, values

    def get_value(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute state value estimate V(s)."""
        _, values = self.forward(obs)
        return values.squeeze(-1)

    def get_action(
        self,
        obs: torch.Tensor | np.ndarray,
        action_mask: torch.Tensor | np.ndarray,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action under invalid action mask.

        Returns:
            action: (batch_size,) sampled discrete action ID (0..110)
            log_prob: (batch_size,) log probability of sampled action
            value: (batch_size,) state value estimate
        """
        device = next(self.parameters()).device
        if isinstance(action_mask, np.ndarray):
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        else:
            mask_t = action_mask.to(device).bool()

        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)

        logits, values = self.forward(obs)
        dist = MaskedCategorical(logits=logits, mask=mask_t)
        action, log_prob = dist.sample_action(deterministic=deterministic)

        return action.squeeze(-1), log_prob.squeeze(-1), values.squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO gradient updates.

        Returns:
            log_probs: (batch_size,) log pi(a | s)
            entropy: (batch_size,) policy entropy
            values: (batch_size,) V(s)
        """
        logits, values = self.forward(obs)
        dist = MaskedCategorical(logits=logits, mask=action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)
