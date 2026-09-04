"""Maskable Proximal Policy Optimization (PPO) with Generalized Advantage Estimation and Health Telemetry."""

from __future__ import annotations

import math
from typing import Generator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS


class RolloutBuffer:
    """Fixed-capacity rollout buffer storing transitions for on-policy PPO updates."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int = 704,
        action_dim: int = TOTAL_DISCRETE_ACTIONS,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | None = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")

        # Tensors
        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size,), dtype=torch.long, device=self.device)
        self.action_masks = torch.zeros((buffer_size, action_dim), dtype=torch.bool, device=self.device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=self.device)

        self.advantages = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)

        self.ptr: int = 0
        self.full: bool = False

    def reset(self) -> None:
        """Reset buffer pointer."""
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        action: int,
        action_mask: np.ndarray | torch.Tensor,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store a single transition step in buffer."""
        if self.ptr >= self.buffer_size:
            self.full = True
            return

        self.observations[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = int(action)
        self.action_masks[self.ptr] = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = bool(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float = 0.0, done: bool = True) -> None:
        """Compute Generalized Advantage Estimation (GAE-lambda) and discounted returns."""
        last_val = torch.tensor(last_value, dtype=torch.float32, device=self.device)
        last_done = torch.tensor(done, dtype=torch.bool, device=self.device)
        last_gae_lam = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        valid_len = self.ptr

        for step in reversed(range(valid_len)):
            next_non_terminal = 1.0 - self.dones[step].float()
            if step == valid_len - 1:
                next_values = last_val
            else:
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns[:valid_len] = self.advantages[:valid_len] + self.values[:valid_len]

        # Normalize advantages over stored trajectory
        adv_mean = self.advantages[:valid_len].mean()
        adv_std = self.advantages[:valid_len].std() + 1e-8
        self.advantages[:valid_len] = (self.advantages[:valid_len] - adv_mean) / adv_std

    def get_batches(
        self,
        batch_size: int = 512,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Yield mini-batches of transitions."""
        valid_len = self.ptr
        indices = np.random.permutation(valid_len)

        for start_idx in range(0, valid_len, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            if len(batch_indices) < 2:
                continue

            idx_t = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
            yield (
                self.observations[idx_t],
                self.actions[idx_t],
                self.action_masks[idx_t],
                self.log_probs[idx_t],
                self.advantages[idx_t],
                self.returns[idx_t],
                self.values[idx_t],
            )


class MaskablePPO:
    """Maskable Proximal Policy Optimization (PPO) agent with exact health telemetry."""

    def __init__(
        self,
        model: TFTActorCritic,
        lr: float = 2.5e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.05,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device or torch.device("cpu")

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def set_learning_rate(self, lr: float) -> None:
        """Update optimizer learning rate."""
        self.lr = max(1e-6, lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def set_entropy_coef(self, coef: float) -> None:
        """Update entropy regularization coefficient."""
        self.entropy_coef = max(0.0001, coef)

    def update(
        self,
        buffer: RolloutBuffer,
        num_epochs: int = 4,
        batch_size: int = 512,
    ) -> dict[str, float]:
        """Perform PPO gradient updates across stored rollouts."""
        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0
        early_stopped = False

        # Compute initial explained variance over the entire buffer
        valid_len = buffer.ptr
        y_true = buffer.returns[:valid_len].cpu().numpy()
        y_pred = buffer.values[:valid_len].cpu().numpy()
        var_y = np.var(y_true)
        if var_y > 1e-8:
            explained_var = 1.0 - (np.var(y_true - y_pred) / var_y)
        else:
            explained_var = 0.0

        for epoch in range(num_epochs):
            if early_stopped:
                break

            for obs, actions, masks, old_log_probs, advantages, returns, old_values in buffer.get_batches(batch_size):
                # Evaluate actions under current policy
                new_log_probs, entropy, new_values = self.model.evaluate_actions(obs, actions, masks)

                # Ratio r_t(theta) = exp(log pi_theta - log pi_theta_old)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Approximate KL Divergence for safety / early stopping
                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())

                if approx_kl > self.target_kl:
                    early_stopped = True
                    break

                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with optional clipping
                v_clipped = old_values + torch.clamp(new_values - old_values, -self.clip_ratio, self.clip_ratio)
                v_loss1 = F.mse_loss(new_values, returns)
                v_loss2 = F.mse_loss(v_clipped, returns)
                value_loss = torch.max(v_loss1, v_loss2)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + (self.value_coef * value_loss) + (self.entropy_coef * entropy_loss)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Tracking metrics
                with torch.no_grad():
                    clip_fraction = float((torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item())

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy_loss += float(entropy_loss.item())
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_updates += 1

        n = max(1, num_updates)
        # Compute post-update explained variance to measure value network convergence
        with torch.no_grad():
            v_post = self.model.get_value(buffer.observations[:valid_len]).squeeze(-1).cpu().numpy()
        var_y = float(np.var(y_true))
        if var_y > 1e-8:
            explained_var = float(1.0 - (np.var(y_true - v_post) / var_y))
        else:
            explained_var = 0.0

        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy_loss": total_entropy_loss / n,
            "policy_entropy": - (total_entropy_loss / n),
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_fraction / n,
            "explained_variance": float(explained_var),
            "early_stopped": float(early_stopped),
            "learning_rate": self.lr,
            "entropy_coef": self.entropy_coef,
        }
