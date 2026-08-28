"""Maskable Proximal Policy Optimization (PPO) and Rollout Buffer for TFT RL."""

from __future__ import annotations

from typing import Generator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tft_ai_player.rl.models.networks import TFTActorCritic


class RolloutBuffer:
    """Stores experience trajectories for on-policy PPO training with GAE computation."""

    def __init__(
        self,
        buffer_size: int,
        obs_sample: dict[str, np.ndarray],
        action_dim: int = 1721,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | None = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")

        # Allocate observation buffers matching keys and shapes
        self.obs_buffers: dict[str, torch.Tensor] = {}
        for key, arr in obs_sample.items():
            if key != "action_mask":
                shape = (buffer_size, *arr.shape)
                self.obs_buffers[key] = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Actions, masks, values, rewards, log_probs, dones
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

    def add(
        self,
        obs_dict: dict[str, np.ndarray],
        action: int,
        action_mask: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add a single transition to the rollout buffer."""
        for key, arr in obs_dict.items():
            if key != "action_mask":
                self.obs_buffers[key][self.ptr] = torch.as_tensor(arr, dtype=torch.float32, device=self.device)

        self.actions[self.ptr] = int(action)
        self.action_masks[self.ptr] = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = bool(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float, done: bool) -> None:
        """Compute Generalized Advantage Estimation (GAE-lambda) and discounted returns."""
        last_val = torch.tensor(last_value, dtype=torch.float32, device=self.device)
        last_done = torch.tensor(done, dtype=torch.bool, device=self.device)
        last_gae_lam = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = (~last_done).float()
                next_value = last_val
            else:
                next_non_terminal = (~self.dones[step + 1]).float()
                next_value = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_batches(
        self,
        batch_size: int,
    ) -> Generator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Generate randomized mini-batches for PPO epochs."""
        total_steps = self.ptr
        indices = np.random.permutation(total_steps)

        for start_idx in range(0, total_steps, batch_size):
            end_idx = min(start_idx + batch_size, total_steps)
            batch_indices = indices[start_idx:end_idx]

            batch_obs = {k: v[batch_indices] for k, v in self.obs_buffers.items()}
            batch_actions = self.actions[batch_indices]
            batch_masks = self.action_masks[batch_indices]
            batch_log_probs = self.log_probs[batch_indices]
            batch_advantages = self.advantages[batch_indices]
            batch_returns = self.returns[batch_indices]

            yield batch_obs, batch_actions, batch_masks, batch_log_probs, batch_advantages, batch_returns

    def reset(self) -> None:
        """Reset the buffer pointer."""
        self.ptr = 0
        self.full = False


class MaskablePPO:
    """Maskable Proximal Policy Optimization (PPO) trainer for TFT."""

    def __init__(
        self,
        actor_critic: TFTActorCritic,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        self.actor_critic = actor_critic
        self.lr = lr
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        self.actor_critic.to(self.device)
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=lr, eps=1e-5)

    def train_epoch(
        self,
        buffer: RolloutBuffer,
        batch_size: int = 64,
        num_epochs: int = 4,
    ) -> dict[str, float]:
        """Perform PPO gradient updates across buffer trajectories."""
        self.actor_critic.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        num_updates = 0

        for _ in range(num_epochs):
            for batch_obs, actions, masks, old_log_probs, advantages, returns in buffer.get_batches(batch_size):
                # Normalize advantages per mini-batch for variance reduction
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                norm_advantages = (advantages - adv_mean) / adv_std

                # Evaluate current policy
                new_log_probs, new_values, entropy = self.actor_critic.evaluate_actions(
                    batch_obs, actions, masks
                )

                # Policy ratio & clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * norm_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * norm_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(new_values, returns)

                # Entropy regularization bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + (self.value_coef * value_loss) + (self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                num_updates += 1

        n = max(num_updates, 1)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": -total_entropy_loss / n,
        }
