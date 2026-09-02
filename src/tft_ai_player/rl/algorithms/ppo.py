"""Maskable Proximal Policy Optimization (PPO) and Rollout Buffer for TFT RL."""

from __future__ import annotations

from typing import Generator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS


class RolloutBuffer:
    """Stores experience trajectories for on-policy PPO training with GAE computation and recurrent state."""

    def __init__(
        self,
        buffer_size: int,
        obs_sample: dict[str, np.ndarray],
        action_dim: int = TOTAL_DISCRETE_ACTIONS,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | None = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")

        # Allocate observation buffers matching keys and shapes
        self.obs_buffers: dict[str, torch.Tensor] = {}
        for key, arr in obs_sample.items():
            if key != "action_mask":
                shape = (buffer_size, *arr.shape)
                self.obs_buffers[key] = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Actions, masks, values, rewards, log_probs, dones, hidden_states
        self.actions = torch.zeros((buffer_size,), dtype=torch.long, device=self.device)
        self.action_masks = torch.zeros((buffer_size, action_dim), dtype=torch.bool, device=self.device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=self.device)
        self.hidden_states = torch.zeros((buffer_size, hidden_dim), dtype=torch.float32, device=self.device)

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
        hidden_state: torch.Tensor | np.ndarray | None = None,
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

        if hidden_state is not None:
            if isinstance(hidden_state, tuple):
                h, _ = hidden_state
                h_vec = h.detach().view(-1)
            elif isinstance(hidden_state, torch.Tensor):
                h_vec = hidden_state.detach().view(-1)
            else:
                h_vec = torch.as_tensor(hidden_state, dtype=torch.float32, device=self.device).view(-1)
            self.hidden_states[self.ptr] = h_vec[: self.hidden_dim]

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
    ) -> Generator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Generate randomized mini-batches for PPO epochs including hidden states."""
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
            batch_old_values = self.values[batch_indices]
            batch_hiddens = self.hidden_states[batch_indices]

            yield batch_obs, batch_actions, batch_masks, batch_log_probs, batch_advantages, batch_returns, batch_old_values, batch_hiddens

    def reset(self) -> None:
        """Reset the buffer pointer."""
        self.ptr = 0
        self.full = False


class MaskablePPO:
    """Maskable Proximal Policy Optimization (PPO) trainer with dynamic entropy scheduling for TFT."""

    def __init__(
        self,
        actor_critic: TFTActorCritic,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        value_coef: float = 0.05,
        entropy_coef: float = 0.05,
        entropy_coef_start: float = 0.05,
        entropy_coef_min: float = 0.005,
        entropy_decay_rate: float = 0.997,
        max_grad_norm: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        self.actor_critic = actor_critic
        self.lr = lr
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_min = entropy_coef_min
        self.entropy_decay_rate = entropy_decay_rate
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        self.actor_critic.to(self.device)
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=lr, eps=1e-5)

    def step_entropy_schedule(self, generation: int) -> float:
        """Update dynamic entropy coefficient using exponential decay schedule."""
        decayed = self.entropy_coef_start * (self.entropy_decay_rate ** generation)
        self.entropy_coef = max(self.entropy_coef_min, float(decayed))
        return self.entropy_coef

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
            for batch_obs, actions, masks, old_log_probs, advantages, returns, old_values, batch_hiddens in buffer.get_batches(batch_size):
                # Normalize advantages per mini-batch for variance reduction
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                norm_advantages = (advantages - adv_mean) / adv_std

                # Evaluate current policy with recorded hidden states
                new_log_probs, new_values, entropy, aux_win, aux_place = self.actor_critic.evaluate_actions(
                    batch_obs, actions, masks, hidden_state=batch_hiddens
                )

                # Policy ratio & clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * norm_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * norm_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # PPO Normalized Clipped Value Loss (guarantees numerical stability on long rollouts)
                ret_mean = returns.mean()
                ret_std = returns.std() + 1e-8
                norm_returns = (returns - ret_mean) / ret_std
                norm_new_values = (new_values - ret_mean) / ret_std
                norm_old_values = (old_values - ret_mean) / ret_std

                values_clipped = norm_old_values + (norm_new_values - norm_old_values).clamp(-self.clip_range, self.clip_range)
                v_loss_unclipped = F.mse_loss(norm_new_values, norm_returns)
                v_loss_clipped = F.mse_loss(values_clipped, norm_returns)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)

                # Entropy regularization bonus
                entropy_loss = -entropy.mean()

                # Self-Supervised Auxiliary Loss (Suphx style)
                # Calibrates internal representation against advantage direction & placement proxy
                target_win_proxy = (norm_advantages > 0).float()
                aux_win_loss = F.binary_cross_entropy(aux_win, target_win_proxy)
                target_place_proxy = torch.clamp(((1.0 - torch.sigmoid(norm_returns)) * 7.0).long(), 0, 7)
                aux_place_loss = F.cross_entropy(aux_place, target_place_proxy)
                aux_loss = 0.05 * aux_win_loss + 0.02 * aux_place_loss

                loss = policy_loss + (self.value_coef * value_loss) + (self.entropy_coef * entropy_loss) + aux_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()

                # Diagnostic metrics from SOTA PPO literature
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                    var_y = torch.var(returns)
                    ev = 1.0 - (torch.var(returns - new_values) / (var_y + 1e-8)) if var_y > 1e-8 else torch.tensor(0.0, device=returns.device)

                total_approx_kl = total_approx_kl + approx_kl.item() if "total_approx_kl" in locals() else approx_kl.item()
                total_clip_frac = total_clip_frac + clip_frac.item() if "total_clip_frac" in locals() else clip_frac.item()
                total_ev = total_ev + ev.item() if "total_ev" in locals() else ev.item()
                num_updates += 1

        n = max(num_updates, 1)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": -total_entropy_loss / n,
            "entropy_coef": self.entropy_coef,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_frac / n,
            "explained_variance": max(-1.0, min(1.0, total_ev / n)),
        }
