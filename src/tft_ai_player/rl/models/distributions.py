"""Invalid Action Masking and Categorical Distribution helpers for TFT factorized discrete action spaces."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS


def numpy_masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable masked softmax probabilities using NumPy."""
    b_mask = mask.astype(bool)
    if not np.any(b_mask):
        probs = np.zeros_like(logits, dtype=np.float32)
        probs[0] = 1.0  # fallback to PASS_ROUND
        return probs

    temp = max(temperature, 1e-5)
    masked_logits = np.where(b_mask, logits / temp, -1e9)
    shifted = masked_logits - np.max(masked_logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)
    return probs.astype(np.float32)


def numpy_masked_sample(
    logits: np.ndarray,
    mask: np.ndarray,
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[int, float]:
    """Sample a valid discrete action from logits using NumPy. Returns (action, log_prob)."""
    probs = numpy_masked_softmax(logits, mask)
    if deterministic:
        action = int(np.argmax(probs))
    else:
        generator = rng or np.random.default_rng()
        action = int(generator.choice(len(probs), p=probs))

    log_prob = float(np.log(max(probs[action], 1e-12)))
    return action, log_prob


class MaskedCategorical(Categorical):
    """PyTorch categorical distribution with strict pre-softmax invalid action masking."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        self.mask = mask.bool()
        # Fallback safeguard: if entire row is False, ensure PASS_ROUND (index 0) is valid
        all_false = ~self.mask.any(dim=-1, keepdim=True)
        if all_false.any():
            safe_mask = self.mask.clone()
            safe_mask[..., 0] = safe_mask[..., 0] | all_false.squeeze(-1)
        else:
            safe_mask = self.mask

        # Apply -1e9 logit penalty to invalid actions
        masked_logits = logits.masked_fill(~safe_mask, -1e9)
        super().__init__(logits=masked_logits)

    def sample_action(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        if deterministic:
            action = torch.argmax(self.logits, dim=-1)
        else:
            action = self.sample()
        log_prob = self.log_prob(action)
        return action, log_prob
