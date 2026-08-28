"""Invalid Action Masking and Categorical Distribution helpers for TFT micro-action spaces."""

from __future__ import annotations

import numpy as np


def numpy_masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable masked softmax probabilities using NumPy."""
    # Convert mask to boolean
    b_mask = mask.astype(bool)
    if not np.any(b_mask):
        # If all masked, fallback to uniform
        probs = np.ones_like(logits, dtype=np.float32) / len(logits)
        return probs

    # Apply large negative penalty to invalid actions
    masked_logits = np.where(b_mask, logits / max(temperature, 1e-5), -1e9)
    # Subtract max for numerical stability
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


try:
    import torch
    import torch.nn as nn
    from torch.distributions.categorical import Categorical

    class MaskedCategorical(Categorical):
        """PyTorch categorical distribution with invalid action masking."""

        def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
            # Mask is expected to be boolean tensor (True = valid, False = invalid)
            self.mask = mask.bool()
            # Set invalid actions to a large negative constant
            masked_logits = torch.where(self.mask, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))
            super().__init__(logits=masked_logits)

        def sample_action(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
            """Sample action and return (action, log_prob)."""
            if deterministic:
                action = torch.argmax(self.probs, dim=-1)
            else:
                action = super().sample()
            log_prob = self.log_prob(action)
            return action, log_prob

except ImportError:  # pragma: no cover
    MaskedCategorical = None  # type: ignore[assignment, misc]
