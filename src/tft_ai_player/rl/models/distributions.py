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


# 12 Categorical Action Types with precise slice offsets (306 actions)
ACTION_TYPE_RANGES: list[tuple[int, int]] = [
    (0, 1),       # 0: PASS (len 1)
    (1, 6),       # 1: BUY_SHOP (len 5)
    (6, 7),       # 2: REROLL_SHOP (len 1)
    (7, 8),       # 3: BUY_EXP (len 1)
    (8, 9),       # 4: TOGGLE_LOCK (len 1)
    (9, 18),      # 5: SELL_BENCH (len 9)
    (18, 30),     # 6: SELL_BOARD (len 12)
    (30, 39),     # 7: DEPLOY_BENCH_TO_BOARD (len 9)
    (39, 51),     # 8: RECALL_BOARD_TO_BENCH (len 12)
    (51, 171),    # 9: EQUIP_ITEM_BOARD (len 120)
    (171, 261),   # 10: EQUIP_ITEM_BENCH (len 90)
    (261, 306),   # 11: COMBINE_ITEMS (len 45)
]
NUM_ACTION_TYPES = len(ACTION_TYPE_RANGES)

# Precomputed mapping from action_id (0..305) -> (type_id, arg_offset)
_ACTION_TO_TYPE_ARRAY = np.zeros(306, dtype=np.int64)
_ACTION_TO_OFFSET_ARRAY = np.zeros(306, dtype=np.int64)
for _t_idx, (_start, _end) in enumerate(ACTION_TYPE_RANGES):
    for _a_idx in range(_start, _end):
        _ACTION_TO_TYPE_ARRAY[_a_idx] = _t_idx
        _ACTION_TO_OFFSET_ARRAY[_a_idx] = _a_idx - _start


def get_action_type_and_offset(action_id: int) -> tuple[int, int]:
    """Get high-level action type index (0..11) and argument offset within category."""
    return int(_ACTION_TO_TYPE_ARRAY[action_id]), int(_ACTION_TO_OFFSET_ARRAY[action_id])


try:
    import torch
    import torch.nn as nn
    from torch.distributions.categorical import Categorical

    # Precomputed torch lookup tensors
    _ACTION_TO_TYPE_TENSOR = torch.as_tensor(_ACTION_TO_TYPE_ARRAY, dtype=torch.long)
    _ACTION_TO_OFFSET_TENSOR = torch.as_tensor(_ACTION_TO_OFFSET_ARRAY, dtype=torch.long)

    class MaskedCategorical(Categorical):
        """PyTorch categorical distribution with invalid action masking."""

        def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
            # Mask is expected to be boolean tensor (True = valid, False = invalid)
            self.mask = mask.bool()
            # If all are False for a row (should not happen with valid action mask), unmask PASS (0)
            all_false = ~self.mask.any(dim=-1, keepdim=True)
            safe_mask = self.mask.clone()
            if all_false.any():
                safe_mask = safe_mask | (all_false & torch.zeros_like(safe_mask).scatter_(-1, torch.zeros_like(all_false, dtype=torch.long), True))
            # Set invalid actions to a large negative constant
            masked_logits = torch.where(safe_mask, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))
            super().__init__(logits=masked_logits)

        def sample_action(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
            """Sample action and return (action, log_prob)."""
            if deterministic:
                action = torch.argmax(self.probs, dim=-1)
            else:
                action = super().sample()
            log_prob = self.log_prob(action)
            return action, log_prob

    def compute_type_mask(action_mask: torch.Tensor) -> torch.Tensor:
        """Compute (batch, 13) boolean type mask indicating which action types have >=1 valid action."""
        batch_size = action_mask.shape[0]
        device = action_mask.device
        type_mask = torch.zeros((batch_size, NUM_ACTION_TYPES), dtype=torch.bool, device=device)
        for t_idx, (start, end) in enumerate(ACTION_TYPE_RANGES):
            type_mask[:, t_idx] = action_mask[:, start:end].any(dim=-1)
        # Ensure at least PASS (0) is valid if somehow all false
        all_false = ~type_mask.any(dim=-1, keepdim=True)
        if all_false.any():
            type_mask[:, 0] = type_mask[:, 0] | all_false.squeeze(-1)
        return type_mask

    class HierarchicalMaskedCategorical:
        """Hierarchical policy distribution: Type Head (13) + Conditioned Argument Head."""

        def __init__(
            self,
            type_logits: torch.Tensor,
            arg_logits: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> None:
            self.type_logits = type_logits
            self.arg_logits = arg_logits
            self.action_mask = action_mask.bool()
            self.type_mask = compute_type_mask(self.action_mask)
            self.type_dist = MaskedCategorical(logits=type_logits, mask=self.type_mask)

        def sample_action(self, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
            """Sample hierarchical action: Type -> Argument -> Global Action."""
            batch_size = self.type_logits.shape[0]
            device = self.type_logits.device

            # 1. Sample action type
            selected_type, type_log_prob = self.type_dist.sample_action(deterministic=deterministic)

            # 2. For each sample in batch, sample argument within selected type's range
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            total_log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)

            for i in range(batch_size):
                t = int(selected_type[i].item())
                start, end = ACTION_TYPE_RANGES[t]
                sub_logits = self.arg_logits[i, start:end].unsqueeze(0)
                sub_mask = self.action_mask[i, start:end].unsqueeze(0)
                sub_dist = MaskedCategorical(logits=sub_logits, mask=sub_mask)
                arg_idx, arg_log_prob = sub_dist.sample_action(deterministic=deterministic)
                global_act = start + int(arg_idx.item())
                actions[i] = global_act
                total_log_probs[i] = type_log_prob[i] + arg_log_prob.squeeze(0)

            return actions, total_log_probs

        def evaluate(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute joint log-probabilities and entropy for given actions."""
            batch_size = actions.shape[0]
            device = actions.device

            type_lookup = _ACTION_TO_TYPE_TENSOR.to(device)
            offset_lookup = _ACTION_TO_OFFSET_TENSOR.to(device)

            types = type_lookup[actions]
            offsets = offset_lookup[actions]

            # 1. Type log probability
            type_log_probs = self.type_dist.log_prob(types)
            type_entropy = self.type_dist.entropy()

            # 2. Argument log probability and entropy per item in batch
            arg_log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
            arg_entropy = torch.zeros(batch_size, dtype=torch.float32, device=device)

            for i in range(batch_size):
                t = int(types[i].item())
                start, end = ACTION_TYPE_RANGES[t]
                sub_logits = self.arg_logits[i, start:end].unsqueeze(0)
                sub_mask = self.action_mask[i, start:end].unsqueeze(0)
                sub_dist = MaskedCategorical(logits=sub_logits, mask=sub_mask)
                arg_log_probs[i] = sub_dist.log_prob(offsets[i:i+1])
                arg_entropy[i] = sub_dist.entropy()

            total_log_probs = type_log_probs + arg_log_probs
            total_entropy = type_entropy + arg_entropy
            return total_log_probs, total_entropy

except ImportError:  # pragma: no cover
    MaskedCategorical = None  # type: ignore[assignment, misc]
    HierarchicalMaskedCategorical = None  # type: ignore[assignment, misc]
