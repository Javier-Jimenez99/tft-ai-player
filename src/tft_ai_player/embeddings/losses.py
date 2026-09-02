"""Robust, simplified loss functions for Tri-Objective Multi-Task Pre-training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Clean, stable Temperature-scaled Time-Contrastive InfoNCE Loss."""

    def __init__(self, temperature: float = 0.07, **kwargs) -> None:
        super().__init__()
        self.temperature = max(1e-4, temperature)

    def forward(
        self,
        anchor_proj: torch.Tensor,
        positive_proj: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z_a = F.normalize(anchor_proj, p=2, dim=-1)
        z_p = F.normalize(positive_proj, p=2, dim=-1)

        batch_size = z_a.shape[0]
        if batch_size <= 1:
            sim = (z_a * z_p).sum(dim=-1)
            loss = (1.0 - sim).mean()
            return loss, {"info_nce_loss": float(loss.item()), "pos_sim": float(sim.mean().item()), "neg_sim": 0.0}

        sim_matrix = torch.matmul(z_a, z_p.T) / self.temperature  # (B, B)
        labels = torch.arange(batch_size, device=z_a.device)

        loss_a2p = F.cross_entropy(sim_matrix, labels)
        loss_p2a = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_a2p + loss_p2a) / 2.0

        with torch.no_grad():
            diag_sim = torch.diagonal(torch.matmul(z_a, z_p.T)).mean().item()
            off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_a.device)
            off_diag_sim = torch.matmul(z_a, z_p.T)[off_diag_mask].mean().item() if off_diag_mask.any() else 0.0

            # Contrastive Top-1 Retrieval Accuracy
            preds = torch.argmax(sim_matrix, dim=-1)
            retrieval_acc = (preds == labels).float().mean().item()

        metrics = {
            "info_nce_loss": float(loss.item()),
            "pos_sim": float(diag_sim),
            "neg_sim": float(off_diag_sim),
            "contrast_retrieval_acc": float(retrieval_acc),
        }
        return loss, metrics


class MultiTaskTrunkLoss(nn.Module):
    """Tri-Objective Joint Loss with PVP Masking and Balanced Gradients.

    Macro: Top-4 Placement Cross-Entropy (weight = 1.0)
    Micro: PVP Round Win Probability BCE (weight = 0.5, masked on true PVP rounds)
    Flow: Time-Contrastive InfoNCE on Board Features (weight = 0.15)
    """

    def __init__(
        self,
        value_weight: float = 1.0,
        micro_weight: float = 0.5,
        contrast_weight: float = 0.15,
        temperature: float = 0.07,
        **kwargs,
    ) -> None:
        super().__init__()
        self.value_weight = value_weight
        self.micro_weight = micro_weight
        self.contrast_weight = contrast_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.combat_bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.contrastive_loss = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        value_logits: torch.Tensor,
        value_targets: torch.Tensor,
        anchor_proj: torch.Tensor,
        positive_proj: torch.Tensor,
        combat_win_logits: torch.Tensor | None = None,
        combat_win_targets: torch.Tensor | None = None,
        is_pvp_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Macro Loss (Top-4 Placement)
        l_val = self.ce_loss(value_logits, value_targets.long())

        # 2. Flow Loss (Time-Contrastive InfoNCE on isolated board features)
        l_contrast, c_metrics = self.contrastive_loss(
            anchor_proj=anchor_proj,
            positive_proj=positive_proj,
        )

        # 3. Micro Loss (Immediate Round Combat Win Probability BCE with PVP Masking)
        if combat_win_logits is not None and combat_win_targets is not None:
            pred_logits = combat_win_logits.squeeze(-1).float()
            tgt_combat = combat_win_targets.float()
            raw_bce = self.combat_bce_loss(pred_logits, tgt_combat)

            if is_pvp_mask is not None:
                mask = is_pvp_mask.float()
                mask_sum = mask.sum()
                if mask_sum > 0:
                    l_micro = (raw_bce * mask).sum() / mask_sum
                else:
                    l_micro = raw_bce.mean()
            else:
                mask = torch.ones_like(tgt_combat)
                l_micro = raw_bce.mean()

            with torch.no_grad():
                combat_probs = torch.sigmoid(pred_logits)
                combat_preds = (combat_probs >= 0.5).float()
                correct = (combat_preds == tgt_combat).float() * mask
                combat_acc = (correct.sum() / max(1.0, mask.sum().item())).item()
        else:
            l_micro = torch.tensor(0.0, device=value_logits.device)
            combat_acc = 0.0

        total_loss = (
            (self.value_weight * l_val)
            + (self.micro_weight * l_micro)
            + (self.contrast_weight * l_contrast)
        )

        with torch.no_grad():
            preds = torch.argmax(value_logits, dim=-1)
            top4_acc = (preds == value_targets.long()).float().mean().item()

        metrics = {
            "total_loss": float(total_loss.item()),
            "value_ce_loss": float(l_val.item()),
            "value_acc": float(top4_acc),
            "micro_combat_loss": float(l_micro.item()),
            "combat_acc": float(combat_acc),
            **c_metrics,
        }
        return total_loss, metrics
