"""Deep Learning and Pre-trained Embedding models for TFT Round Winner Prediction.

Implements differentiable Siamese Neural Networks (GPU-native) and embedding feature
extractors using the pre-trained Multi-Modal Fusion Trunk (trunk_pretrained.pt).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.embeddings.dataset import parse_loc_to_row_col, parse_stage_string


class DeepSiameseCombatNet(nn.Module):
    """GPU-native Deep Differentiable Siamese Combat Prediction Network.

    Takes focal player and opponent player board representations through
    the Multi-Modal Fusion Trunk and computes symmetric/anti-symmetric
    differential interaction features to predict P(focal_wins).
    """

    def __init__(
        self,
        trunk: MultiModalFusionTrunk,
        freeze_trunk: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        logit_scale: float = 2.30,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.freeze_trunk = freeze_trunk
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale), dtype=torch.float32))

        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False

        f_dim = self.trunk.fused_dim  # e.g. 384 or 320

        # 1. Intrinsic Board Strength Head (Bradley-Terry formulation: s(A) - s(B))
        self.strength_mlp = nn.Sequential(
            nn.Linear(f_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 2. Relational Matchup Interaction Head (g(A, B) - g(B, A))
        in_features = f_dim * 4 + 1
        self.advantage_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """Load state dict with automatic translation for legacy 'interaction_mlp' keys."""
        remapped: dict[str, Any] = {}
        for k, v in state_dict.items():
            if k.startswith("interaction_mlp."):
                remapped[k.replace("interaction_mlp.", "advantage_mlp.")] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)

    def forward_single_board(
        self,
        board_champ_ids: torch.Tensor,
        board_star_levels: torch.Tensor,
        board_item_ids: torch.Tensor,
        board_traits: torch.Tensor,
        state_scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Extract multi-modal fused state vector (B, f_dim)."""
        if self.freeze_trunk:
            with torch.no_grad():
                return self.trunk(
                    board_champ_ids=board_champ_ids,
                    board_star_levels=board_star_levels,
                    board_item_ids=board_item_ids,
                    board_traits=board_traits,
                    state_scalars=state_scalars,
                )
        return self.trunk(
            board_champ_ids=board_champ_ids,
            board_star_levels=board_star_levels,
            board_item_ids=board_item_ids,
            board_traits=board_traits,
            state_scalars=state_scalars,
        )

    def predict_board_power(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute scalar intrinsic board strength s(A)."""
        z = self.forward_single_board(
            batch["board_champ_ids"],
            batch["board_star_levels"],
            batch["board_item_ids"],
            batch["board_traits"],
            batch["state_scalars"],
        )
        return (self.strength_mlp(z).squeeze(-1)) * self.logit_scale

    def compute_advantage_diff(self, z_focal: torch.Tensor, z_opp: torch.Tensor) -> torch.Tensor:
        """Compute mathematically exact anti-symmetric combat logit.

        logit(A, B) = [s(A) - s(B)] + [g(A, B) - g(B, A)]
        Guarantees:
            1. logit(A, A) = 0 => P(A vs A) = 0.500000 (Exact Mirror Match)
            2. logit(B, A) = -logit(A, B) => P(A vs B) + P(B vs A) = 1.000000 (Exact Anti-Symmetry)
        """
        # Intrinsic strength delta
        s_focal = self.strength_mlp(z_focal)
        s_opp = self.strength_mlp(z_opp)
        strength_delta = s_focal - s_opp

        # Relational matchup delta
        diff = z_focal - z_opp
        prod = z_focal * z_opp
        norm_focal = F.normalize(z_focal, p=2, dim=-1, eps=1e-8)
        norm_opp = F.normalize(z_opp, p=2, dim=-1, eps=1e-8)
        cos_sim = (norm_focal * norm_opp).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        feat_ab = torch.cat([z_focal, z_opp, diff, prod, cos_sim], dim=-1)
        feat_ba = torch.cat([z_opp, z_focal, -diff, prod, cos_sim], dim=-1)

        combined = torch.cat([feat_ab, feat_ba], dim=0)
        matchup_scores = self.advantage_mlp(combined)
        score_ab, score_ba = matchup_scores.chunk(2, dim=0)
        matchup_delta = score_ab - score_ba

        logit = (strength_delta + matchup_delta).squeeze(-1) * self.logit_scale
        return logit

    def forward(
        self,
        focal_batch: dict[str, torch.Tensor],
        opp_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combat win logit: P(focal_wins) = sigmoid(logit)."""
        z_focal = self.forward_single_board(
            focal_batch["board_champ_ids"],
            focal_batch["board_star_levels"],
            focal_batch["board_item_ids"],
            focal_batch["board_traits"],
            focal_batch["state_scalars"],
        )
        z_opp = self.forward_single_board(
            opp_batch["board_champ_ids"],
            opp_batch["board_star_levels"],
            opp_batch["board_item_ids"],
            opp_batch["board_traits"],
            opp_batch["state_scalars"],
        )

        return self.compute_advantage_diff(z_focal, z_opp)


def build_optimizer_param_groups(
    model: DeepSiameseCombatNet,
    weight_decay: float = 1e-4,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split model parameters into weight-decay and no-decay groups.

    Excludes 1D biases, LayerNorms, and final output linear projections from decay.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_final_head = "strength_mlp.4" in name or "advantage_mlp.8" in name
        if p.dim() >= 2 and not is_final_head:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    return decay_params, no_decay_params


def _safe_float(val: Any, default: float) -> float:
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (ValueError, TypeError):
        return default


class CombatDataset(Dataset):
    """Dataset parsing input_state_json into focal and opponent PyTorch tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: ChampionVocabulary,
        item_vocab: ItemVocabulary,
        trait_vocab: TraitVocabulary,
    ) -> None:
        self.vocab = vocab
        self.item_vocab = item_vocab
        self.trait_vocab = trait_vocab
        self.samples = []

        for _, row in df.iterrows():
            raw_json = row.get("input_state_json")
            if not isinstance(raw_json, (str, dict)):
                continue
            data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            if "focal_board" not in data or "opponent_board" not in data:
                continue

            f_board = data["focal_board"]
            o_board = data["opponent_board"]
            stage_str = str(row.get("round_stage", "2-1"))
            stage_tuple = parse_stage_string(stage_str)

            f_items_equipped = sum(len(u.get("items", [])) for u in f_board if isinstance(u, dict))
            raw_f_items = row.get("focal_item_count")
            f_item_val = _safe_float(raw_f_items, float(f_items_equipped))
            if f_item_val <= 0:
                f_item_val = float(f_items_equipped)

            o_items_equipped = sum(len(u.get("items", [])) for u in o_board if isinstance(u, dict))
            raw_o_items = row.get("opponent_item_count")
            o_item_val = _safe_float(raw_o_items, float(o_items_equipped))
            if o_item_val <= 0:
                o_item_val = float(o_items_equipped)

            raw_f_units = row.get("focal_unit_count")
            f_unit_val = _safe_float(raw_f_units, float(len(f_board)))
            if f_unit_val <= 0:
                f_unit_val = float(len(f_board))

            raw_o_units = row.get("opponent_unit_count")
            o_unit_val = _safe_float(raw_o_units, float(len(o_board)))
            if o_unit_val <= 0:
                o_unit_val = float(len(o_board))

            f_health = _safe_float(row.get("focal_health"), 100.0)
            f_gold = _safe_float(row.get("focal_gold"), 50.0)
            f_level = _safe_float(row.get("focal_level"), 6.0)

            o_health = _safe_float(row.get("opponent_health"), 100.0)
            o_gold = _safe_float(row.get("opponent_gold"), 50.0)
            o_level = _safe_float(row.get("opponent_level"), 6.0)

            f_scalars = np.array([
                f_health / 100.0,
                f_gold / 100.0,
                f_level / 10.0,
                0.0,
                float(stage_tuple[0]) / 10.0,
                float(stage_tuple[1]) / 10.0,
                f_unit_val / 10.0,
                f_item_val / 10.0,
            ], dtype=np.float32)

            o_scalars = np.array([
                o_health / 100.0,
                o_gold / 100.0,
                o_level / 10.0,
                0.0,
                float(stage_tuple[0]) / 10.0,
                float(stage_tuple[1]) / 10.0,
                o_unit_val / 10.0,
                o_item_val / 10.0,
            ], dtype=np.float32)

            f_enc = self._encode_board(f_board, f_scalars)
            o_enc = self._encode_board(o_board, o_scalars)
            label = int(row.get("label", 0))
            meta_prob = float(row.get("metatft_win_prob", 0.5))

            self.samples.append({
                "focal": f_enc,
                "opp": o_enc,
                "label": label,
                "metatft_win_prob": meta_prob,
                "round_stage": stage_str,
                "match_id": str(row.get("match_id", "")),
            })

    def _encode_board(self, units: list[dict[str, Any]], scalars: np.ndarray) -> dict[str, np.ndarray]:
        board_grid = np.zeros((4, 7), dtype=np.int64)
        star_grid = np.zeros((4, 7), dtype=np.int64)
        item_grid = np.zeros((4, 7, 3), dtype=np.int64)
        champ_names: list[str] = []
        occupied: set[tuple[int, int]] = set()

        for unit in units:
            if not isinstance(unit, dict):
                continue
            u_name = unit.get("unit") or unit.get("champion") or unit.get("apiName")
            u_tier = int(unit.get("tier", 1) or 1)
            u_loc = unit.get("loc")
            raw_items = unit.get("items", []) or []

            c_idx = self.vocab.encode(u_name)
            if u_name:
                champ_names.append(u_name)

            item_idxs = [self.item_vocab.encode(it) for it in raw_items[:3]]
            while len(item_idxs) < 3:
                item_idxs.append(0)

            coords = parse_loc_to_row_col(u_loc)
            if coords is not None and coords not in occupied:
                r, c = coords
                board_grid[r, c] = c_idx
                star_grid[r, c] = u_tier
                item_grid[r, c] = item_idxs
                occupied.add(coords)
            else:
                for r in range(4):
                    for c in range(7):
                        if (r, c) not in occupied:
                            board_grid[r, c] = c_idx
                            star_grid[r, c] = u_tier
                            item_grid[r, c] = item_idxs
                            occupied.add((r, c))
                            break
                    if coords in occupied:
                        break

        trait_vec = self.trait_vocab.compute_trait_vector(champ_names)
        return {
            "board_champ_ids": board_grid,
            "board_star_levels": star_grid,
            "board_item_ids": item_grid,
            "board_traits": trait_vec,
            "state_scalars": scalars,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def combat_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    focal_b = torch.tensor(np.stack([b["focal"]["board_champ_ids"] for b in batch]), dtype=torch.long)
    focal_s = torch.tensor(np.stack([b["focal"]["board_star_levels"] for b in batch]), dtype=torch.long)
    focal_i = torch.tensor(np.stack([b["focal"]["board_item_ids"] for b in batch]), dtype=torch.long)
    focal_t = torch.tensor(np.stack([b["focal"]["board_traits"] for b in batch]), dtype=torch.float32)
    focal_sc = torch.tensor(np.stack([b["focal"]["state_scalars"] for b in batch]), dtype=torch.float32)

    opp_b = torch.tensor(np.stack([b["opp"]["board_champ_ids"] for b in batch]), dtype=torch.long)
    opp_s = torch.tensor(np.stack([b["opp"]["board_star_levels"] for b in batch]), dtype=torch.long)
    opp_i = torch.tensor(np.stack([b["opp"]["board_item_ids"] for b in batch]), dtype=torch.long)
    opp_t = torch.tensor(np.stack([b["opp"]["board_traits"] for b in batch]), dtype=torch.float32)
    opp_sc = torch.tensor(np.stack([b["opp"]["state_scalars"] for b in batch]), dtype=torch.float32)

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    metatft_probs = np.array([b["metatft_win_prob"] for b in batch], dtype=np.float32)
    stages = [b["round_stage"] for b in batch]

    return {
        "focal": {
            "board_champ_ids": focal_b,
            "board_star_levels": focal_s,
            "board_item_ids": focal_i,
            "board_traits": focal_t,
            "state_scalars": focal_sc,
        },
        "opp": {
            "board_champ_ids": opp_b,
            "board_star_levels": opp_s,
            "board_item_ids": opp_i,
            "board_traits": opp_t,
            "state_scalars": opp_sc,
        },
        "labels": labels,
        "metatft_probs": metatft_probs,
        "stages": stages,
    }
