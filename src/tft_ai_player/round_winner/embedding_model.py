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
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.freeze_trunk = freeze_trunk

        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False

        f_dim = self.trunk.fused_dim  # e.g. 384 or 320
        # Interaction features: [z_a, z_b, z_a - z_b, z_a * z_b] = 4 * f_dim + 1 (cos_sim)
        in_features = f_dim * 4 + 1

        self.interaction_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

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

        diff = z_focal - z_opp
        prod = z_focal * z_opp
        norm_focal = F.normalize(z_focal, p=2, dim=-1, eps=1e-8)
        norm_opp = F.normalize(z_opp, p=2, dim=-1, eps=1e-8)
        cos_sim = (norm_focal * norm_opp).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        interaction = torch.cat([z_focal, z_opp, diff, prod, cos_sim], dim=-1)
        logit = self.interaction_mlp(interaction).squeeze(-1)
        return logit


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

            f_scalars = np.array([
                float(row.get("focal_health", 100)) / 100.0,
                float(row.get("focal_gold", 50)) / 100.0,
                float(row.get("focal_level", 6)) / 10.0,
                0.0,
                float(stage_tuple[0]) / 10.0,
                float(stage_tuple[1]) / 10.0,
                float(row.get("focal_unit_count", len(f_board))) / 10.0,
                float(row.get("focal_item_count", 0)) / 10.0,
            ], dtype=np.float32)

            o_scalars = np.array([
                float(row.get("opponent_health", 100)) / 100.0,
                50.0 / 100.0,
                float(row.get("opponent_level", 6)) / 10.0,
                0.0,
                float(stage_tuple[0]) / 10.0,
                float(stage_tuple[1]) / 10.0,
                float(row.get("opponent_unit_count", len(o_board))) / 10.0,
                float(row.get("opponent_item_count", 0)) / 10.0,
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
