r"""State Transition Predictor for TFT Offline Behavioral Cloning.

Learns deterministic state-to-state transitions (s_t -> s_{t+1}) entirely within the
320D latent space of a frozen MultiModalFusionTrunk.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .dataset import (
    SnapshotPairCollate,
    TFTPretrainDataset,
    is_pvp_round,
    parse_loc_to_row_col,
    parse_stage_string,
)
from .model import MultiModalFusionTrunk
from .vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary

logger = logging.getLogger(__name__)


class ResidualMLPBlock(nn.Module):
    r"""Residual MLP Block with LayerNorm, ReLU, and Dropout.

    f(x) = x + Dropout(Linear(ReLU(LayerNorm(Linear(x))))).
    Preserves vector identity across deep representations.
    """

    def __init__(
        self,
        dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.dropout(self.net(x)))


class StateTransitionPredictor(nn.Module):
    r"""Deep Residual MLP for TFT State Transition Prediction (s_t -> s_{t+1}).

    Maps current round 320D latent representation to the predicted ideal next-round
    320D latent representation. Preserves base vector identity using residual connections.

    Input:  s_t in R^{320} (Frozen MultiModalFusionTrunk fused_state)
    Hidden: Linear(320, 512) -> LayerNorm -> ReLU -> Dropout -> ResidualBlocks
    Output: \hat{s}_{t+1} in R^{320}
    """

    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 512,
        output_dim: int = 320,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual_delta: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual_delta = use_residual_delta and (input_dim == output_dim)

        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(dim=hidden_dim, dropout=dropout)
            for _ in range(max(1, num_layers))
        ])

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        r"""Forward pass predicting the next state \hat{s}_{t+1} from s_t.

        Args:
            s_t: Tensor of shape (Batch, 320) representing current latent states.

        Returns:
            \hat{s}_{t+1}: Tensor of shape (Batch, 320) representing next latent states.
        """
        h = self.in_proj(s_t)
        for block in self.blocks:
            h = block(h)
        delta = self.out_proj(h)

        if self.use_residual_delta:
            return s_t + delta
        return delta

    @torch.no_grad()
    def predict_next_state(self, s_t: torch.Tensor) -> torch.Tensor:
        """Inference helper for next-round state prediction."""
        self.eval()
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
            return self.forward(s_t).squeeze(0)
        return self.forward(s_t)

    def save_predictor(self, path: str | Path, extra_info: dict[str, Any] | None = None) -> None:
        """Save model weights and architectural config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "use_residual_delta": self.use_residual_delta,
            },
            "extra_info": extra_info or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_predictor(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> StateTransitionPredictor:
        """Load model weights and reconstruct instance from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model = cls(
            input_dim=config.get("input_dim", 320),
            hidden_dim=config.get("hidden_dim", 512),
            output_dim=config.get("output_dim", 320),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.1),
            use_residual_delta=config.get("use_residual_delta", True),
        )
        if isinstance(checkpoint, dict):
            sd = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(sd)
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model


class StateTransitionLoss(nn.Module):
    r"""Dual-Objective Loss for Latent State Transition Prediction.

    L_Transition = L_Huber(\hat{s}_{t+1}, s_{t+1}) + \lambda * L_Cosine(\hat{s}_{t+1}, s_{t+1})

    1. Huber Loss (Smooth L1):
       Handles absolute coordinate distance between predicted and ground-truth vectors.
       Prevents gradient explosions during massive multi-unit pivots.

    2. Cosine Embedding Loss:
       Ensures the predicted vector points in the exact same mathematical direction as
       the ground-truth next state, aligning with the InfoNCE contrastive geometry.
    """

    def __init__(
        self,
        lambda_cosine: float = 0.5,
        huber_beta: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_cosine = max(0.0, lambda_cosine)
        self.huber_beta = max(1e-4, huber_beta)
        self.eps = eps
        self.huber = nn.SmoothL1Loss(beta=self.huber_beta)

    def forward(
        self,
        pred_next: torch.Tensor,
        true_next: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Compute dual-objective transition loss and detailed metrics.

        Args:
            pred_next: Predicted next state \hat{s}_{t+1} (Batch, 320).
            true_next: Ground-truth next state s_{t+1} (Batch, 320).

        Returns:
            total_loss: Scalar loss tensor with attached computational graph.
            metrics: Dictionary of evaluated metrics (huber, cosine, errors).
        """
        # 1. Huber (Smooth L1) Coordinate Distance Loss
        l_huber = self.huber(pred_next, true_next)

        # 2. Cosine Directional Loss: 1 - cosine_similarity (with numerically stable normalization)
        pred_norm = torch.clamp(torch.norm(pred_next, p=2, dim=-1, keepdim=True), min=self.eps)
        true_norm = torch.clamp(torch.norm(true_next, p=2, dim=-1, keepdim=True), min=self.eps)
        cos_sim = (pred_next * true_next).sum(dim=-1, keepdim=True) / (pred_norm * true_norm)
        cos_sim = cos_sim.squeeze(-1)
        mean_cos_sim = cos_sim.mean()
        l_cosine = (1.0 - mean_cos_sim).clamp(min=0.0)

        # Joint Dual-Objective Loss
        total_loss = l_huber + (self.lambda_cosine * l_cosine)

        with torch.no_grad():
            l1_error = F.l1_loss(pred_next, true_next).item()
            l2_error = F.mse_loss(pred_next, true_next).item()
            pred_n = torch.norm(pred_next, p=2, dim=-1).mean().item()
            true_n = torch.norm(true_next, p=2, dim=-1).mean().item()

        metrics = {
            "total_loss": float(total_loss.item()),
            "huber_loss": float(l_huber.item()),
            "cosine_loss": float(l_cosine.item()),
            "cosine_similarity": float(mean_cos_sim.item()),
            "l1_error": float(l1_error),
            "l2_error": float(l2_error),
            "pred_norm": float(pred_n),
            "true_norm": float(true_n),
        }
        return total_loss, metrics


def is_valid_pvp_transition(
    stage_curr: tuple[int, int],
    round_type_curr: str | None,
    stage_next: tuple[int, int],
    round_type_next: str | None,
) -> bool:
    """Strictly filter out Stage 1 creeps, Krugs, Wolves, Raptors, and Carousels.

    Transition pairs (s_t, s_{t+1}) must represent standard player-driven shopping/planning phases.
    """
    # 1. Reject if either snapshot is Stage 1 (1-1..1-4)
    if stage_curr[0] < 2 or stage_next[0] < 2:
        return False

    # 2. Reject Carousel rounds (x-4 in all stages)
    if stage_curr[1] == 4 or stage_next[1] == 4:
        return False

    # 3. Reject PvE Neutral Creep rounds (x-7 in all stages: Krugs 2-7, Wolves 3-7, Raptors 4-7, Dragon 5-7, Elder 6-7)
    if stage_curr[1] == 7 or stage_next[1] == 7:
        return False

    # 4. Round type string checks (if metadata present)
    for r_type in (round_type_curr, round_type_next):
        if r_type:
            cleaned = str(r_type).strip().lower()
            if "creep" in cleaned or "pve" in cleaned or "carousel" in cleaned:
                return False

    return True


class TransitionPairCollate:
    """Collate function for transition snapshot pairs."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        anchor_boards = [item["anchor"]["board_champ_ids"] for item in batch]
        anchor_stars = [item["anchor"]["board_star_levels"] for item in batch]
        anchor_items = [item["anchor"]["board_item_ids"] for item in batch]
        anchor_traits = [item["anchor"]["board_traits"] for item in batch]
        anchor_states = [item["anchor"]["state_scalars"] for item in batch]

        pos_boards = [item["positive"]["board_champ_ids"] for item in batch]
        pos_stars = [item["positive"]["board_star_levels"] for item in batch]
        pos_items = [item["positive"]["board_item_ids"] for item in batch]
        pos_traits = [item["positive"]["board_traits"] for item in batch]
        pos_states = [item["positive"]["state_scalars"] for item in batch]

        return {
            "anchor": {
                "board_champ_ids": torch.tensor(np.array(anchor_boards), dtype=torch.long),
                "board_star_levels": torch.tensor(np.array(anchor_stars), dtype=torch.long),
                "board_item_ids": torch.tensor(np.array(anchor_items), dtype=torch.long),
                "board_traits": torch.tensor(np.array(anchor_traits), dtype=torch.float32),
                "state_scalars": torch.tensor(np.array(anchor_states), dtype=torch.float32),
            },
            "positive": {
                "board_champ_ids": torch.tensor(np.array(pos_boards), dtype=torch.long),
                "board_star_levels": torch.tensor(np.array(pos_stars), dtype=torch.long),
                "board_item_ids": torch.tensor(np.array(pos_items), dtype=torch.long),
                "board_traits": torch.tensor(np.array(pos_traits), dtype=torch.float32),
                "state_scalars": torch.tensor(np.array(pos_states), dtype=torch.float32),
            },
        }


class TransitionTrajectoryDataset(Dataset):
    """PyTorch Dataset loading consecutive snapshot pairs with strict PVP transition filtering.

    Loads match snapshots from CSVs or DataFrames, forms (s_t, s_{t+1}) pairs,
    and groups train/val partitions strictly by match_id.
    """

    def __init__(
        self,
        data: Sequence[dict[str, Any]] | pd.DataFrame | str | Path | None = None,
        data_dir: str | Path | None = None,
        vocab: ChampionVocabulary | None = None,
        item_vocab: ItemVocabulary | None = None,
        trait_vocab: TraitVocabulary | None = None,
        max_samples: int | None = None,
        allowed_tiers: Sequence[str] | None = ("CHALLENGER",),
    ) -> None:
        self.vocab = vocab or ChampionVocabulary()
        self.item_vocab = item_vocab or ItemVocabulary()
        self.trait_vocab = trait_vocab or TraitVocabulary()
        self.allowed_tiers = allowed_tiers
        self.pairs: list[dict[str, Any]] = []
        self.match_ids: list[str] = []

        target_data = data if data is not None else data_dir
        if target_data is not None:
            if isinstance(target_data, (str, Path)):
                self._load_from_path(Path(target_data))
            elif isinstance(target_data, pd.DataFrame):
                self._load_from_records(target_data.to_dict(orient="records"))
            elif isinstance(target_data, Sequence):
                self._load_from_records(target_data)

        if max_samples is not None and len(self.pairs) > max_samples:
            self.pairs = self.pairs[:max_samples]
            self.match_ids = self.match_ids[:max_samples]

    def _load_from_path(self, path: Path) -> None:
        if path.is_file():
            csv_files = [path]
        elif path.is_dir():
            csv_files = list(path.glob("*.csv"))
            if not csv_files and (path / "players").exists():
                csv_files = list((path / "players").glob("*.csv"))
            if not csv_files and (path / "tiers").exists():
                csv_files = list((path / "tiers").rglob("*.csv"))
            if not csv_files:
                csv_files = list(path.rglob("*.csv"))
        else:
            csv_files = []

        records: list[dict[str, Any]] = []
        for csv_file in csv_files:
            try:
                with open(csv_file, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append(row)
            except Exception:
                try:
                    with open(csv_file, mode="r", encoding="latin-1") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            records.append(row)
                except Exception:
                    continue

        self._load_from_records(records)

    def _load_from_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        target_tiers = (
            {t.strip().upper() for t in self.allowed_tiers if t.strip()}
            if self.allowed_tiers is not None
            else None
        )

        for r in records:
            match_id = str(r.get("match_id", "")).strip()
            focal_player = str(r.get("focal_player", "")).strip()
            if not match_id or not focal_player:
                continue

            if target_tiers:
                tier_cat = str(r.get("tier_category", "")).strip().upper()
                focal_tier = str(r.get("focal_tier", "")).strip().upper()
                avg_rating = str(r.get("avg_match_rating", "")).strip().upper()
                has_tier_info = bool(tier_cat or focal_tier or avg_rating)
                if has_tier_info:
                    matches = (
                        tier_cat in target_tiers
                        or any(t in focal_tier for t in target_tiers)
                        or any(t in avg_rating for t in target_tiers)
                    )
                    if not matches:
                        continue

            stage_str = str(r.get("round_stage", "")).strip()
            stage_tuple = parse_stage_string(stage_str)
            round_type = str(r.get("round_type", "pvp")).strip()

            state_raw = r.get("input_state_json", {})
            if isinstance(state_raw, str):
                try:
                    state_dict = json.loads(state_raw)
                except Exception:
                    state_dict = {}
            elif isinstance(state_raw, dict):
                state_dict = state_raw
            else:
                state_dict = {}

            focal_board = state_dict.get("focal_board", [])
            for u in focal_board:
                if isinstance(u, dict):
                    u_name = u.get("unit") or u.get("champion") or u.get("apiName")
                    if u_name:
                        self.vocab.add_champion(u_name)
                    for it in u.get("items", []) or []:
                        if it:
                            self.item_vocab.add_item(it)

            item = {
                "match_id": match_id,
                "focal_player": focal_player,
                "round_stage": stage_str,
                "stage_tuple": stage_tuple,
                "round_type": round_type,
                "focal_health": float(r.get("focal_health", 100) or 100),
                "focal_level": float(r.get("focal_level", 1) or 1),
                "focal_gold": float(r.get("focal_gold", 0) or 0),
                "focal_unit_count": float(r.get("focal_unit_count", 0) or 0),
                "focal_item_count": float(r.get("focal_item_count", 0) or 0),
                "label": int(r.get("label", 0) or 0),
                "focal_board": focal_board,
            }
            trajectories[(match_id, focal_player)].append(item)

        for (mid, player), traj in trajectories.items():
            if len(traj) < 2:
                continue
            traj.sort(key=lambda x: x["stage_tuple"])

            current_streak = 0.0
            for i in range(len(traj) - 1):
                anchor = traj[i]
                positive = traj[i + 1]

                # STRICT PVP TRANSITION FILTERING: Exclude creeps and carousels
                if not is_valid_pvp_transition(
                    stage_curr=anchor["stage_tuple"],
                    round_type_curr=anchor["round_type"],
                    stage_next=positive["stage_tuple"],
                    round_type_next=positive["round_type"],
                ):
                    continue

                combat_win = int(anchor["label"] == 1)

                anchor_copy = dict(anchor)
                anchor_copy["streak"] = current_streak
                anchor_copy["top4_label"] = 1
                anchor_copy["combat_label"] = combat_win
                anchor_copy["is_pvp"] = 1

                if combat_win == 1:
                    current_streak = current_streak + 1.0 if current_streak >= 0 else 1.0
                else:
                    current_streak = current_streak - 1.0 if current_streak <= 0 else -1.0

                pos_copy = dict(positive)
                pos_copy["streak"] = current_streak
                pos_copy["top4_label"] = 1
                pos_copy["combat_label"] = int(positive["label"] == 1)
                pos_copy["is_pvp"] = 1

                self.pairs.append({
                    "anchor": anchor_copy,
                    "positive": pos_copy,
                })
                self.match_ids.append(mid)

    def split_by_match_id(self, val_split: float = 0.15, seed: int = 42) -> tuple[Subset, Subset]:
        """Grouped train/validation partition strictly by match_id."""
        if val_split <= 0.0 or len(self.pairs) == 0:
            return Subset(self, list(range(len(self)))), Subset(self, [])

        unique_matches = list(set(self.match_ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_matches)

        num_val_matches = max(1, int(len(unique_matches) * val_split))
        val_match_set = set(unique_matches[:num_val_matches])

        train_indices: list[int] = []
        val_indices: list[int] = []

        for idx, mid in enumerate(self.match_ids):
            if mid in val_match_set:
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        if not train_indices:
            train_indices = val_indices[: len(val_indices) // 2]
            val_indices = val_indices[len(val_indices) // 2 :]

        return Subset(self, train_indices), Subset(self, val_indices)

    def _encode_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Encode snapshot into board grids, trait vector, and state scalars."""
        board_grid = np.zeros((4, 7), dtype=np.int64)
        star_grid = np.zeros((4, 7), dtype=np.int64)
        item_grid = np.zeros((4, 7, 3), dtype=np.int64)

        board_champ_names: list[str] = []
        occupied: set[tuple[int, int]] = set()

        for unit in snapshot.get("focal_board", []):
            if not isinstance(unit, dict):
                continue
            u_name = unit.get("unit") or unit.get("champion") or unit.get("apiName")
            u_tier = int(unit.get("tier", 1) or 1)
            u_loc = unit.get("loc")
            raw_items = unit.get("items", []) or []

            c_idx = self.vocab.encode(u_name)
            if u_name:
                board_champ_names.append(u_name)

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

        trait_vec = self.trait_vocab.compute_trait_vector(board_champ_names)
        stage_tuple = snapshot.get("stage_tuple", (2, 1))
        state_scalars = np.array(
            [
                snapshot.get("focal_health", 100.0) / 100.0,
                snapshot.get("focal_gold", 0.0) / 100.0,
                snapshot.get("focal_level", 1.0) / 10.0,
                snapshot.get("streak", 0.0) / 10.0,
                float(stage_tuple[0]) / 10.0,
                float(stage_tuple[1]) / 10.0,
                snapshot.get("focal_unit_count", 0.0) / 10.0,
                snapshot.get("focal_item_count", 0.0) / 10.0,
            ],
            dtype=np.float32,
        )

        return {
            "board_champ_ids": board_grid,
            "board_star_levels": star_grid,
            "board_item_ids": item_grid,
            "board_traits": trait_vec,
            "state_scalars": state_scalars,
            "top4_label": snapshot.get("top4_label", 1),
            "combat_label": snapshot.get("combat_label", 0),
            "is_pvp": snapshot.get("is_pvp", 1),
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        anchor_encoded = self._encode_snapshot(pair["anchor"])
        pos_encoded = self._encode_snapshot(pair["positive"])
        return {
            "anchor": anchor_encoded,
            "positive": pos_encoded,
        }


def extract_latent_transition_pairs(
    trunk: MultiModalFusionTrunk,
    dataset: Dataset | Sequence[dict[str, Any]],
    batch_size: int = 512,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and cache frozen 320D latent vectors (s_t, s_{t+1}) from the trunk in torch.no_grad().

    Precomputes full trajectory pairs into memory for ultra-fast training epochs.
    """
    trunk = trunk.to(device)
    trunk.freeze()
    trunk.eval()

    collate_fn = TransitionPairCollate()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    states_t_list: list[torch.Tensor] = []
    states_next_list: list[torch.Tensor] = []

    iterator = loader
    if verbose and tqdm is not None:
        iterator = tqdm(loader, desc="Extracting 320D Latents", unit="batch")

    with torch.no_grad():
        for batch in iterator:
            anchor = {k: v.to(device) for k, v in batch["anchor"].items()}
            pos = {k: v.to(device) for k, v in batch["positive"].items()}

            s_t = trunk(
                board_champ_ids=anchor["board_champ_ids"],
                board_star_levels=anchor["board_star_levels"],
                board_item_ids=anchor["board_item_ids"],
                board_traits=anchor["board_traits"],
                state_scalars=anchor["state_scalars"],
            )

            s_next = trunk(
                board_champ_ids=pos["board_champ_ids"],
                board_star_levels=pos["board_star_levels"],
                board_item_ids=pos["board_item_ids"],
                board_traits=pos["board_traits"],
                state_scalars=pos["state_scalars"],
            )

            states_t_list.append(s_t.cpu())
            states_next_list.append(s_next.cpu())

    if states_t_list:
        states_t = torch.cat(states_t_list, dim=0)
        states_next = torch.cat(states_next_list, dim=0)
    else:
        states_t = torch.empty((0, trunk.fused_dim), dtype=torch.float32)
        states_next = torch.empty((0, trunk.fused_dim), dtype=torch.float32)

    return states_t, states_next


class TransitionPredictorTrainer:
    """End-to-End Trainer for Phase 2: State Transition Predictor.

    Orchestrates:
    - Pre-trained Trunk feature extraction
    - Residual MLP training with Dual-Objective Huber + Cosine loss
    - Strict match-grouped validation
    - Real-time WandB logging & checkpointing
    """

    def __init__(
        self,
        predictor: StateTransitionPredictor | None = None,
        trunk: MultiModalFusionTrunk | None = None,
        input_dim: int = 320,
        hidden_dim: int = 512,
        output_dim: int = 320,
        num_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_cosine: float = 0.5,
        huber_beta: float = 1.0,
        log_interval: int = 20,
        use_wandb: bool = False,
        wandb_project: str = "tft-embeddings",
        wandb_run_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.trunk = trunk
        if self.trunk is not None:
            self.trunk = self.trunk.to(self.device)
            self.trunk.freeze()
            self.trunk.eval()

        if predictor is not None:
            self.predictor = predictor.to(self.device)
        else:
            self.predictor = StateTransitionPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_residual_delta=True,
            ).to(self.device)

        self.criterion = StateTransitionLoss(
            lambda_cosine=lambda_cosine,
            huber_beta=huber_beta,
        ).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_cosine = lambda_cosine
        self.huber_beta = huber_beta
        self.log_interval = max(1, log_interval)

        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_entity = wandb_entity
        self.wandb_group = wandb_group
        self._wandb_run = None
        self.global_step = 0
        self.history: list[dict[str, Any]] = []

    def _init_wandb_if_needed(self, total_epochs: int, batch_size: int) -> None:
        if not self.use_wandb or self._wandb_run is not None:
            return

        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name or f"transition-predictor-{int(time.time())}",
                entity=self.wandb_entity,
                group=self.wandb_group or "transition-predictor",
                config={
                    "input_dim": self.predictor.input_dim,
                    "hidden_dim": self.predictor.hidden_dim,
                    "output_dim": self.predictor.output_dim,
                    "num_layers": self.predictor.num_layers,
                    "dropout": self.predictor.dropout,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "lambda_cosine": self.lambda_cosine,
                    "huber_beta": self.huber_beta,
                    "total_epochs": total_epochs,
                    "batch_size": batch_size,
                    "device": str(self.device),
                },
                reinit=True,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}. Running without WandB logging.")
            self.use_wandb = False

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = 1,
        total_epochs: int = 1,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Execute one training epoch over latent transition batches."""
        self.predictor.train()
        accum_metrics: dict[str, float] = {}
        total_batches = len(train_loader)
        current_lr = self.optimizer.param_groups[0]["lr"]

        iterator = train_loader
        pbar = None
        if verbose and tqdm is not None:
            pbar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Train",
                unit="batch",
                leave=False,
            )
            iterator = pbar

        for batch_idx, batch in enumerate(iterator, start=1):
            if isinstance(batch, (list, tuple)):
                s_t, s_next = batch[0].to(self.device), batch[1].to(self.device)
            else:
                s_t, s_next = batch["s_t"].to(self.device), batch["s_next"].to(self.device)

            pred_next = self.predictor(s_t)
            loss, metrics = self.criterion(pred_next=pred_next, true_next=s_next)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=5.0)
            self.optimizer.step()

            self.global_step += 1

            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v

            if pbar is not None:
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "cos_sim": f"{metrics['cosine_similarity']:.4f}",
                    "huber": f"{metrics['huber_loss']:.4f}",
                })

            if self.use_wandb and self._wandb_run is not None and (self.global_step % self.log_interval == 0):
                import wandb

                wandb.log(
                    {
                        "train/step_total_loss": metrics["total_loss"],
                        "train/step_huber_loss": metrics["huber_loss"],
                        "train/step_cosine_loss": metrics["cosine_loss"],
                        "train/step_cosine_similarity": metrics["cosine_similarity"],
                        "train/step_l1_error": metrics["l1_error"],
                        "train/step_l2_error": metrics["l2_error"],
                        "train/grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                        "train/lr": current_lr,
                        "global_step": self.global_step,
                    },
                    step=self.global_step,
                )

        mean_metrics = {k: v / max(1, total_batches) for k, v in accum_metrics.items()}
        return mean_metrics

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        epoch: int = 1,
        total_epochs: int = 1,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Evaluate predictor performance on validation partition."""
        self.predictor.eval()
        accum_metrics: dict[str, float] = {}
        total_batches = len(val_loader)

        iterator = val_loader
        pbar = None
        if verbose and tqdm is not None:
            pbar = tqdm(
                val_loader,
                desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Val",
                unit="batch",
                leave=False,
            )
            iterator = pbar

        for batch in iterator:
            if isinstance(batch, (list, tuple)):
                s_t, s_next = batch[0].to(self.device), batch[1].to(self.device)
            else:
                s_t, s_next = batch["s_t"].to(self.device), batch["s_next"].to(self.device)

            pred_next = self.predictor(s_t)
            _, metrics = self.criterion(pred_next=pred_next, true_next=s_next)

            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v

        mean_metrics = {k: v / max(1, total_batches) for k, v in accum_metrics.items()}
        return mean_metrics

    def fit(
        self,
        dataset: Dataset | Sequence[dict[str, Any]],
        val_dataset: Dataset | None = None,
        val_split: float = 0.15,
        epochs: int = 15,
        batch_size: int = 512,
        output_dir: str | Path = "models/transition_predictor",
        lr_scheduler_type: str = "cosine",
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Execute complete training & validation pipeline with latent caching."""
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._init_wandb_if_needed(total_epochs=epochs, batch_size=batch_size)

        # 1. Check if dataset is raw snapshots or pre-extracted tensors
        if isinstance(dataset, TensorDataset):
            train_ds = dataset
            val_ds = val_dataset
        elif isinstance(dataset, TransitionTrajectoryDataset) or isinstance(dataset, TFTPretrainDataset):
            if val_dataset is None:
                train_sub, val_sub = dataset.split_by_match_id(val_split=val_split)
            else:
                train_sub, val_sub = dataset, val_dataset

            if self.trunk is None:
                raise ValueError("Frozen MultiModalFusionTrunk is required when training from raw trajectory datasets.")

            if verbose:
                print(f" [+] Pre-extracting 320D latents from {len(train_sub)} train and {len(val_sub)} val snapshot pairs...")

            train_s_t, train_s_next = extract_latent_transition_pairs(
                self.trunk, train_sub, batch_size=batch_size, device=self.device, verbose=verbose
            )
            val_s_t, val_s_next = extract_latent_transition_pairs(
                self.trunk, val_sub, batch_size=batch_size, device=self.device, verbose=verbose
            )

            train_ds = TensorDataset(train_s_t, train_s_next)
            val_ds = TensorDataset(val_s_t, val_s_next) if len(val_s_t) > 0 else None
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

        scheduler = None
        if lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        best_val_cos_sim = -1.0
        best_val_loss = float("inf")
        best_metrics: dict[str, Any] = {}

        if verbose:
            print("\n" + "=" * 80)
            print(f" [STATE TRANSITION PREDICTOR] Starting Training: {epochs} Epochs | Batch Size: {batch_size}")
            print(f"  Train Samples: {len(train_ds)} | Val Samples: {len(val_ds) if val_ds else 0}")
            print(f"  Lambda Cosine: {self.lambda_cosine} | Huber Beta: {self.huber_beta} | LR: {self.lr}")
            print("=" * 80)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader, epoch=epoch, total_epochs=epochs, verbose=verbose)

            if val_loader is not None and len(val_loader) > 0:
                val_m = self.evaluate(val_loader, epoch=epoch, total_epochs=epochs, verbose=verbose)
            else:
                val_m = dict(train_m)

            if scheduler is not None:
                scheduler.step()

            epoch_time = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_m["total_loss"],
                "train_huber": train_m["huber_loss"],
                "train_cosine_loss": train_m["cosine_loss"],
                "train_cos_sim": train_m["cosine_similarity"],
                "train_l1_error": train_m["l1_error"],
                "val_loss": val_m["total_loss"],
                "val_huber": val_m["huber_loss"],
                "val_cosine_loss": val_m["cosine_loss"],
                "val_cos_sim": val_m["cosine_similarity"],
                "val_l1_error": val_m["l1_error"],
                "lr": current_lr,
                "epoch_time_sec": round(epoch_time, 2),
            }
            self.history.append(epoch_log)

            if verbose:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {train_m['total_loss']:.4f} (Cos: {train_m['cosine_similarity']:.4f}, Huber: {train_m['huber_loss']:.4f}) | "
                    f"Val Loss: {val_m['total_loss']:.4f} (Cos: {val_m['cosine_similarity']:.4f}, Huber: {val_m['huber_loss']:.4f}) | "
                    f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s"
                )

            if self.use_wandb and self._wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "epoch/train_total_loss": train_m["total_loss"],
                        "epoch/train_huber_loss": train_m["huber_loss"],
                        "epoch/train_cosine_loss": train_m["cosine_loss"],
                        "epoch/train_cosine_similarity": train_m["cosine_similarity"],
                        "epoch/train_l1_error": train_m["l1_error"],
                        "epoch/val_total_loss": val_m["total_loss"],
                        "epoch/val_huber_loss": val_m["huber_loss"],
                        "epoch/val_cosine_loss": val_m["cosine_loss"],
                        "epoch/val_cosine_similarity": val_m["cosine_similarity"],
                        "epoch/val_l1_error": val_m["l1_error"],
                        "epoch/lr": current_lr,
                    },
                    step=self.global_step,
                )

            # Checkpoint winning model based on validation cosine similarity
            if val_m["cosine_similarity"] > best_val_cos_sim:
                best_val_cos_sim = val_m["cosine_similarity"]
                best_val_loss = val_m["total_loss"]
                best_metrics = dict(val_m)
                best_path = output_dir / "predictor_best.pt"
                self.predictor.save_predictor(best_path, extra_info={"epoch": epoch, "best_val_metrics": val_m})
                if verbose:
                    print(f"  [*] New Best Model Saved! Val Cos Sim: {best_val_cos_sim:.4f} -> {best_path}")

        # Save final model & summary
        final_path = output_dir / "predictor_final.pt"
        self.predictor.save_predictor(final_path, extra_info={"final_metrics": self.history[-1] if self.history else {}})

        total_time = round(time.time() - start_time, 2)
        summary = {
            "total_time_sec": total_time,
            "total_epochs": epochs,
            "best_val_cosine_similarity": best_val_cos_sim,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "final_metrics": self.history[-1] if self.history else {},
            "history": self.history,
        }

        summary_path = output_dir / "predictor_summary.json"
        with open(summary_path, mode="w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.run.summary["best_val_cosine_similarity"] = best_val_cos_sim
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["total_time_sec"] = total_time
            wandb.finish()

        if verbose:
            print("\n" + "=" * 80)
            print(f" [+] Training Completed in {total_time}s!")
            print(f"     Best Val Cosine Similarity: {best_val_cos_sim:.4f} | Best Val Loss: {best_val_loss:.4f}")
            print(f"     Artifacts Saved To: {output_dir.resolve()}")
            print("=" * 80 + "\n")

        return summary


def create_synthetic_transition_dataset(
    num_matches: int = 30,
    rounds_per_match: int = 15,
    vocab: ChampionVocabulary | None = None,
    item_vocab: ItemVocabulary | None = None,
    trait_vocab: TraitVocabulary | None = None,
) -> TransitionTrajectoryDataset:
    """Generate a clean synthetic trajectory dataset for testing transition prediction."""
    vocab = vocab or ChampionVocabulary()
    item_vocab = item_vocab or ItemVocabulary()
    trait_vocab = trait_vocab or TraitVocabulary()

    records: list[dict[str, Any]] = []

    for m in range(num_matches):
        match_id = f"synth_match_{m:03d}"
        for p in range(8):
            focal_player = f"player_{p}"
            health = 100.0
            gold = 10.0
            level = 1.0

            for r_idx in range(rounds_per_match):
                stage = 2 + (r_idx // 5)
                round_num = 1 + (r_idx % 5)
                stage_str = f"{stage}-{round_num}"

                gold = min(100.0, gold + 5.0)
                level = min(10.0, 1.0 + (stage * 1.2))
                health = max(0.0, health - (1.5 if r_idx % 2 == 0 else 0.0))

                num_units = min(int(level), 9)
                focal_board = []
                for u_i in range(num_units):
                    c_name = f"TFT13_Champ_{((u_i + m + p) % 30) + 1}"
                    vocab.add_champion(c_name)
                    items = []
                    if u_i == 0:
                        i_name = f"TFT_Item_{((u_i + p) % 20) + 1}"
                        item_vocab.add_item(i_name)
                        items.append(i_name)
                    focal_board.append({
                        "unit": c_name,
                        "tier": 2 if u_i == 0 else 1,
                        "loc": f"A_{u_i + 1}",
                        "items": items,
                    })

                records.append({
                    "match_id": match_id,
                    "focal_player": focal_player,
                    "round_stage": stage_str,
                    "round_type": "pvp",
                    "focal_health": health,
                    "focal_gold": gold,
                    "focal_level": level,
                    "focal_unit_count": float(num_units),
                    "focal_item_count": 1.0,
                    "label": 1 if (r_idx + p) % 2 == 0 else 0,
                    "input_state_json": json.dumps({"focal_board": focal_board}),
                })

    return TransitionTrajectoryDataset(
        data=records,
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
    )
