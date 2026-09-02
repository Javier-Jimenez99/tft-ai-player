"""Dataset loader, trajectory builder, match-grouped splitting, and batch collators for Tri-Objective Multi-Task Pre-training."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from .vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary


def parse_loc_to_row_col(loc_str: str | None) -> tuple[int, int] | None:
    """Parse hex board location strings like 'A1'..'D7' or 'C_6' into (row, col) in [0..3, 0..6]."""
    if not loc_str or not isinstance(loc_str, str):
        return None
    cleaned = loc_str.strip().upper()
    match = re.search(r"([A-D])_?([1-7])", cleaned)
    if match:
        row_char, col_char = match.groups()
        row = ord(row_char) - ord("A")
        col = int(col_char) - 1
        if 0 <= row < 4 and 0 <= col < 7:
            return row, col
    return None


def parse_stage_string(stage_str: str | None) -> tuple[int, int]:
    """Parse round stage strings like '2-1', '3-2', '5-6' into tuple (stage, round)."""
    if not stage_str or not isinstance(stage_str, str):
        return 0, 0
    parts = stage_str.strip().split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return 0, 0
    return 0, 0


def is_pvp_round(stage_tuple: tuple[int, int], round_type_str: str | None = None) -> int:
    """Determine whether a snapshot represents genuine PVP combat (excluding PvE creeps and carousel)."""
    stage, round_in_stage = stage_tuple
    if stage < 2:
        return 0
    # Carousel rounds (x-4 in stages 2-4, 5-4, 6-4)
    if round_in_stage == 4:
        return 0
    # Creep / neutral monster rounds (x-7 in stages 2-6)
    if round_in_stage == 7:
        return 0
    if round_type_str:
        cleaned = str(round_type_str).strip().lower()
        if "pve" in cleaned or "creep" in cleaned or "carousel" in cleaned:
            return 0
        if "pvp" in cleaned:
            return 1
    return 1


class TFTPretrainDataset(Dataset):
    """PyTorch Dataset grouping match snapshots into consecutive temporal pairs (s_t, s_{t+1}).

    Encodes multi-modal states and tri-objective targets:
    1. Macro Target: Match Top-4 Placement (0 or 1)
    2. Micro Target: Immediate Combat Round Win (0 or 1, with is_pvp flag)
    3. Flow Target: Positive next snapshot pairing
    """

    def __init__(
        self,
        data: Sequence[dict[str, Any]] | pd.DataFrame | str | Path | None = None,
        data_dir: str | Path | None = None,
        vocab: ChampionVocabulary | None = None,
        item_vocab: ItemVocabulary | None = None,
        trait_vocab: TraitVocabulary | None = None,
        max_samples: int | None = None,
        min_trajectory_len: int = 2,
    ) -> None:
        self.vocab = vocab or ChampionVocabulary()
        self.item_vocab = item_vocab or ItemVocabulary()
        self.trait_vocab = trait_vocab or TraitVocabulary()
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

        for r in records:
            match_id = str(r.get("match_id", "")).strip()
            focal_player = str(r.get("focal_player", "")).strip()
            if not match_id or not focal_player:
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
            # Pre-scan vocabularies to register all champion and item tokens in the dataset
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
                "is_pvp": is_pvp_round(stage_tuple, round_type),
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

            last_round = traj[-1]
            last_stage = last_round["stage_tuple"]
            last_health = last_round["focal_health"]

            # Macro Ground Truth: Top-4 vs Bot-4
            is_top4 = int(last_stage >= (5, 5) or (len(traj) >= 18 and last_health > 0))

            current_streak = 0.0
            for i in range(len(traj) - 1):
                anchor = traj[i]
                positive = traj[i + 1]

                # Micro Ground Truth: Immediate Combat Round Win (0 or 1)
                combat_win = int(anchor["label"] == 1)

                # Pre-combat streak: streak entering round i (before round i combat occurs)
                anchor_copy = dict(anchor)
                anchor_copy["streak"] = current_streak
                anchor_copy["top4_label"] = is_top4
                anchor_copy["combat_label"] = combat_win

                # Update streak after round i combat finishes
                if combat_win == 1:
                    current_streak = current_streak + 1.0 if current_streak >= 0 else 1.0
                else:
                    current_streak = current_streak - 1.0 if current_streak <= 0 else -1.0

                # Positive snapshot is state entering round i+1 (carries the updated streak)
                pos_copy = dict(positive)
                pos_copy["streak"] = current_streak
                pos_copy["top4_label"] = is_top4
                pos_copy["combat_label"] = int(positive["label"] == 1)

                self.pairs.append({
                    "anchor": anchor_copy,
                    "positive": pos_copy,
                })
                self.match_ids.append(mid)

    def split_by_match_id(self, val_split: float = 0.15, seed: int = 42) -> tuple[Subset, Subset]:
        """Grouped train/validation partition by unique match_id to strictly prevent intra-match temporal data leakage."""
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

        # Fallback if all fell into one split
        if not train_indices:
            train_indices = val_indices[: len(val_indices) // 2]
            val_indices = val_indices[len(val_indices) // 2 :]

        return Subset(self, train_indices), Subset(self, val_indices)

    def _encode_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Convert a snapshot dict into 4x7 board grid, star grid, item grid, traits vector, and state scalars."""
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

        # Compute active trait synergies vector
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
            "top4_label": snapshot.get("top4_label", 0),
            "combat_label": int(snapshot.get("combat_label", snapshot.get("label", 0)) == 1),
            "is_pvp": int(snapshot.get("is_pvp", 1)),
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


class SnapshotPairCollate:
    """Collate function assembling consecutive snapshot pairs and Tri-Objective targets."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        anchor_boards = [item["anchor"]["board_champ_ids"] for item in batch]
        anchor_stars = [item["anchor"]["board_star_levels"] for item in batch]
        anchor_items = [item["anchor"]["board_item_ids"] for item in batch]
        anchor_traits = [item["anchor"]["board_traits"] for item in batch]
        anchor_states = [item["anchor"]["state_scalars"] for item in batch]
        anchor_targets = [item["anchor"]["top4_label"] for item in batch]
        anchor_combat_labels = [item["anchor"]["combat_label"] for item in batch]
        anchor_is_pvp = [item["anchor"]["is_pvp"] for item in batch]

        pos_boards = [item["positive"]["board_champ_ids"] for item in batch]
        pos_stars = [item["positive"]["board_star_levels"] for item in batch]
        pos_items = [item["positive"]["board_item_ids"] for item in batch]
        pos_traits = [item["positive"]["board_traits"] for item in batch]
        pos_states = [item["positive"]["state_scalars"] for item in batch]
        pos_targets = [item["positive"]["top4_label"] for item in batch]
        pos_combat_labels = [item["positive"]["combat_label"] for item in batch]
        pos_is_pvp = [item["positive"]["is_pvp"] for item in batch]

        return {
            "anchor": {
                "board_champ_ids": torch.tensor(np.array(anchor_boards), dtype=torch.long),
                "board_star_levels": torch.tensor(np.array(anchor_stars), dtype=torch.long),
                "board_item_ids": torch.tensor(np.array(anchor_items), dtype=torch.long),
                "board_traits": torch.tensor(np.array(anchor_traits), dtype=torch.float32),
                "state_scalars": torch.tensor(np.array(anchor_states), dtype=torch.float32),
                "value_targets": torch.tensor(anchor_targets, dtype=torch.long),
                "combat_targets": torch.tensor(anchor_combat_labels, dtype=torch.float32),
                "is_pvp_mask": torch.tensor(anchor_is_pvp, dtype=torch.float32),
            },
            "positive": {
                "board_champ_ids": torch.tensor(np.array(pos_boards), dtype=torch.long),
                "board_star_levels": torch.tensor(np.array(pos_stars), dtype=torch.long),
                "board_item_ids": torch.tensor(np.array(pos_items), dtype=torch.long),
                "board_traits": torch.tensor(np.array(pos_traits), dtype=torch.float32),
                "state_scalars": torch.tensor(np.array(pos_states), dtype=torch.float32),
                "value_targets": torch.tensor(pos_targets, dtype=torch.long),
                "combat_targets": torch.tensor(pos_combat_labels, dtype=torch.float32),
                "is_pvp_mask": torch.tensor(pos_is_pvp, dtype=torch.float32),
            },
        }


def create_synthetic_trajectory_dataset(
    num_matches: int = 20,
    rounds_per_match: int = 15,
    vocab: ChampionVocabulary | None = None,
    item_vocab: ItemVocabulary | None = None,
    trait_vocab: TraitVocabulary | None = None,
) -> TFTPretrainDataset:
    vocab = vocab or ChampionVocabulary()
    item_vocab = item_vocab or ItemVocabulary()
    trait_vocab = trait_vocab or TraitVocabulary()

    sample_champions = [
        "DA_18_Cassiopeia", "DA_18_Camille", "DA_18_Elise", "DA_Karma18",
        "DA_18_Rengar", "DA_18_Rammus", "DA_Vi18", "DA_18_Alistar", "DA_18_Zyra"
    ]
    sample_items = [
        "DA_InfinityEdge", "DA_GuinsoosRageblade", "DA_ArchangelsStaff",
        "DA_WarmogsArmor", "DA_Component_RecurveBow"
    ]

    for c in sample_champions:
        vocab.add_champion(c)
    for it in sample_items:
        item_vocab.add_item(it)

    records: list[dict[str, Any]] = []
    for m in range(num_matches):
        match_id = f"synth_match_{m:04d}"
        player = f"Player_{m % 4}"
        hp = 100
        gold = 20
        level = 3
        is_high_performer = (m % 2 == 0)

        for r_idx in range(rounds_per_match):
            stage = 2 + (r_idx // 5)
            round_in_stage = 1 + (r_idx % 5)
            stage_str = f"{stage}-{round_in_stage}"

            if is_high_performer:
                outcome = "victory"
                label = 1
                hp = max(10, hp - np.random.randint(0, 3))
            else:
                outcome = "defeat"
                label = 0
                hp = max(0, hp - np.random.randint(2, 8))

            gold = min(100, gold + 5)
            if r_idx % 4 == 0:
                level = min(9, level + 1)

            num_units = min(level, 7)
            chosen_champs = np.random.choice(sample_champions, size=min(num_units, len(sample_champions)), replace=False)
            board_units = []
            rows = ["A", "B", "C", "D"]
            for idx, cname in enumerate(chosen_champs):
                r_char = rows[idx % 4]
                col_char = (idx % 7) + 1
                u_items = [sample_items[0], sample_items[1]] if idx == 0 and is_high_performer else []
                board_units.append({
                    "unit": cname,
                    "tier": 2 if is_high_performer and idx == 0 else 1,
                    "loc": f"{r_char}{col_char}",
                    "items": u_items,
                })

            records.append({
                "match_id": match_id,
                "focal_player": player,
                "round_stage": stage_str,
                "round_type": "pvp" if round_in_stage not in (4, 7) else "pve",
                "focal_health": hp,
                "focal_level": level,
                "focal_gold": gold,
                "focal_unit_count": len(board_units),
                "focal_item_count": sum(len(u["items"]) for u in board_units),
                "outcome": outcome,
                "label": label,
                "input_state_json": json.dumps({"focal_board": board_units}),
            })

    return TFTPretrainDataset(data=records, vocab=vocab, item_vocab=item_vocab, trait_vocab=trait_vocab)
