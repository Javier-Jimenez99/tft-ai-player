"""Dataset loader for Board Quality and Placement Prediction.

Extracts intermediate board states from player trajectories and labels them
with the final tournament finish placement and Top-4 qualification.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tft_ai_player.embeddings.dataset import parse_loc_to_row_col, parse_stage_string
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary


def infer_placement_from_trajectory(
    last_stage: tuple[int, int],
    last_health: float = 0.0,
    last_outcome: str | None = None,
) -> float:
    """Infer final match placement from final survival round, health, and outcome.

    Calibrated empirically to authentic TFT lobby elimination distributions:
    - Victory in deep stage (>= 6-1) -> 1st place (1.0)
    - Victory in stage 5-5+ -> 1.5
    - Defeat in stage >= 6-5 -> 2.0 (runner-up final showdown)
    - Defeat in stage 6-1 to 6-4 -> 3.0 (top-3 podium)
    - Defeat in stage 5-5 to 5-6 -> 4.0 (top-4 bubble winner)
    - Defeat in stage 5-1 to 5-4 -> 5.0 (5th place early stage 5 out)
    - Defeat in stage 4-4 to 4-6 -> 6.0 (6th place late stage 4 out)
    - Defeat in stage 4-1 to 4-3 -> 7.0 (7th place early stage 4 out)
    - Defeat before stage 4-1 -> 8.0 (8th place fast eighth)
    """
    outcome_str = str(last_outcome).strip().lower() if last_outcome else ""
    is_victory = outcome_str == "victory"

    if is_victory and last_stage >= (6, 1):
        return 1.0
    if is_victory and last_stage >= (5, 5):
        return 1.5

    # Defeat progression (elimination stage)
    if last_stage >= (6, 5):
        return 2.0
    if last_stage >= (6, 1):
        return 3.0
    if last_stage >= (5, 5):
        return 4.0
    if last_stage >= (5, 1):
        return 5.0
    if last_stage >= (4, 4):
        return 6.0
    if last_stage >= (4, 1):
        return 7.0
    return 8.0


class BoardPlacementDataset(Dataset):
    """PyTorch Dataset mapping board snapshot observations to final placement outcomes."""

    def __init__(
        self,
        data_dir: str | Path = "D:/tft-winner-data/set18/players",
        vocab: ChampionVocabulary | None = None,
        item_vocab: ItemVocabulary | None = None,
        trait_vocab: TraitVocabulary | None = None,
        max_samples: int | None = None,
        max_files: int | None = None,
        allowed_tiers: Sequence[str] | None = None,
    ) -> None:
        if vocab is not None:
            self.vocab = vocab
        elif Path("models/trunk/vocab.json").exists():
            self.vocab = ChampionVocabulary.load("models/trunk/vocab.json")
        else:
            self.vocab = ChampionVocabulary()

        if item_vocab is not None:
            self.item_vocab = item_vocab
        elif Path("models/trunk/item_vocab.json").exists():
            self.item_vocab = ItemVocabulary.load("models/trunk/item_vocab.json")
        else:
            self.item_vocab = ItemVocabulary()

        if trait_vocab is not None:
            self.trait_vocab = trait_vocab
        elif Path("models/trunk/trait_vocab.json").exists():
            self.trait_vocab = TraitVocabulary.load("models/trunk/trait_vocab.json")
        else:
            self.trait_vocab = TraitVocabulary()

        self.samples: list[dict[str, Any]] = []

        path = Path(data_dir)
        if path.exists():
            self._load_from_path(path, allowed_tiers=allowed_tiers, max_samples=max_samples, max_files=max_files)

    def _load_from_path(
        self,
        path: Path,
        allowed_tiers: Sequence[str] | None = None,
        max_samples: int | None = None,
        max_files: int | None = None,
    ) -> None:
        csv_files = list(path.glob("*.csv")) if path.is_dir() else [path]
        if not csv_files and (path / "players").exists():
            csv_files = list((path / "players").glob("*.csv"))
        if max_files:
            csv_files = csv_files[:max_files]

        trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        target_tiers = (
            {t.strip().upper() for t in allowed_tiers if t.strip()}
            if allowed_tiers is not None
            else None
        )

        for csv_file in csv_files:
            try:
                with open(csv_file, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        match_id = str(r.get("match_id", "")).strip()
                        focal_player = str(r.get("focal_player", "")).strip()
                        if not match_id or not focal_player:
                            continue

                        if target_tiers:
                            f_tier = str(r.get("focal_tier", "")).strip().upper()
                            if not any(t in f_tier for t in target_tiers):
                                continue

                        stage_str = str(r.get("round_stage", "")).strip()
                        stage_tuple = parse_stage_string(stage_str)

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

                        trajectories[(match_id, focal_player)].append({
                            "round_stage": stage_str,
                            "stage_tuple": stage_tuple,
                            "outcome": str(r.get("outcome", "")).strip().lower(),
                            "focal_health": float(r.get("focal_health", 100) or 100),
                            "focal_level": float(r.get("focal_level", 1) or 1),
                            "focal_gold": float(r.get("focal_gold", 0) or 0),
                            "focal_unit_count": float(r.get("focal_unit_count", 0) or 0),
                            "focal_item_count": float(r.get("focal_item_count", 0) or 0),
                            "focal_board": focal_board,
                        })
            except Exception:
                continue

            if max_samples and sum(len(v) for v in trajectories.values()) >= max_samples:
                break

        # Process trajectories and associate final placement
        for (mid, player), traj in trajectories.items():
            if len(traj) < 2:
                continue
            traj.sort(key=lambda x: x["stage_tuple"])
            last_round = traj[-1]
            final_place = infer_placement_from_trajectory(
                last_stage=last_round["stage_tuple"],
                last_health=last_round["focal_health"],
                last_outcome=last_round.get("outcome"),
            )
            is_top4 = 1 if final_place <= 4.0 else 0

            for obs in traj:
                self.samples.append({
                    "board": obs["focal_board"],
                    "stage_tuple": obs["stage_tuple"],
                    "health": obs["focal_health"],
                    "level": obs["focal_level"],
                    "gold": obs["focal_gold"],
                    "unit_count": obs["focal_unit_count"],
                    "item_count": obs["focal_item_count"],
                    "streak": 0.0,
                    "target_placement": float(final_place),
                    "target_top4": int(is_top4),
                })
                if max_samples and len(self.samples) >= max_samples:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]

        board_champs = torch.zeros(28, dtype=torch.long)
        board_stars = torch.zeros(28, dtype=torch.long)
        board_items = torch.zeros((28, 3), dtype=torch.long)
        board_champ_names: list[str] = []

        for u in item["board"]:
            if not isinstance(u, dict):
                continue
            loc = u.get("location") or u.get("hex") or u.get("cell") or 0
            if isinstance(loc, int):
                hex_idx = loc
            elif isinstance(loc, str):
                parsed = parse_loc_to_row_col(loc)
                hex_idx = parsed[0] * 7 + parsed[1] if parsed else -1
            else:
                hex_idx = -1

            c_name = u.get("unit") or u.get("champion") or u.get("apiName") or ""
            if c_name:
                board_champ_names.append(c_name)

            if 0 <= hex_idx < 28:
                board_champs[hex_idx] = self.vocab.encode(c_name)
                board_stars[hex_idx] = int(u.get("tier") or u.get("star_level") or 1)
                for it_i, it_name in enumerate(u.get("items", [])[:3]):
                    board_items[hex_idx, it_i] = self.item_vocab.encode(it_name)

        if hasattr(self.trait_vocab, "compute_trait_vector"):
            tv = self.trait_vocab.compute_trait_vector(board_champ_names)
            board_traits = tv if isinstance(tv, torch.Tensor) else torch.tensor(tv, dtype=torch.float32)
        else:
            num_traits = max(1, len(self.trait_vocab))
            board_traits = torch.zeros(num_traits, dtype=torch.float32)

        stage_s = item["stage_tuple"]
        scalars = torch.tensor(
            [
                item["health"] / 100.0,
                item["gold"] / 100.0,
                item["level"] / 10.0,
                item["streak"] / 10.0,
                float(stage_s[0]) / 10.0,
                float(stage_s[1]) / 10.0,
                item["unit_count"] / 10.0,
                item["item_count"] / 10.0,
            ],
            dtype=torch.float32,
        )

        return {
            "board_champ_ids": board_champs,
            "board_star_levels": board_stars,
            "board_item_ids": board_items,
            "board_traits": board_traits,
            "state_scalars": scalars,
            "target_placement": torch.tensor([item["target_placement"]], dtype=torch.float32),
            "target_top4": torch.tensor(item["target_top4"], dtype=torch.long),
        }
