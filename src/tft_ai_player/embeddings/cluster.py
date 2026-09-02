"""Composition Archetype Extraction & Latent Clustering (Z-Index) for TFT.

This module provides end-to-end tooling to:
1. Curate endgame boards of winning players (Top-4, Stage >= 5-1).
2. Extract isolated 256D board embeddings via the frozen MultiModalFusionTrunk (ignoring StateMLP).
3. Partition the latent space into canonical archetype centroids: the Z-Index (Z = {z_1, ..., z_K}).
4. Profile and interpret each archetype (units, items, trait tiers, representative boards).
5. Generate dimensional reduction plots (UMAP/t-SNE/PCA) and log metrics/tables to Weights & Biases.
6. Export z_index.pt and cluster_profiles.json artifacts for downstream decision making.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .dataset import parse_loc_to_row_col, parse_stage_string
from .model import MultiModalFusionTrunk
from .vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary

logger = logging.getLogger(__name__)


@dataclass
class CuratedEndgameBoard:
    """Structured representation of a curated late-game board snapshot."""

    board_champ_ids: np.ndarray  # (4, 7) int64
    board_star_levels: np.ndarray  # (4, 7) int64
    board_item_ids: np.ndarray  # (4, 7, 3) int64
    board_traits: np.ndarray  # (num_traits,) float32
    unit_names: list[str] = field(default_factory=list)
    unit_tiers: list[int] = field(default_factory=list)
    unit_items: list[list[str]] = field(default_factory=list)
    unit_locs: list[str] = field(default_factory=list)
    active_traits: dict[str, int] = field(default_factory=dict)
    match_id: str = ""
    focal_player: str = ""
    round_stage: str = ""
    stage_tuple: tuple[int, int] = (5, 1)
    placement: int = 1
    health: float = 100.0


@dataclass
class ArchetypeProfile:
    """Summary profile of an extracted composition archetype cluster."""

    cluster_id: int
    name: str
    size: int
    percentage: float
    top_units: list[dict[str, Any]]
    dominant_traits: list[dict[str, Any]]
    top_items: list[dict[str, Any]]
    avg_placement: float
    representative_boards: list[dict[str, Any]]


def load_curated_endgame_snapshots(
    data: Sequence[dict[str, Any]] | pd.DataFrame | str | Path | None = None,
    data_dir: str | Path | None = None,
    vocab: ChampionVocabulary | None = None,
    item_vocab: ItemVocabulary | None = None,
    trait_vocab: TraitVocabulary | None = None,
    min_stage: int = 5,
    max_placement: int = 4,
    max_samples: int | None = None,
) -> tuple[list[CuratedEndgameBoard], ChampionVocabulary, ItemVocabulary, TraitVocabulary]:
    """Curate end-game snapshots from Top-4 finishing players at Stage >= min_stage.

    Filters out early/mid game transition noise and retains canonical meta boards.
    """
    vocab = vocab or ChampionVocabulary()
    item_vocab = item_vocab or ItemVocabulary()
    trait_vocab = trait_vocab or TraitVocabulary()

    from tqdm import tqdm

    target_data = data if data is not None else data_dir
    raw_files: list[Path] = []
    raw_records: list[dict[str, Any]] = []

    if target_data is not None:
        if isinstance(target_data, (str, Path)):
            path = Path(target_data)
            if path.is_file():
                raw_files = [path]
            elif path.is_dir():
                raw_files = list(path.glob("*.csv"))
                if not raw_files and (path / "players").exists():
                    raw_files = list((path / "players").glob("*.csv"))
                if not raw_files:
                    raw_files = list(path.rglob("*.csv"))
        elif isinstance(target_data, pd.DataFrame):
            raw_records = target_data.to_dict(orient="records")
        elif isinstance(target_data, Sequence):
            raw_records = list(target_data)

    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    if raw_files:
        iterator = tqdm(raw_files, desc="Scanning match CSVs", unit="file")
        for csv_file in iterator:
            try:
                with open(csv_file, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mid = str(row.get("match_id", "")).strip()
                        fp = str(row.get("focal_player", "")).strip()
                        if not mid or not fp:
                            continue
                        stage_str = str(row.get("round_stage", "")).strip()
                        stage_tuple = parse_stage_string(stage_str)
                        placement_raw = row.get("placement") or row.get("focal_placement") or row.get("rank")
                        placement_val = int(placement_raw) if placement_raw is not None and str(placement_raw).isdigit() else None
                        trajectories[(mid, fp)].append({
                            "match_id": mid,
                            "focal_player": fp,
                            "round_stage": stage_str,
                            "stage_tuple": stage_tuple,
                            "focal_health": float(row.get("focal_health", 100) or 100),
                            "focal_level": float(row.get("focal_level", 1) or 1),
                            "focal_gold": float(row.get("focal_gold", 0) or 0),
                            "input_state_json": row.get("input_state_json"),
                            "placement": placement_val,
                        })
            except Exception:
                continue
    elif raw_records:
        for r in raw_records:
            mid = str(r.get("match_id", "")).strip()
            fp = str(r.get("focal_player", "")).strip()
            if not mid or not fp:
                continue
            stage_str = str(r.get("round_stage", "")).strip()
            stage_tuple = parse_stage_string(stage_str)
            placement_raw = r.get("placement") or r.get("focal_placement") or r.get("rank")
            placement_val = int(placement_raw) if placement_raw is not None and str(placement_raw).isdigit() else None
            trajectories[(mid, fp)].append({
                "match_id": mid,
                "focal_player": fp,
                "round_stage": stage_str,
                "stage_tuple": stage_tuple,
                "focal_health": float(r.get("focal_health", 100) or 100),
                "focal_level": float(r.get("focal_level", 1) or 1),
                "focal_gold": float(r.get("focal_gold", 0) or 0),
                "input_state_json": r.get("input_state_json"),
                "focal_board": r.get("focal_board"),
                "placement": placement_val,
            })

    curated_boards: list[CuratedEndgameBoard] = []
    traj_iterator = tqdm(trajectories.items(), desc="Extracting Top-4 endgame boards", unit="trajectory")

    for (mid, player), traj in traj_iterator:
        if not traj:
            continue
        traj.sort(key=lambda x: x["stage_tuple"])
        last_round = traj[-1]
        last_stage = last_round["stage_tuple"]
        last_health = last_round["focal_health"]

        # Determine if this player finished in Top 4
        explicit_placement = last_round.get("placement")
        if explicit_placement is not None:
            is_top4 = explicit_placement <= max_placement
            placement_num = explicit_placement
        else:
            is_top4 = last_stage >= (5, 5) or (len(traj) >= 18 and last_health > 0)
            placement_num = 1 if last_health > 50 else (2 if last_stage >= (6, 1) else (3 if last_stage >= (5, 5) else 4))

        if not is_top4:
            continue

        # Extract endgame rounds (Stage >= min_stage) - only parse JSON on filtered rounds
        for snapshot in traj:
            st = snapshot["stage_tuple"]
            if st[0] < min_stage:
                continue

            focal_board = snapshot.get("focal_board")
            if not focal_board:
                state_raw = snapshot.get("input_state_json", {})
                if isinstance(state_raw, str):
                    try:
                        state_dict = json.loads(state_raw)
                        focal_board = state_dict.get("focal_board", [])
                    except Exception:
                        focal_board = []
                elif isinstance(state_raw, dict):
                    focal_board = state_raw.get("focal_board", [])
                else:
                    focal_board = []

            if not focal_board:
                continue

            # Encode 4x7 grid
            board_grid = np.zeros((4, 7), dtype=np.int64)
            star_grid = np.zeros((4, 7), dtype=np.int64)
            item_grid = np.zeros((4, 7, 3), dtype=np.int64)

            board_champ_names: list[str] = []
            unit_names: list[str] = []
            unit_tiers: list[int] = []
            unit_items: list[list[str]] = []
            unit_locs: list[str] = []

            occupied: set[tuple[int, int]] = set()
            for unit in focal_board:
                if not isinstance(unit, dict):
                    continue
                u_name = unit.get("unit") or unit.get("champion") or unit.get("apiName") or ""
                u_tier = int(unit.get("tier", 1) or 1)
                u_loc = str(unit.get("loc", ""))
                raw_items = [str(it) for it in (unit.get("items", []) or []) if it]

                c_idx = vocab.encode(u_name)
                if u_name:
                    board_champ_names.append(u_name)
                    unit_names.append(u_name)
                    unit_tiers.append(u_tier)
                    unit_items.append(raw_items)
                    unit_locs.append(u_loc)

                item_idxs = [item_vocab.encode(it) for it in raw_items[:3]]
                while len(item_idxs) < 3:
                    item_idxs.append(0)

                coords = parse_loc_to_row_col(u_loc)
                if coords is not None and coords not in occupied:
                    r_idx, c_idx_col = coords
                    board_grid[r_idx, c_idx_col] = c_idx
                    star_grid[r_idx, c_idx_col] = u_tier
                    item_grid[r_idx, c_idx_col] = item_idxs
                    occupied.add(coords)
                else:
                    for r_idx in range(4):
                        for c_idx_col in range(7):
                            if (r_idx, c_idx_col) not in occupied:
                                board_grid[r_idx, c_idx_col] = c_idx
                                star_grid[r_idx, c_idx_col] = u_tier
                                item_grid[r_idx, c_idx_col] = item_idxs
                                occupied.add((r_idx, c_idx_col))
                                break
                        if (r_idx, c_idx_col) in occupied:
                            break

            trait_vec = trait_vocab.compute_trait_vector(board_champ_names)
            active_traits = trait_vocab.get_active_synergies(board_champ_names)

            curated_board = CuratedEndgameBoard(
                board_champ_ids=board_grid,
                board_star_levels=star_grid,
                board_item_ids=item_grid,
                board_traits=trait_vec,
                unit_names=unit_names,
                unit_tiers=unit_tiers,
                unit_items=unit_items,
                unit_locs=unit_locs,
                active_traits=active_traits,
                match_id=mid,
                focal_player=player,
                round_stage=snapshot["round_stage"],
                stage_tuple=st,
                placement=placement_num,
                health=snapshot["focal_health"],
            )
            curated_boards.append(curated_board)

            if max_samples is not None and len(curated_boards) >= max_samples:
                break
        if max_samples is not None and len(curated_boards) >= max_samples:
            break

    return curated_boards, vocab, item_vocab, trait_vocab


def create_synthetic_endgame_dataset(
    num_samples: int = 120,
    vocab: ChampionVocabulary | None = None,
    item_vocab: ItemVocabulary | None = None,
    trait_vocab: TraitVocabulary | None = None,
    seed: int = 42,
) -> tuple[list[CuratedEndgameBoard], ChampionVocabulary, ItemVocabulary, TraitVocabulary]:
    """Generate realistic synthetic endgame boards across distinct archetype archetypes."""
    vocab = vocab or ChampionVocabulary()
    item_vocab = item_vocab or ItemVocabulary()
    trait_vocab = trait_vocab or TraitVocabulary()

    rng = np.random.default_rng(seed)

    # Canonical archetype templates for Set 18 / Set 17 testing
    archetypes = [
        {
            "name": "Rebel / Cybernetic Reroll",
            "units": ["TFT18_Irelia", "TFT18_Jinx", "TFT18_Vi", "TFT18_Ekko", "TFT18_Illaoi", "TFT18_Singed", "TFT18_Sejuani", "TFT18_Warwick"],
            "carry": "TFT18_Jinx",
            "tank": "TFT18_Illaoi",
            "carry_items": ["TFT_Item_GuinsoosRageblade", "TFT_Item_InfinityEdge", "TFT_Item_LastWhisper"],
            "tank_items": ["TFT_Item_WarmogsArmor", "TFT_Item_BrambleVest", "TFT_Item_DragonsClaw"],
        },
        {
            "name": "Invoker / Preserver Fast-8",
            "units": ["TFT18_Cassiopeia", "TFT18_Syndra", "TFT18_Karma", "TFT18_Morgana", "TFT18_Rakan", "TFT18_Nasus", "TFT18_Zilean", "TFT18_Milio"],
            "carry": "TFT18_Karma",
            "tank": "TFT18_Nasus",
            "carry_items": ["TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap"],
            "tank_items": ["TFT_Item_GargoyleStoneplate", "TFT_Item_WarmogsArmor", "TFT_Item_Redemption"],
        },
        {
            "name": "Duelist / Vanguard Tempo",
            "units": ["TFT18_Ashe", "TFT18_Jax", "TFT18_Twitch", "TFT18_LeeSin", "TFT18_Shen", "TFT18_TahmKench", "TFT18_Olaf", "TFT18_Volibear"],
            "carry": "TFT18_Olaf",
            "tank": "TFT18_TahmKench",
            "carry_items": ["TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_SteraksGage"],
            "tank_items": ["TFT_Item_SunfireCape", "TFT_Item_WarmogsArmor", "TFT_Item_SteadfastHeart"],
        },
        {
            "name": "Arcanist / Bastion Burst",
            "units": ["TFT18_Zoe", "TFT18_Ahri", "TFT18_Veigar", "TFT18_Taric", "TFT18_Diana", "TFT18_Vex", "TFT18_Nami", "TFT18_Norra"],
            "carry": "TFT18_Veigar",
            "tank": "TFT18_Taric",
            "carry_items": ["TFT_Item_SpearOfShojin", "TFT_Item_ArchangelsStaff", "TFT_Item_NashorsTooth"],
            "tank_items": ["TFT_Item_IonicSpark", "TFT_Item_Crownguard", "TFT_Item_WarmogsArmor"],
        },
        {
            "name": "Legendary Fast-9 Capstone",
            "units": ["TFT18_Camille", "TFT18_Briar", "TFT18_Smolder", "TFT18_Xerath", "TFT18_Diana", "TFT18_Morgana", "TFT18_Norra", "TFT18_Milio", "TFT18_Varus"],
            "carry": "TFT18_Smolder",
            "tank": "TFT18_Diana",
            "carry_items": ["TFT_Item_GuinsoosRageblade", "TFT_Item_InfinityEdge", "TFT_Item_Bloodthirster"],
            "tank_items": ["TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_BrambleVest"],
        },
    ]

    # Register vocabulary tokens and trait synergies
    archetype_traits: dict[str, list[str]] = {
        "TFT18_Irelia": ["Rebel", "Duelist"],
        "TFT18_Jinx": ["Rebel", "Gunner"],
        "TFT18_Vi": ["Rebel", "Bruiser"],
        "TFT18_Ekko": ["Rebel", "Cybernetic"],
        "TFT18_Illaoi": ["Cybernetic", "Bastion"],
        "TFT18_Singed": ["Cybernetic", "Alchemist"],
        "TFT18_Sejuani": ["Cybernetic", "Vanguard"],
        "TFT18_Warwick": ["Cybernetic", "Predator"],
        "TFT18_Cassiopeia": ["Invoker", "Witch"],
        "TFT18_Syndra": ["Invoker", "Arcanist"],
        "TFT18_Karma": ["Invoker", "Preserver"],
        "TFT18_Morgana": ["Preserver", "Witchcraft"],
        "TFT18_Rakan": ["Preserver", "Bastion"],
        "TFT18_Nasus": ["Preserver", "Vanguard"],
        "TFT18_Zilean": ["Invoker", "Preserver"],
        "TFT18_Milio": ["Invoker", "Faerie"],
        "TFT18_Ashe": ["Duelist", "Sniper"],
        "TFT18_Jax": ["Duelist", "Bastion"],
        "TFT18_Twitch": ["Duelist", "Hunter"],
        "TFT18_LeeSin": ["Duelist", "Monk"],
        "TFT18_Shen": ["Vanguard", "Mystic"],
        "TFT18_TahmKench": ["Vanguard", "Glutton"],
        "TFT18_Olaf": ["Duelist", "Berserker"],
        "TFT18_Volibear": ["Vanguard", "Storm"],
        "TFT18_Zoe": ["Arcanist", "Portal"],
        "TFT18_Ahri": ["Arcanist", "Fated"],
        "TFT18_Veigar": ["Arcanist", "Mage"],
        "TFT18_Taric": ["Bastion", "Preserver"],
        "TFT18_Diana": ["Bastion", "Lunar"],
        "TFT18_Vex": ["Arcanist", "Yordle"],
        "TFT18_Nami": ["Arcanist", "Ocean"],
        "TFT18_Norra": ["Arcanist", "Portal"],
        "TFT18_Camille": ["Chrono", "Duelist"],
        "TFT18_Briar": ["Vampire", "Bruiser"],
        "TFT18_Smolder": ["Dragon", "Blaster"],
        "TFT18_Xerath": ["Ascendant", "Arcanist"],
        "TFT18_Varus": ["Sniper", "Pyro"],
    }

    for arch in archetypes:
        for u in arch["units"]:
            vocab.add_champion(u)
            if u in archetype_traits:
                trait_vocab.register_champ_traits(u, archetype_traits[u])
        for it in arch["carry_items"] + arch["tank_items"]:
            item_vocab.add_item(it)

    curated_boards: list[CuratedEndgameBoard] = []

    for i in range(num_samples):
        arch = archetypes[i % len(archetypes)]
        match_id = f"SYNTH_MATCH_{i // 4:04d}"
        focal_player = f"Player_{i % 4 + 1}"
        stage_num = int(rng.choice([5, 6]))
        round_in_stage = int(rng.choice([1, 2, 3, 5, 6]))
        stage_tuple = (stage_num, round_in_stage)
        stage_str = f"{stage_num}-{round_in_stage}"
        placement = int((i % 4) + 1)
        health = float(rng.uniform(15.0, 95.0))

        # Sample units with small variations (simulating flex slots)
        base_units = list(arch["units"])
        if rng.random() > 0.4 and len(base_units) > 6:
            swap_idx = int(rng.integers(2, len(base_units)))
            base_units[swap_idx] = "TFT18_Illaoi" if base_units[swap_idx] != "TFT18_Illaoi" else "TFT18_Zilean"
            vocab.add_champion(base_units[swap_idx])

        board_grid = np.zeros((4, 7), dtype=np.int64)
        star_grid = np.zeros((4, 7), dtype=np.int64)
        item_grid = np.zeros((4, 7, 3), dtype=np.int64)

        board_champ_names: list[str] = []
        unit_names: list[str] = []
        unit_tiers: list[int] = []
        unit_items: list[list[str]] = []
        unit_locs: list[str] = []

        occupied: set[tuple[int, int]] = set()

        for u_name in base_units:
            c_idx = vocab.encode(u_name)
            board_champ_names.append(u_name)
            unit_names.append(u_name)

            # Star level
            if u_name == arch["carry"] or u_name == arch["tank"]:
                tier = int(rng.choice([2, 3], p=[0.75, 0.25]))
            else:
                tier = int(rng.choice([1, 2], p=[0.2, 0.8]))
            unit_tiers.append(tier)

            # Items
            if u_name == arch["carry"]:
                items = list(arch["carry_items"])
            elif u_name == arch["tank"]:
                items = list(arch["tank_items"])
            else:
                items = []
                if rng.random() > 0.6:
                    items.append(str(rng.choice(arch["carry_items"] + arch["tank_items"])))
            unit_items.append(items)

            item_idxs = [item_vocab.encode(it) for it in items[:3]]
            while len(item_idxs) < 3:
                item_idxs.append(0)

            # Placement: Tanks in front rows (0, 1), Carries in back rows (2, 3)
            is_frontline = (u_name == arch["tank"] or "Vanguard" in u_name or "Bastion" in u_name or "Illaoi" in u_name)
            target_rows = [0, 1] if is_frontline else [2, 3]

            placed = False
            for r_try in target_rows:
                for c_try in range(7):
                    if (r_try, c_try) not in occupied:
                        board_grid[r_try, c_try] = c_idx
                        star_grid[r_try, c_try] = tier
                        item_grid[r_try, c_try] = item_idxs
                        occupied.add((r_try, c_try))
                        row_letter = chr(ord("A") + r_try)
                        unit_locs.append(f"{row_letter}{c_try + 1}")
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                for r_try in range(4):
                    for c_try in range(7):
                        if (r_try, c_try) not in occupied:
                            board_grid[r_try, c_try] = c_idx
                            star_grid[r_try, c_try] = tier
                            item_grid[r_try, c_try] = item_idxs
                            occupied.add((r_try, c_try))
                            row_letter = chr(ord("A") + r_try)
                            unit_locs.append(f"{row_letter}{c_try + 1}")
                            placed = True
                            break
                    if placed:
                        break

        trait_vec = trait_vocab.compute_trait_vector(board_champ_names)
        active_traits = trait_vocab.get_active_synergies(board_champ_names)

        curated_board = CuratedEndgameBoard(
            board_champ_ids=board_grid,
            board_star_levels=star_grid,
            board_item_ids=item_grid,
            board_traits=trait_vec,
            unit_names=unit_names,
            unit_tiers=unit_tiers,
            unit_items=unit_items,
            unit_locs=unit_locs,
            active_traits=active_traits,
            match_id=match_id,
            focal_player=focal_player,
            round_stage=stage_str,
            stage_tuple=stage_tuple,
            placement=placement,
            health=health,
        )
        curated_boards.append(curated_board)

    return curated_boards, vocab, item_vocab, trait_vocab


def extract_board_latents(
    trunk: MultiModalFusionTrunk,
    boards: Sequence[CuratedEndgameBoard],
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, list[CuratedEndgameBoard]]:
    """Extract isolated 256D board representations via frozen BoardHexTransformer.

    Deliberately bypasses StateMLP (HP, Gold, Level scalars) to ensure strategic
    archetype identity is invariant to the player's economic/HP state.
    """
    trunk = trunk.to(device)
    trunk.eval()
    trunk.freeze()

    all_feats: list[torch.Tensor] = []
    num_samples = len(boards)

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            batch_slice = boards[start_idx : start_idx + batch_size]
            b_champ = torch.tensor(np.array([b.board_champ_ids for b in batch_slice]), dtype=torch.long, device=device)
            b_star = torch.tensor(np.array([b.board_star_levels for b in batch_slice]), dtype=torch.long, device=device)
            b_item = torch.tensor(np.array([b.board_item_ids for b in batch_slice]), dtype=torch.long, device=device)
            b_trait = torch.tensor(np.array([b.board_traits for b in batch_slice]), dtype=torch.float32, device=device)

            feat = trunk.encode_board(
                board_champ_ids=b_champ,
                board_star_levels=b_star,
                board_item_ids=b_item,
                board_traits=b_trait,
            )
            all_feats.append(feat.cpu())

    if all_feats:
        latent_tensor = torch.cat(all_feats, dim=0)
    else:
        latent_tensor = torch.empty((0, trunk.board_feat_dim), dtype=torch.float32)

    return latent_tensor, list(boards)


class CompositionClusterer:
    """Unsupervised geometric clustering of TFT endgame board latent vectors.

    Distills canonical meta archetypes into K mathematical centroids (Z-Index):
    Z = {z_1, z_2, ..., z_K} in R^(K x 256).
    """

    def __init__(
        self,
        n_clusters: int = 15,
        algorithm: str = "kmeans",
        random_state: int = 42,
        normalize_latents: bool = True,
        center_latents: bool = True,
        min_cluster_size: int = 2,
    ) -> None:
        self.n_clusters = max(2, n_clusters)
        self.algorithm = algorithm.lower()
        self.random_state = random_state
        self.normalize_latents = normalize_latents
        self.center_latents = center_latents
        self.min_cluster_size = min_cluster_size

        self.centroids: torch.Tensor | None = None  # (K, 256)
        self.mean_vector: torch.Tensor | None = None  # (256,)
        self.labels_: np.ndarray | None = None  # (N,)
        self.metrics_: dict[str, Any] = {}
        self.profiles_: list[ArchetypeProfile] = []
        self.is_fitted = False

    def fit(
        self,
        latents: torch.Tensor | np.ndarray,
        boards: Sequence[CuratedEndgameBoard] | None = None,
    ) -> CompositionClusterer:
        """Fit K-Means clustering over the extracted 256D latent vectors."""
        if isinstance(latents, torch.Tensor):
            x_arr = latents.detach().cpu().numpy()
        else:
            x_arr = np.asarray(latents)

        num_samples, feat_dim = x_arr.shape
        if num_samples < self.n_clusters:
            effective_k = max(1, num_samples)
        else:
            effective_k = self.n_clusters

        if self.center_latents:
            mean_np = np.mean(x_arr, axis=0, keepdims=True)
            self.mean_vector = torch.tensor(mean_np.squeeze(0), dtype=torch.float32)
            x_work = x_arr - mean_np
        else:
            self.mean_vector = None
            x_work = x_arr

        if self.normalize_latents:
            norms = np.linalg.norm(x_work, axis=1, keepdims=True) + 1e-8
            x_input = x_work / norms
        else:
            x_input = x_work

        # Fit K-Means
        if effective_k > 1:
            if num_samples > 2000:
                kmeans = MiniBatchKMeans(
                    n_clusters=effective_k,
                    random_state=self.random_state,
                    batch_size=256,
                    n_init=3,
                )
            else:
                kmeans = KMeans(
                    n_clusters=effective_k,
                    random_state=self.random_state,
                    n_init=10,
                )
            labels = kmeans.fit_predict(x_input)
            centroids_np = kmeans.cluster_centers_
            inertia = float(kmeans.inertia_)
        else:
            labels = np.zeros(num_samples, dtype=np.int64)
            centroids_np = np.mean(x_input, axis=0, keepdims=True)
            inertia = 0.0

        self.labels_ = labels
        self.centroids = torch.tensor(centroids_np, dtype=torch.float32)

        # Compute validation clustering metrics
        metrics: dict[str, Any] = {
            "n_samples": int(num_samples),
            "n_clusters": int(effective_k),
            "feature_dim": int(feat_dim),
            "inertia": float(inertia),
        }

        unique_labels = len(np.unique(labels))
        if num_samples > effective_k and unique_labels > 1:
            try:
                metrics["silhouette_score"] = float(silhouette_score(x_input, labels, metric="euclidean"))
            except Exception:
                metrics["silhouette_score"] = 0.0

            try:
                metrics["davies_bouldin_index"] = float(davies_bouldin_score(x_input, labels))
            except Exception:
                metrics["davies_bouldin_index"] = 0.0

            try:
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(x_input, labels))
            except Exception:
                metrics["calinski_harabasz_score"] = 0.0
        else:
            metrics["silhouette_score"] = 0.0
            metrics["davies_bouldin_index"] = 0.0
            metrics["calinski_harabasz_score"] = 0.0

        self.metrics_ = metrics
        self.is_fitted = True

        # Generate archetype interpretability profiles if boards are provided
        if boards is not None and len(boards) == num_samples:
            self.profiles_ = profile_clusters(
                self,
                torch.tensor(x_arr, dtype=torch.float32),
                list(boards),
            )

        return self

    def score_similarity(self, board_feat: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between arbitrary board features and all Z-Index centroids.

        Args:
            board_feat: Tensor of shape (Batch, 256) or (256,).

        Returns:
            Cosine similarity matrix of shape (Batch, K) or (K,).
        """
        if not self.is_fitted or self.centroids is None:
            raise RuntimeError("CompositionClusterer is not fitted yet. Call .fit() first.")

        if board_feat.dim() == 1:
            feat = board_feat.unsqueeze(0)
            single = True
        else:
            feat = board_feat
            single = False

        device = feat.device
        centroids = self.centroids.to(device)

        if self.mean_vector is not None:
            feat = feat - self.mean_vector.to(device)

        # Normalize features and centroids for cosine similarity
        feat_norm = F.normalize(feat, p=2, dim=-1)
        cent_norm = F.normalize(centroids, p=2, dim=-1)

        sim = torch.mm(feat_norm, cent_norm.t())  # (Batch, K)
        return sim.squeeze(0) if single else sim

    def predict(self, board_feat: torch.Tensor) -> torch.Tensor:
        """Predict the closest archetype centroid index for given board feature(s)."""
        sim = self.score_similarity(board_feat)
        if sim.dim() == 1:
            return torch.argmax(sim, dim=0)
        return torch.argmax(sim, dim=-1)


def profile_clusters(
    clusterer: CompositionClusterer,
    latents: torch.Tensor,
    boards: Sequence[CuratedEndgameBoard],
    top_n_champs: int = 8,
    top_n_traits: int = 5,
    top_n_items: int = 6,
) -> list[ArchetypeProfile]:
    """Profile and interpret each cluster into actionable human-readable archetypes."""
    if clusterer.labels_ is None or clusterer.centroids is None:
        return []

    labels = clusterer.labels_
    centroids = clusterer.centroids
    num_clusters = len(centroids)
    total_boards = len(boards)

    profiles: list[ArchetypeProfile] = []

    for k in range(num_clusters):
        cluster_mask = (labels == k)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = int(len(cluster_indices))

        if cluster_size == 0:
            profiles.append(
                ArchetypeProfile(
                    cluster_id=k,
                    name=f"Archetype_{k:02d} (Empty)",
                    size=0,
                    percentage=0.0,
                    top_units=[],
                    dominant_traits=[],
                    top_items=[],
                    avg_placement=4.0,
                    representative_boards=[],
                )
            )
            continue

        percentage = (cluster_size / total_boards) * 100.0

        # Champion frequencies and star levels
        champ_counts: Counter[str] = Counter()
        champ_tiers: defaultdict[str, list[int]] = defaultdict(list)
        item_counts: Counter[str] = Counter()
        trait_tiers: defaultdict[str, list[int]] = defaultdict(list)
        placements: list[int] = []

        for idx in cluster_indices:
            b = boards[idx]
            placements.append(b.placement)
            for u_name, u_tier, items in zip(b.unit_names, b.unit_tiers, b.unit_items):
                champ_counts[u_name] += 1
                champ_tiers[u_name].append(u_tier)
                for it in items:
                    item_counts[it] += 1

            for trait_name, tier in b.active_traits.items():
                if tier > 0:
                    trait_tiers[trait_name].append(tier)

        avg_placement = float(np.mean(placements)) if placements else 4.0

        # Top champions
        top_units_list: list[dict[str, Any]] = []
        for u_name, count in champ_counts.most_common(top_n_champs):
            avg_tier = float(np.mean(champ_tiers[u_name])) if champ_tiers[u_name] else 1.0
            top_units_list.append({
                "champion": u_name,
                "frequency_pct": round((count / cluster_size) * 100.0, 1),
                "avg_star_level": round(avg_tier, 2),
            })

        # Top traits
        dominant_traits_list: list[dict[str, Any]] = []
        for t_name, tiers in sorted(trait_tiers.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_traits]:
            dominant_traits_list.append({
                "trait": t_name,
                "frequency_pct": round((len(tiers) / cluster_size) * 100.0, 1),
                "avg_tier": round(float(np.mean(tiers)), 1),
            })

        # Top items
        top_items_list: list[dict[str, Any]] = []
        for it_name, count in item_counts.most_common(top_n_items):
            top_items_list.append({
                "item": it_name,
                "count": count,
                "frequency_pct": round((count / cluster_size) * 100.0, 1),
            })

        # Nearest representative boards to centroid (Top 5 per cluster)
        centroid_k = centroids[k]
        cluster_latents_raw = latents[cluster_indices]
        if clusterer.mean_vector is not None:
            cluster_latents_work = cluster_latents_raw - clusterer.mean_vector.to(cluster_latents_raw.device)
        else:
            cluster_latents_work = cluster_latents_raw

        if clusterer.normalize_latents:
            cluster_latents_norm = F.normalize(cluster_latents_work, p=2, dim=-1)
            cent_norm_k = F.normalize(centroid_k.unsqueeze(0), p=2, dim=-1).to(cluster_latents_norm.device)
            cos_sims = torch.mm(cluster_latents_norm, cent_norm_k.t()).squeeze(-1).detach().cpu().numpy()
            dists = 1.0 - cos_sims
        else:
            dists = torch.norm(cluster_latents_work - centroid_k, dim=1).detach().cpu().numpy()

        sorted_rel_idx = np.argsort(dists)[:5]

        representative_boards_list: list[dict[str, Any]] = []
        for rank_idx, rel_idx in enumerate(sorted_rel_idx, start=1):
            orig_idx = cluster_indices[rel_idx]
            rep_b = boards[orig_idx]
            cos_sim_val = float(1.0 - dists[rel_idx]) if clusterer.normalize_latents else float(dists[rel_idx])
            representative_boards_list.append({
                "rank": rank_idx,
                "match_id": rep_b.match_id,
                "round_stage": rep_b.round_stage,
                "placement": rep_b.placement,
                "distance_to_centroid": round(float(dists[rel_idx]), 4),
                "cosine_similarity": round(cos_sim_val, 4),
                "units": [
                    f"{u} (★{t})" + (f" [{' ,'.join(its)}]" if its else "")
                    for u, t, its in zip(rep_b.unit_names, rep_b.unit_tiers, rep_b.unit_items)
                ],
                "active_traits": rep_b.active_traits,
            })

        # Synthesize archetype name from dominant traits and top carries
        carry_raw = top_units_list[0]["champion"] if top_units_list else "Unknown"
        carry_clean = re.sub(r"^(TFT18_|DA_18_|DA_)", "", carry_raw, flags=re.IGNORECASE)
        carry_clean = re.sub(r"(18|_AP|_Small|_small)$", "", carry_clean)

        if len(top_units_list) > 1:
            second_raw = top_units_list[1]["champion"]
            second_clean = re.sub(r"^(TFT18_|DA_18_|DA_)", "", second_raw, flags=re.IGNORECASE)
            second_clean = re.sub(r"(18|_AP|_Small|_small)$", "", second_clean)
            units_summary = f"{carry_clean} & {second_clean}"
        else:
            units_summary = carry_clean

        if dominant_traits_list:
            trait_parts = [f"{t['avg_tier']:.0f} {t['trait']}" for t in dominant_traits_list[:2]]
            trait_str = " / ".join(trait_parts)
            name = f"Cluster {k:02d}: {trait_str} ({units_summary} Core)"
        else:
            name = f"Cluster {k:02d}: {units_summary} Flex"

        profiles.append(
            ArchetypeProfile(
                cluster_id=k,
                name=name,
                size=cluster_size,
                percentage=round(percentage, 2),
                top_units=top_units_list,
                dominant_traits=dominant_traits_list,
                top_items=top_items_list,
                avg_placement=round(avg_placement, 2),
                representative_boards=representative_boards_list,
            )
        )

    return profiles


def log_clustering_to_wandb(
    clusterer: CompositionClusterer,
    latents: torch.Tensor,
    boards: Sequence[CuratedEndgameBoard],
    wandb_project: str = "tft-clustering",
    run_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    output_dir: str | Path | None = None,
    use_wandb: bool = True,
) -> dict[str, Any]:
    """Generate dimensionality reduction plots, heatmaps, and log metrics to Weights & Biases."""
    output_path = Path(output_dir) if output_dir else Path("models/clustering")
    fig_dir = output_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    latents_np = latents.detach().cpu().numpy()
    labels = clusterer.labels_ if clusterer.labels_ is not None else np.zeros(len(latents_np))
    centroids = clusterer.centroids.detach().cpu().numpy() if clusterer.centroids is not None else np.zeros((1, 256))

    # 1. 2D Dimensionality Reduction (PCA + UMAP if available)
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(latents_np)
    centroids_2d = pca.transform(centroids)

    # Plot 1: 2D Projection Map
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cmap = plt.get_cmap("tab20", clusterer.n_clusters)
    scatter = ax.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.6,
        s=25,
        edgecolors="none",
    )
    # Plot centroids
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="black",
        marker="X",
        s=120,
        linewidths=1.5,
        edgecolors="white",
        label="Z-Index Centroids",
    )
    for k in range(len(centroids_2d)):
        ax.annotate(
            f"Z_{k}",
            (centroids_2d[k, 0], centroids_2d[k, 1]),
            fontsize=9,
            fontweight="bold",
            color="black",
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_title(
        f"TFT Composition Latent Space (PCA Projection)\n"
        f"K={clusterer.n_clusters} Archetypes | Silhouette={clusterer.metrics_.get('silhouette_score', 0):.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(f"PCA-1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PCA-2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()

    pca_plot_path = fig_dir / "pca_latent_clusters.png"
    fig.savefig(pca_plot_path)
    plt.close(fig)

    # Plot 2: Centroid Cosine Similarity Matrix Heatmap
    if clusterer.centroids is not None and len(clusterer.centroids) > 1:
        cent_norm = F.normalize(clusterer.centroids, p=2, dim=-1)
        sim_matrix = torch.mm(cent_norm, cent_norm.t()).numpy()

        fig_sim, ax_sim = plt.subplots(figsize=(9, 8), dpi=150)
        cax = ax_sim.matshow(sim_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        fig_sim.colorbar(cax, fraction=0.046, pad=0.04)

        ax_sim.set_xticks(range(len(cent_norm)))
        ax_sim.set_yticks(range(len(cent_norm)))
        ax_sim.set_xticklabels([f"z_{k}" for k in range(len(cent_norm))], rotation=45)
        ax_sim.set_yticklabels([f"z_{k}" for k in range(len(cent_norm))])
        ax_sim.set_title("Z-Index Centroid Pairwise Cosine Similarity", fontsize=12, fontweight="bold", pad=20)

        for i in range(len(cent_norm)):
            for j in range(len(cent_norm)):
                val = sim_matrix[i, j]
                ax_sim.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.6 else "black", fontsize=7)

        plt.tight_layout()
        sim_plot_path = fig_dir / "centroid_cosine_similarity.png"
        fig_sim.savefig(sim_plot_path)
        plt.close(fig_sim)
    else:
        sim_plot_path = None

    # Plot 3: Archetype Population Distribution
    fig_pop, ax_pop = plt.subplots(figsize=(10, 5), dpi=150)
    sizes = [p.size for p in clusterer.profiles_]
    names = [f"Z_{p.cluster_id}\n({p.percentage}%)" for p in clusterer.profiles_]
    bars = ax_pop.bar(range(len(sizes)), sizes, color="royalblue", alpha=0.85, edgecolor="black")
    ax_pop.set_xticks(range(len(sizes)))
    ax_pop.set_xticklabels(names, fontsize=8)
    ax_pop.set_ylabel("Endgame Boards Count")
    ax_pop.set_title("Archetype Cluster Population Distribution", fontsize=12, fontweight="bold")
    ax_pop.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        ax_pop.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    pop_plot_path = fig_dir / "cluster_population.png"
    fig_pop.savefig(pop_plot_path)
    plt.close(fig_pop)

    # WandB Logging
    if use_wandb:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                name=run_name or f"z_index_k{clusterer.n_clusters}_{int(time.time())}",
                entity=wandb_entity,
                group=wandb_group,
                reinit=True,
            )

            # Compute off-diagonal centroid cosine similarity metrics
            if clusterer.centroids is not None and len(clusterer.centroids) > 1:
                cent_norm = F.normalize(clusterer.centroids, p=2, dim=-1)
                sim_mat = torch.mm(cent_norm, cent_norm.t()).detach().cpu().numpy()
                n_c = len(cent_norm)
                off_mask = ~np.eye(n_c, dtype=bool)
                off_sims = sim_mat[off_mask]
                off_mean = float(np.mean(off_sims))
                off_max = float(np.max(off_sims))
                off_min = float(np.min(off_sims))
            else:
                off_mean, off_max, off_min = 0.0, 0.0, 0.0

            # Log scalar metrics
            wandb.log({
                "clustering/n_samples": clusterer.metrics_.get("n_samples", 0),
                "clustering/n_clusters": clusterer.metrics_.get("n_clusters", 0),
                "clustering/silhouette_score": clusterer.metrics_.get("silhouette_score", 0.0),
                "clustering/davies_bouldin_index": clusterer.metrics_.get("davies_bouldin_index", 0.0),
                "clustering/calinski_harabasz_score": clusterer.metrics_.get("calinski_harabasz_score", 0.0),
                "clustering/inertia": clusterer.metrics_.get("inertia", 0.0),
                "clustering/off_diagonal_mean_cosine_sim": off_mean,
                "clustering/off_diagonal_max_cosine_sim": off_max,
                "clustering/off_diagonal_min_cosine_sim": off_min,
                "plots/pca_latent_map": wandb.Image(str(pca_plot_path)),
                "plots/cluster_population": wandb.Image(str(pop_plot_path)),
            })

            if sim_plot_path is not None and sim_plot_path.exists():
                wandb.log({"plots/centroid_cosine_similarity": wandb.Image(str(sim_plot_path))})

            # Log Archetype Summary Table
            table = wandb.Table(
                columns=["Cluster ID", "Archetype Name", "Size", "Share %", "Avg Placement", "Top Champions", "Dominant Traits", "Core Items"]
            )
            for p in clusterer.profiles_:
                top_champs_str = ", ".join([f"{u['champion']} ({u['frequency_pct']}%)" for u in p.top_units[:4]])
                dominant_traits_str = ", ".join([f"{t['avg_tier']:.0f} {t['trait']}" for t in p.dominant_traits[:3]])
                top_items_str = ", ".join([f"{it['item']}" for it in p.top_items[:3]])
                table.add_data(
                    p.cluster_id,
                    p.name,
                    p.size,
                    p.percentage,
                    p.avg_placement,
                    top_champs_str,
                    dominant_traits_str,
                    top_items_str,
                )
            wandb.log({"archetypes/profiles_summary_table": table})

            # Log Top-5 Representative Compositions Table per Centroid
            rep_table = wandb.Table(
                columns=[
                    "Cluster ID",
                    "Archetype Name",
                    "Board Rank (1-5)",
                    "Stage",
                    "Placement",
                    "Centroid Cosine Sim",
                    "Carries & Items",
                    "Tanks & Items",
                    "Full Board Units",
                    "Active Synergies",
                    "Match ID",
                ]
            )
            for p in clusterer.profiles_:
                for rep in p.representative_boards:
                    carry_units: list[str] = []
                    tank_units: list[str] = []
                    for u_str in rep["units"]:
                        if "[" in u_str:
                            if any(it in u_str for it in ["Shojin", "Deathcap", "Gauntlet", "Rageblade", "InfinityEdge", "LastWhisper", "Kraken", "Hatchet", "BlueBuff", "Archangel", "Morello", "RedBuff"]):
                                carry_units.append(u_str)
                            else:
                                tank_units.append(u_str)
                    
                    carries_str = ", ".join(carry_units) if carry_units else "None"
                    tanks_str = ", ".join(tank_units) if tank_units else "None"
                    full_units_str = ", ".join(rep["units"])
                    active_traits = rep.get("active_traits", {})
                    synergies_str = ", ".join([f"{cnt} {t}" for t, cnt in sorted(active_traits.items(), key=lambda x: x[1], reverse=True)]) if active_traits else "N/A"
                    cos_sim_val = rep.get("cosine_similarity", round(1.0 - rep.get("distance_to_centroid", 0.0), 4))

                    rep_table.add_data(
                        p.cluster_id,
                        p.name,
                        rep.get("rank", 1),
                        rep.get("round_stage", ""),
                        rep.get("placement", 4),
                        cos_sim_val,
                        carries_str,
                        tanks_str,
                        full_units_str,
                        synergies_str,
                        rep.get("match_id", ""),
                    )
            wandb.log({"archetypes/top5_representative_compositions": rep_table})

            wandb.finish()
            logger.info("Successfully logged clustering metrics and plots to Weights & Biases.")
        except Exception as e:
            logger.warning(f"Could not log to Weights & Biases: {e}")

    return {
        "pca_plot": str(pca_plot_path),
        "population_plot": str(pop_plot_path),
        "similarity_plot": str(sim_plot_path) if sim_plot_path else None,
    }


def save_z_index_artifacts(
    clusterer: CompositionClusterer,
    output_dir: str | Path,
    vocab: ChampionVocabulary | None = None,
    item_vocab: ItemVocabulary | None = None,
    trait_vocab: TraitVocabulary | None = None,
) -> dict[str, Path]:
    """Save z_index.pt tensor checkpoint, cluster_profiles.json, and markdown summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    z_index_path = out / "z_index.pt"
    json_path = out / "cluster_profiles.json"
    md_path = out / "cluster_profiles.md"

    # 1. Save PyTorch artifact
    torch.save(
        {
            "z_index": clusterer.centroids,  # (K, 256)
            "mean_vector": clusterer.mean_vector,  # (256,)
            "k": clusterer.n_clusters,
            "feature_dim": 256,
            "metrics": clusterer.metrics_,
            "config": {
                "algorithm": clusterer.algorithm,
                "normalize_latents": clusterer.normalize_latents,
                "center_latents": clusterer.center_latents,
                "random_state": clusterer.random_state,
            },
            "num_champs": len(vocab) if vocab else 500,
            "num_items": len(item_vocab) if item_vocab else 300,
            "num_traits": len(trait_vocab) if trait_vocab else 60,
        },
        z_index_path,
    )

    # 2. Save JSON profiles
    profiles_dict = [asdict(p) for p in clusterer.profiles_]
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": clusterer.metrics_,
                "num_archetypes": len(clusterer.profiles_),
                "archetypes": profiles_dict,
            },
            f,
            indent=2,
        )

    # 3. Save Markdown table summary
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write("# TFT Extracted Composition Archetypes (Z-Index)\n\n")
        f.write(f"**Total Archetypes (K):** {clusterer.n_clusters}  \n")
        f.write(f"**Silhouette Score:** {clusterer.metrics_.get('silhouette_score', 0.0):.4f}  \n")
        f.write(f"**Davies-Bouldin Index:** {clusterer.metrics_.get('davies_bouldin_index', 0.0):.4f}  \n\n")
        f.write("| ID | Archetype Name | Size | Share | Avg Place | Top Champions | Dominant Traits |\n")
        f.write("| :-: | :--- | :-: | :-: | :-: | :--- | :--- |\n")
        for p in clusterer.profiles_:
            champs = ", ".join([f"{u['champion']} (★{u['avg_star_level']:.1f})" for u in p.top_units[:3]])
            traits = ", ".join([f"{t['avg_tier']:.0f} {t['trait']}" for t in p.dominant_traits[:2]])
            f.write(f"| Z_{p.cluster_id:02d} | {p.name} | {p.size} | {p.percentage}% | #{p.avg_placement:.1f} | {champs} | {traits} |\n")

    return {
        "z_index_pt": z_index_path,
        "profiles_json": json_path,
        "profiles_md": md_path,
    }


def load_z_index(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load Z-Index centroid tensors and metadata from a saved .pt file."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return checkpoint


def run_clustering_pipeline(
    data_dir: str | Path | None = None,
    trunk_checkpoint: str | Path | None = None,
    min_stage: int = 5,
    max_placement: int = 4,
    n_clusters: int = 15,
    batch_size: int = 128,
    device: str = "cpu",
    use_wandb: bool = True,
    wandb_project: str = "tft-clustering",
    run_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    output_dir: str | Path = "models/clustering",
    synthetic: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Orchestrate the end-to-end Z-Index clustering and profiling pipeline."""
    print("=" * 75)
    print(" [*] TFT Composition Archetype Extraction & Latent Clustering (Z-Index)")
    print("=" * 75)

    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    # 1. Load curated dataset
    if synthetic or data_dir is None or not Path(data_dir).exists():
        if not synthetic and data_dir is not None:
            print(f"[!] Data directory '{data_dir}' not found. Falling back to synthetic dataset.")
        print(" [+] Generating synthetic endgame dataset (Stage >= 5, Top-4)...")
        boards, vocab, item_vocab, trait_vocab = create_synthetic_endgame_dataset(
            num_samples=150,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
        )
    else:
        print(f" [+] Curating endgame boards from: {data_dir} (Stage >= {min_stage}, Top-{max_placement})")
        boards, vocab, item_vocab, trait_vocab = load_curated_endgame_snapshots(
            data_dir=data_dir,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
            min_stage=min_stage,
            max_placement=max_placement,
            max_samples=max_samples,
        )

    print(f" [+] Extracted {len(boards)} curated winning endgame boards.")
    if not boards:
        print(" [!] No endgame boards match curation criteria.", file=sys.stderr)
        return {"error": "no_boards_found"}

    # 2. Load pre-trained trunk
    checkpoint_path: Path | None = None
    if trunk_checkpoint:
        p = Path(trunk_checkpoint)
        if p.exists():
            checkpoint_path = p
        elif (p.parent / "trunk_best.pt").exists():
            checkpoint_path = p.parent / "trunk_best.pt"
        elif (p.parent / "trunk_final.pt").exists():
            checkpoint_path = p.parent / "trunk_final.pt"
    if checkpoint_path is None and Path("models/trunk/trunk_best.pt").exists():
        checkpoint_path = Path("models/trunk/trunk_best.pt")

    if checkpoint_path and checkpoint_path.exists():
        print(f" [+] Loading pre-trained MultiModalFusionTrunk from: {checkpoint_path}")
        trunk = MultiModalFusionTrunk.load_trunk(checkpoint_path, map_location=device)
    else:
        print(" [+] Instantiating MultiModalFusionTrunk from vocabularies...")
        trunk = MultiModalFusionTrunk(
            num_champs=len(vocab),
            num_items=len(item_vocab),
            num_traits=len(trait_vocab),
            champ_embed_dim=32,
            board_feat_dim=256,
            state_feat_dim=64,
            fused_dim=320,
        )

    # 3. Extract isolated 256D board features
    print(" [+] Extracting isolated 256D board feature latents (bypassing StateMLP / economy)...")
    latents, curated_boards = extract_board_latents(
        trunk=trunk,
        boards=boards,
        batch_size=batch_size,
        device=device,
    )
    print(f" [+] Latents extracted: shape {latents.shape} (N={latents.shape[0]}, Dim={latents.shape[1]})")

    # 4. Fit K-Means Clustering & compute Z-Index
    print(f" [+] Partitioning latent manifold into K={n_clusters} archetype centroids...")
    clusterer = CompositionClusterer(
        n_clusters=n_clusters,
        algorithm="kmeans",
        random_state=42,
    )
    clusterer.fit(latents, curated_boards)

    print(f" [+] Clustering complete:")
    print(f"     - Silhouette Score:       {clusterer.metrics_.get('silhouette_score', 0.0):.4f}")
    print(f"     - Davies-Bouldin Index:  {clusterer.metrics_.get('davies_bouldin_index', 0.0):.4f}")
    print(f"     - Inertia:               {clusterer.metrics_.get('inertia', 0.0):.2f}")

    # 5. Visualizations & WandB Logging
    print(" [+] Generating projection plots and logging metrics...")
    plot_artifacts = log_clustering_to_wandb(
        clusterer=clusterer,
        latents=latents,
        boards=curated_boards,
        wandb_project=wandb_project,
        run_name=run_name,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        output_dir=output_dir,
        use_wandb=use_wandb,
    )

    # 6. Save Artifacts
    print(f" [+] Persisting Z-Index artifacts to: {Path(output_dir).resolve()}")
    saved_paths = save_z_index_artifacts(
        clusterer=clusterer,
        output_dir=output_dir,
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
    )

    print("\n" + "=" * 75)
    print(f" [+] Successfully extracted {len(clusterer.profiles_)} Z-Index composition archetypes!")
    print(f"     - Centroids Checkpoint: {saved_paths['z_index_pt']}")
    print(f"     - Archetype Profiles:   {saved_paths['profiles_json']}")
    print(f"     - Markdown Summary:     {saved_paths['profiles_md']}")
    print("=" * 75 + "\n")

    return {
        "metrics": clusterer.metrics_,
        "saved_paths": saved_paths,
        "plot_artifacts": plot_artifacts,
        "profiles": [asdict(p) for p in clusterer.profiles_],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for composition clustering."""
    parser = argparse.ArgumentParser(
        prog="python -m tft_ai_player.embeddings.cluster",
        description="Extract composition archetypes and Z-Index centroids from curated endgame boards.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\tft-winner-data\set18\players"), help="Path to player CSV dataset directory")
    parser.add_argument("--trunk-checkpoint", type=Path, default=None, help="Path to pre-trained MultiModalFusionTrunk checkpoint (.pt)")
    parser.add_argument("--min-stage", type=int, default=5, help="Minimum stage threshold for endgame boards (default: 5)")
    parser.add_argument("--min-placement", type=int, default=4, help="Maximum placement rank to retain (default: 4 for Top 4)")
    parser.add_argument("--n-clusters", "-k", type=int, default=15, help="Number of composition archetypes K (default: 15)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("models/clustering"), help="Output directory for Z-Index artifacts")
    parser.add_argument("--wandb-project", type=str, default="tft-clustering", help="WandB project name")
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb-group", type=str, default=None, help="WandB group")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for demonstration/testing")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum boards to extract")

    args = parser.parse_args(argv)

    results = run_clustering_pipeline(
        data_dir=args.data_dir,
        trunk_checkpoint=args.trunk_checkpoint,
        min_stage=args.min_stage,
        max_placement=args.min_placement,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        device=args.device,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
        max_samples=args.max_samples,
    )

    return 0 if "error" not in results else 1


if __name__ == "__main__":
    sys.exit(main())
