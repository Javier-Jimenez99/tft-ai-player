"""Dataset loader for full-match Elo regression and OOD modeling.

Extracts complete match trajectories, generates dense game feature vectors,
and computes two distinct Elo target labels:
  - Target A: Match Lobby Elo (`match_elo`)
  - Target B: Player Last/Peak Elo (`player_last_elo`)
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, extract_game_features


TIER_BASE_ELO = {
    "IRON": 200.0,
    "BRONZE": 600.0,
    "SILVER": 1000.0,
    "GOLD": 1400.0,
    "PLATINUM": 1800.0,
    "EMERALD": 2200.0,
    "DIAMOND": 2600.0,
    "MASTER": 3000.0,
    "GRANDMASTER": 3500.0,
    "CHALLENGER": 4200.0,
}

DIV_OFFSET = {"IV": 0.0, "III": 100.0, "II": 200.0, "I": 300.0}


def parse_elo_from_record(
    tier_str: str | None,
    numeric_val: str | float | None,
) -> float | None:
    """Extract continuous numerical Elo from tier string and optional numeric score."""
    if numeric_val and str(numeric_val).strip() and str(numeric_val) != "None":
        try:
            val = float(numeric_val)
            if val > 50.0:
                return val
        except ValueError:
            pass

    if not tier_str or not isinstance(tier_str, str):
        return None

    s = tier_str.upper().strip()
    for t_name in [
        "GRANDMASTER",
        "CHALLENGER",
        "MASTER",
        "DIAMOND",
        "EMERALD",
        "PLATINUM",
        "GOLD",
        "SILVER",
        "BRONZE",
        "IRON",
    ]:
        if t_name in s:
            base = TIER_BASE_ELO[t_name]
            # Match LP
            import re
            lp_match = re.search(r"(\d+)\s*LP", s)
            lp = float(lp_match.group(1)) if lp_match else 50.0

            if t_name in ("MASTER", "GRANDMASTER", "CHALLENGER"):
                return base + lp

            div_off = 150.0
            for div, off in DIV_OFFSET.items():
                if f" {div} " in s or s.endswith(f" {div}"):
                    div_off = off
                    break
            return base - 150.0 + div_off + (lp % 100)
    return None


class FullMatchEloDataset:
    """Dataset aggregating complete match trajectories into game features and dual Elo labels."""

    def __init__(
        self,
        data_dir: str | Path = "D:/tft-winner-data/set18/players",
        max_files: int | None = None,
        min_rounds_per_match: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.matches: list[dict[str, Any]] = []
        self.feature_matrix: np.ndarray = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        self.match_elos: np.ndarray = np.empty(0, dtype=np.float32)
        self.player_last_elos: np.ndarray = np.empty(0, dtype=np.float32)

        if self.data_dir.exists():
            self._load_dataset(max_files=max_files, min_rounds=min_rounds_per_match)

    def _load_dataset(self, max_files: int | None = None, min_rounds: int = 5) -> None:
        csv_files = list(self.data_dir.glob("*.csv")) if self.data_dir.is_dir() else [self.data_dir]
        if max_files:
            csv_files = csv_files[:max_files]

        # 1. Group observations into player match trajectories
        player_matches: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        player_dates: dict[str, dict[str, str]] = defaultdict(dict)
        player_match_elo: dict[str, dict[str, float]] = defaultdict(dict)
        player_focal_elo: dict[str, dict[str, float]] = defaultdict(dict)

        for csv_path in csv_files:
            try:
                with open(csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        mid = str(r.get("match_id", "")).strip()
                        p = str(r.get("focal_player", "")).strip()
                        if not mid or not p:
                            continue

                        # Parse state JSON if necessary
                        state_raw = r.get("input_state_json")
                        focal_board = []
                        if isinstance(state_raw, str) and state_raw.strip():
                            try:
                                state_dict = json.loads(state_raw)
                                focal_board = state_dict.get("focal_board", [])
                            except Exception:
                                focal_board = []

                        dt = str(r.get("game_datetime", "")).strip()
                        m_elo = parse_elo_from_record(
                            r.get("avg_match_rating"), r.get("avg_match_rating_numeric")
                        )
                        f_elo = parse_elo_from_record(
                            r.get("focal_tier") or r.get("tier_category"), r.get("focal_rating_numeric")
                        )

                        player_matches[p][mid].append({
                            "round_stage": r.get("round_stage", "2-1"),
                            "focal_health": r.get("focal_health", 100),
                            "focal_level": r.get("focal_level", 1),
                            "focal_gold": r.get("focal_gold", 0),
                            "outcome": r.get("outcome", ""),
                            "label": r.get("label"),
                            "focal_board": focal_board,
                        })

                        if dt and mid not in player_dates[p]:
                            player_dates[p][mid] = dt
                        if m_elo and mid not in player_match_elo[p]:
                            player_match_elo[p][mid] = m_elo
                        if f_elo and mid not in player_focal_elo[p]:
                            player_focal_elo[p][mid] = f_elo
            except Exception:
                continue

        # 2. Determine each player's latest Elo (Target B)
        player_last_known_elo: dict[str, float] = {}
        for p, m_dict in player_matches.items():
            # Sort matches chronologically
            m_list = list(m_dict.keys())
            m_list.sort(key=lambda mid: player_dates[p].get(mid, ""))
            # Get latest valid focal elo
            for mid in reversed(m_list):
                if mid in player_focal_elo[p]:
                    player_last_known_elo[p] = player_focal_elo[p][mid]
                    break
                if mid in player_match_elo[p]:
                    player_last_known_elo[p] = player_match_elo[p][mid]
                    break

        # 3. Extract game features and pair with targets
        features_list = []
        y_match_list = []
        y_player_list = []

        for p, m_dict in player_matches.items():
            last_elo = player_last_known_elo.get(p)

            for mid, traj in m_dict.items():
                if len(traj) < min_rounds:
                    continue

                m_elo = player_match_elo[p].get(mid) or player_focal_elo[p].get(mid) or last_elo
                p_elo = last_elo or m_elo

                if m_elo is None or p_elo is None:
                    continue

                feat = extract_game_features(traj)
                features_list.append(feat)
                y_match_list.append(m_elo)
                y_player_list.append(p_elo)

                self.matches.append({
                    "player": p,
                    "match_id": mid,
                    "rounds": len(traj),
                    "match_elo": m_elo,
                    "player_last_elo": p_elo,
                })

        if features_list:
            self.feature_matrix = np.vstack(features_list)
            self.match_elos = np.array(y_match_list, dtype=np.float32)
            self.player_last_elos = np.array(y_player_list, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.matches)
