"""Ranked Ladder Replay Loader for Trace-Driven TFT Simulation.

Indexes authentic Top-1 matches partitioned by rank tier (Bronze, Silver, Gold, Platinum,
Emerald, Diamond, Master, Grandmaster, Challenger) to power adaptive ladder progression in RL.
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .replay_loader import ShadowMatchReplay, ShadowMatchRound

logger = logging.getLogger(__name__)

TIER_ORDER = [
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]

DIVISIONS = [4, 3, 2, 1]


class RankedLadderReplayLoader:
    """Indexes and manages Top-1 replays grouped strictly by rank tier."""

    def __init__(
        self,
        tiers_dir: str | Path = r"D:\tft-winner-data\tiers",
        cache_path: str | Path = "models/rl/ranked_ladder_cache.pkl.gz",
        min_pvp_rounds: int = 15,
        min_final_stage: int = 5,
    ) -> None:
        self.tiers_dir = Path(tiers_dir)
        self.cache_path = Path(cache_path)
        self.min_pvp_rounds = min_pvp_rounds
        self.min_final_stage = min_final_stage
        self.replays_by_tier: dict[str, list[ShadowMatchReplay]] = {t: [] for t in TIER_ORDER}

    def build_or_load_index(self, force_rebuild: bool = False) -> dict[str, list[ShadowMatchReplay]]:
        """Load replays from disk cache or scan per-tier directories to index Top-1 games."""
        if not force_rebuild and self.cache_path.exists():
            try:
                print(f" [+] Loading cached ranked ladder replays from {self.cache_path}...")
                with gzip.open(self.cache_path, "rb") as f:
                    self.replays_by_tier = pickle.load(f)
                total = sum(len(v) for v in self.replays_by_tier.values())
                print(f" [+] Successfully loaded {total} Top-1 replays across {len(self.replays_by_tier)} tiers:")
                for t in TIER_ORDER:
                    print(f"     - {t:12s}: {len(self.replays_by_tier.get(t, [])):4d} replays")
                return self.replays_by_tier
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}. Rebuilding...")

        print(" [*] Scanning player files to build Top-1 ranked ladder replays...")
        self.replays_by_tier = {t: [] for t in TIER_ORDER}

        for tier in TIER_ORDER:
            tier_folder = self.tiers_dir / tier.lower() / "players"
            if not tier_folder.exists():
                continue

            all_csvs = sorted(tier_folder.glob("*.csv"))
            for csv_file in all_csvs:
                try:
                    df = pd.read_csv(csv_file)
                    if "match_id" not in df.columns or "outcome" not in df.columns or "focal_health" not in df.columns:
                        continue

                    for match_id, m_df in df.groupby("match_id"):
                        if len(m_df) < self.min_pvp_rounds:
                            continue

                        # Strict Top-1 filter: player must end alive and with victory in final recorded round
                        last_row = m_df.iloc[-1]
                        if int(last_row["focal_health"]) <= 0 or str(last_row["outcome"]).lower() != "victory":
                            continue

                        last_stage_str = str(m_df["round_stage"].iloc[-1])
                        try:
                            last_st_num = int(last_stage_str.split("-")[0])
                        except (ValueError, IndexError):
                            last_st_num = 0

                        if last_st_num < self.min_final_stage:
                            continue

                        focal_player = str(m_df["focal_player"].iloc[0]) if "focal_player" in m_df.columns else "unknown"
                        final_hp = int(last_row["focal_health"])

                        rounds: list[ShadowMatchRound] = []
                        for _, row in m_df.iterrows():
                            st_str = str(row["round_stage"])
                            try:
                                parts = st_str.split("-")
                                st_num = int(parts[0])
                                r_num = int(parts[1])
                            except (ValueError, IndexError):
                                st_num, r_num = 2, 1

                            raw_json = row.get("input_state_json")
                            opp_board: list[dict[str, Any]] = []
                            if isinstance(raw_json, str):
                                try:
                                    state_dict = json.loads(raw_json)
                                    opp_board = state_dict.get("opponent_board", [])
                                except Exception:
                                    opp_board = []
                            elif isinstance(raw_json, dict):
                                opp_board = raw_json.get("opponent_board", [])

                            if not opp_board:
                                continue

                            rounds.append(
                                ShadowMatchRound(
                                    stage_str=st_str,
                                    stage=st_num,
                                    round_in_stage=r_num,
                                    opponent_board=opp_board,
                                    opponent_level=int(row.get("opponent_level", 7)),
                                    opponent_health=int(row.get("opponent_health", 100)),
                                    focal_health=int(row.get("focal_health", 100)),
                                    focal_level=int(row.get("focal_level", 7)),
                                    focal_gold=int(row.get("focal_gold", 20)),
                                )
                            )

                        if len(rounds) >= self.min_pvp_rounds:
                            replay = ShadowMatchReplay(
                                match_id=str(match_id),
                                focal_player=focal_player,
                                focal_tier=tier,
                                rounds=rounds,
                                final_stage=last_stage_str,
                                final_health=final_hp,
                                total_rounds=len(rounds),
                            )
                            self.replays_by_tier[tier].append(replay)

                except Exception as e:
                    logger.debug(f"Error parsing {csv_file.name}: {e}")

            print(f" [+] Indexed {len(self.replays_by_tier[tier]):4d} Top-1 replays for tier {tier}")

        # Save to cache
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.cache_path, "wb") as f:
                pickle.dump(self.replays_by_tier, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f" [+] Cached ranked ladder replays to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Could not write cache to {self.cache_path}: {e}")

        return self.replays_by_tier

    def sample_replay_for_tier(self, tier: str, rng: random.Random | None = None) -> ShadowMatchReplay:
        """Sample a Top-1 replay from the requested tier (with fallback to adjacent tiers)."""
        r = rng or random
        t = tier.upper()
        pool = self.replays_by_tier.get(t, [])
        if pool:
            return r.choice(pool)

        # Fallback to nearest tier with available replays
        if t in TIER_ORDER:
            idx = TIER_ORDER.index(t)
            for dist in range(1, len(TIER_ORDER)):
                # Try lower first, then higher
                for cand_idx in (idx - dist, idx + dist):
                    if 0 <= cand_idx < len(TIER_ORDER):
                        cand_tier = TIER_ORDER[cand_idx]
                        if self.replays_by_tier.get(cand_tier):
                            return r.choice(self.replays_by_tier[cand_tier])

        raise RuntimeError(f"No replays available across any tier in {self.cache_path}")
