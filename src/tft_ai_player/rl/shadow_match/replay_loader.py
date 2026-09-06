"""Shadow Match Replay Loader for Trace-Driven TFT Simulation.

Extracts and indexes real human matches from high-elo players (Challenger/Master/Diamond)
to provide realistic chronological sequences of opponent boards for the RL agent.
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

logger = logging.getLogger(__name__)


@dataclass
class ShadowMatchRound:
    """A single PvP combat round within a real human match."""

    stage_str: str
    stage: int
    round_in_stage: int
    opponent_board: list[dict[str, Any]]
    opponent_level: int
    opponent_health: int
    focal_health: int
    focal_level: int
    focal_gold: int = 0


@dataclass
class ShadowMatchReplay:
    """A complete chronological sequence of rounds from a high-placement match."""

    match_id: str
    focal_player: str
    focal_tier: str
    rounds: list[ShadowMatchRound] = field(default_factory=list)
    final_stage: str = ""
    final_health: int = 100
    total_rounds: int = 0

    def get_round(self, stage: int, round_in_stage: int) -> ShadowMatchRound | None:
        """Find round by stage numbers or closest match."""
        target_str = f"{stage}-{round_in_stage}"
        for r in self.rounds:
            if r.stage_str == target_str or (r.stage == stage and r.round_in_stage == round_in_stage):
                return r
        # Fallback to closest stage
        stage_rounds = [r for r in self.rounds if r.stage == stage]
        if stage_rounds:
            return stage_rounds[min(len(stage_rounds) - 1, max(0, round_in_stage - 1))]
        return None

    def get_round_by_index(self, index: int) -> ShadowMatchRound:
        """Get round by chronological index [0, len-1]."""
        idx = max(0, min(len(self.rounds) - 1, index))
        return self.rounds[idx]


class ShadowMatchLoader:
    """Scans player CSV files and indexes high-survival matches for trace-driven simulation."""

    def __init__(
        self,
        data_dirs: Sequence[str | Path] | None = None,
        data_dir: str | Path | None = None,
        min_pvp_rounds: int = 18,
        min_final_stage: int = 5,
        cache_path: str | Path = "models/rl/shadow_matches_cache.pkl.gz",
    ) -> None:
        if data_dir is not None:
            self.data_dirs = [Path(data_dir)]
        elif data_dirs is not None:
            self.data_dirs = [Path(d) for d in data_dirs]
        else:
            self.data_dirs = [
                Path("D:/tft-winner-data/set18/players"),
                Path("D:/tft-winner-data/tiers/challenger/players"),
                Path("D:/tft-winner-data/tiers/grandmaster/players"),
                Path("D:/tft-winner-data/tiers/master/players"),
            ]
        self.min_pvp_rounds = min_pvp_rounds
        self.min_final_stage = min_final_stage
        self.cache_path = Path(cache_path)
        self.replays: list[ShadowMatchReplay] = []

    def build_or_load_index(self, max_matches: int = 2500, force_rebuild: bool = False) -> list[ShadowMatchReplay]:
        """Load replays from cache or scan directories to build index."""
        if not force_rebuild and self.cache_path.exists():
            try:
                print(f" [+] Loading cached shadow matches from {self.cache_path}...")
                with gzip.open(self.cache_path, "rb") as f:
                    self.replays = pickle.load(f)
                print(f" [+] Successfully loaded {len(self.replays)} high-elo shadow match replays.")
                return self.replays
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}. Rebuilding...")

        print(" [*] Scanning player files to build shadow match replays...")
        all_csvs: list[Path] = []
        for d in self.data_dirs:
            if d.exists():
                all_csvs.extend(sorted(d.glob("*.csv")))

        print(f" [*] Found {len(all_csvs)} player CSV files. Indexing top-placement matches...")
        parsed_replays: list[ShadowMatchReplay] = []

        for csv_file in all_csvs:
            if len(parsed_replays) >= max_matches:
                break
            try:
                df = pd.read_csv(csv_file)
                if "match_id" not in df.columns or "round_stage" not in df.columns or "input_state_json" not in df.columns:
                    continue

                for match_id, m_df in df.groupby("match_id"):
                    if len(m_df) < self.min_pvp_rounds:
                        continue

                    # Verify final stage reached
                    last_stage_str = str(m_df["round_stage"].iloc[-1])
                    try:
                        last_st_num = int(last_stage_str.split("-")[0])
                    except (ValueError, IndexError):
                        last_st_num = 0

                    if last_st_num < self.min_final_stage:
                        continue

                    # Build replay object
                    focal_player = str(m_df["focal_player"].iloc[0]) if "focal_player" in m_df.columns else "unknown"
                    focal_tier = str(m_df["focal_tier"].iloc[0]) if "focal_tier" in m_df.columns else "CHALLENGER"
                    final_hp = int(m_df["focal_health"].iloc[-1]) if "focal_health" in m_df.columns else 0

                    rounds: list[ShadowMatchRound] = []
                    for _, row in m_df.iterrows():
                        st_str = str(row["round_stage"])
                        try:
                            parts = st_str.split("-")
                            st_num = int(parts[0])
                            r_num = int(parts[1])
                        except (ValueError, IndexError):
                            st_num, r_num = 2, 1

                        # Parse opponent board
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
                            focal_tier=focal_tier,
                            rounds=rounds,
                            final_stage=last_stage_str,
                            final_health=final_hp,
                            total_rounds=len(rounds),
                        )
                        parsed_replays.append(replay)
                        if len(parsed_replays) >= max_matches:
                            break

            except Exception as e:
                logger.debug(f"Error parsing {csv_file.name}: {e}")

        self.replays = parsed_replays
        print(f" [+] Indexed {len(self.replays)} high-elo shadow match replays.")

        # Save to cache
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.cache_path, "wb") as f:
                pickle.dump(self.replays, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f" [+] Cached shadow matches to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Could not write cache to {self.cache_path}: {e}")

        return self.replays

    def sample_replay(self, rng: random.Random | None = None) -> ShadowMatchReplay:
        """Select a random replay from the indexed pool."""
        if not self.replays:
            raise RuntimeError("No shadow match replays available. Call build_or_load_index() first.")
        r = rng or random
        return r.choice(self.replays)

    def sample_matches(self, n: int = 50, min_rounds: int = 18, rng: random.Random | None = None) -> list[ShadowMatchReplay]:
        """Sample up to n replays meeting criteria."""
        if not self.replays:
            self.build_or_load_index()
        candidates = [r for r in self.replays if len(r.rounds) >= min_rounds]
        if not candidates:
            candidates = self.replays
        r = rng or random
        if len(candidates) <= n:
            return list(candidates)
        return r.sample(candidates, k=n)

    def load_repository(self, max_matches: int = 2500, force_rebuild: bool = False) -> ShadowMatchLoader:
        """Alias for build_or_load_index returning self for method chaining."""
        self.build_or_load_index(max_matches=max_matches, force_rebuild=force_rebuild)
        return self

    def __len__(self) -> int:
        return len(self.replays)


# Type alias for cleaner semantics
ShadowMatchRepository = ShadowMatchLoader
