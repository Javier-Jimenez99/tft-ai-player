"""Ranked Ladder Environment (AlphaStar v6.2).

In this environment, the RL agent plays against authentic Top-1 replays corresponding
to its current ranked division (e.g. Gold IV -> Challenger). Wins and high placements
grant League Points (LP) and division promotions, while low placements cause LP loss
and demotions.

Includes fundamental action reward shaping to encourage leveling and interest management.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...simulation.actions import TOTAL_DISCRETE_ACTIONS, execute_action
from ...simulation.config import SetData, get_default_set17_data
from ...simulation.models import Player
from .ranked_loader import DIVISIONS, TIER_ORDER, RankedLadderReplayLoader
from .shadow_env import ShadowMatchEnv, create_player_from_opponent_board

logger = logging.getLogger(__name__)


@dataclass
class RankedStatus:
    """Agent's current standing on the Ranked Ladder."""

    tier: str = "GOLD"
    division: int = 4
    lp: int = 0
    total_games: int = 0
    total_wins: int = 0
    total_top4: int = 0
    mmr: float = 1200.0  # Elo-like hidden rating

    @property
    def display_rank(self) -> str:
        if self.tier in ("MASTER", "GRANDMASTER", "CHALLENGER"):
            return f"{self.tier} ({self.lp} LP)"
        return f"{self.tier} {self.division} ({self.lp} LP)"

    def update_lp(self, placement: int) -> int:
        """Update LP and handle promotions/demotions based on match placement.
        
        Returns the delta LP (+/-).
        Placement LP table:
          1st: +40 LP
          2nd: +30 LP
          3rd: +20 LP
          4th: +10 LP
          5th: -10 LP
          6th: -20 LP
          7th: -30 LP
          8th: -40 LP
        """
        self.total_games += 1
        if placement == 1:
            self.total_wins += 1
        if placement <= 4:
            self.total_top4 += 1

        lp_deltas = {
            1: 40,
            2: 30,
            3: 20,
            4: 10,
            5: -10,
            6: -20,
            7: -30,
            8: -40,
        }
        delta = lp_deltas.get(placement, 0)
        self.lp += delta
        self.mmr += float(delta * 2.5)

        # Handle Apex tiers (Master+)
        if self.tier in ("MASTER", "GRANDMASTER", "CHALLENGER"):
            if self.lp >= 500:
                self.tier = "CHALLENGER"
            elif self.lp >= 250:
                self.tier = "GRANDMASTER"
            elif self.lp < 0:
                # Demote to Diamond 1
                self.tier = "DIAMOND"
                self.division = 1
                self.lp = 75
            return delta

        # Handle Standard Tiers (Bronze - Diamond)
        while self.lp >= 100 and self.tier not in ("MASTER", "GRANDMASTER", "CHALLENGER"):
            surplus = self.lp - 100
            if self.division > 1:
                self.division -= 1
                self.lp = surplus
            else:
                # Promotion to next tier
                tier_idx = TIER_ORDER.index(self.tier)
                if tier_idx + 1 < len(TIER_ORDER):
                    self.tier = TIER_ORDER[tier_idx + 1]
                    if self.tier in ("MASTER", "GRANDMASTER", "CHALLENGER"):
                        self.division = 1
                        self.lp = surplus
                        if self.lp >= 500:
                            self.tier = "CHALLENGER"
                        elif self.lp >= 250:
                            self.tier = "GRANDMASTER"
                        break
                    else:
                        self.division = 4
                        self.lp = surplus
                else:
                    self.lp = 100
                    break

        while self.lp < 0 and self.tier not in ("MASTER", "GRANDMASTER", "CHALLENGER"):
            surplus = self.lp  # negative
            if self.division < 4:
                self.division += 1
                self.lp = 100 + surplus
            else:
                # Demotion to previous tier
                tier_idx = TIER_ORDER.index(self.tier)
                if tier_idx > 0:
                    self.tier = TIER_ORDER[tier_idx - 1]
                    self.division = 1
                    self.lp = 100 + surplus
                else:
                    self.lp = 0
                    break

        return delta


class RankedLadderEnv(ShadowMatchEnv):
    """Gymnasium environment that samples Top-1 replays matching the agent's ranked tier."""

    def __init__(
        self,
        ladder_loader: RankedLadderReplayLoader | None = None,
        initial_tier: str = "GOLD",
        initial_division: int = 4,
        initial_lp: int = 0,
        set_data: SetData | None = None,
        **kwargs: Any,
    ) -> None:
        self.ladder_loader = ladder_loader or RankedLadderReplayLoader()
        if not self.ladder_loader.replays_by_tier.get("GOLD"):
            self.ladder_loader.build_or_load_index()

        super().__init__(set_data=set_data, **kwargs)

        self.ranked_status = RankedStatus(
            tier=initial_tier.upper(),
            division=initial_division,
            lp=initial_lp,
        )
        self.last_lp_delta: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and sample a Top-1 match from the agent's current tier."""
        super().reset(seed=seed, options=options)

        # Sample Top-1 replay matching current tier
        self.current_replay = self.ladder_loader.sample_replay_for_tier(
            tier=self.ranked_status.tier,
            rng=self.rng,
        )
        self.replay_round_idx = 0

        obs, mask = self._get_obs_and_masks()
        info = self._get_info(mask)
        info["ranked_status"] = self.ranked_status.display_rank
        info["ranked_tier"] = self.ranked_status.tier
        info["ranked_division"] = self.ranked_status.division
        info["ranked_lp"] = self.ranked_status.lp
        return obs, info

    def _estimate_placement(self, rounds_survived: int) -> int:
        """Empirical placement mapping from human survival rounds.
        
        Calibrated against real dataset distributions:
          - surviving < 15 rounds -> 8th
          - 15..16 rounds -> 7th
          - 17..18 rounds -> 6th
          - 19 rounds -> 5th
          - 20 rounds -> 4th (Top-4 threshold)
          - 21 rounds -> 3rd
          - 22 rounds -> 2nd
          - >= 23 rounds -> 1st
        """
        if rounds_survived < 15:
            return 8
        elif rounds_survived < 17:
            return 7
        elif rounds_survived < 19:
            return 6
        elif rounds_survived < 20:
            return 5
        elif rounds_survived < 21:
            return 4
        elif rounds_survived < 22:
            return 3
        elif rounds_survived < 23:
            return 2
        else:
            return 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step environment with fundamental reward shaping (leveling and economy)."""
        terminated = False
        truncated = False
        reward = 0.0
        r_combat = 0.0
        r_interest = 0.0
        r_terminal = 0.0
        r_shaping = 0.0

        is_pass = (action == 0) or (self.actions_in_current_round >= self.max_actions_per_round)

        if not is_pass:
            gold_before = self.player.gold
            level_before = self.player.level
            stars_before = sum(u.star_level for u in self.player.board.values())
            board_len_before = len(self.player.board)

            success = execute_action(self.player, self.pool, self.set_data, action)
            self.actions_in_current_round += 1

            if success:
                # Interest bonus delta
                int_prev = min(5, gold_before // 10)
                int_curr = min(5, self.player.gold // 10)
                r_int = float(int_curr - int_prev) * 0.05
                r_interest = r_int

                # Star upgrade bonus
                stars_after = sum(u.star_level for u in self.player.board.values())
                r_stars = max(0, stars_after - stars_before) * 0.15

                # Fundamentals Shaping: encourage leveling up and deploying units
                if action == 7:  # BUY_XP
                    # When investing economy into tempo/XP, don't penalize interest drop
                    if r_int < 0:
                        r_interest = 0.0
                    r_shaping += 0.10

                    # Surplus gold investment (>50g slow-leveling fundamentals)
                    if gold_before >= 50:
                        r_shaping += 0.05

                    if self.player.level > level_before:
                        # Milestone reward for gaining a level (unlocks board slot + higher cost champion odds)
                        r_shaping += 0.50

                # Deploy unit bonus (moving bench unit onto active board hex)
                elif 45 <= action <= 72:
                    if len(self.player.board) > board_len_before:
                        r_shaping += 0.10

                reward = r_interest + r_stars + r_shaping
            else:
                reward = -0.02

        else:
            # End of planning phase -> Resolve Combat against Shadow Opponent
            rinfo = self.stage_manager.get_current_round_info()

            if rinfo.is_pvp:
                if self.current_replay and self.replay_round_idx < len(self.current_replay.rounds):
                    rep_round = self.current_replay.rounds[self.replay_round_idx]
                    shadow_opp = create_player_from_opponent_board(
                        opponent_board=rep_round.opponent_board,
                        opponent_level=rep_round.opponent_level,
                        opponent_health=rep_round.opponent_health,
                        set_data=self.set_data,
                    )
                    human_hp = rep_round.focal_health
                else:
                    shadow_opp = Player(player_id=99, set_data=self.set_data, name="Fallback Opponent")
                    shadow_opp.health = 100
                    shadow_opp.level = self.player.level
                    human_hp = 10

                # Resolve Combat
                combat_res = self.combat_resolver.resolve(
                    player_a=self.player,
                    player_b=shadow_opp,
                    is_ghost_b=False,
                    stage=rinfo.stage,
                    stage_str=rinfo.stage_str,
                    set_data=self.set_data,
                    rng=self.rng,
                )

                won = (combat_res.winner_id == self.player.player_id)
                if won:
                    self.player.add_gold(1)
                    if self.player.streak < 0:
                        self.player.streak = 1
                    else:
                        self.player.streak += 1
                    r_comb = 1.0
                else:
                    dmg = combat_res.damage_dealt
                    self.player.health -= dmg
                    if self.player.streak > 0:
                        self.player.streak = -1
                    else:
                        self.player.streak -= 1
                    r_comb = -0.05 * dmg

                # Health preservation bonus vs human winner
                hp_delta = self.player.health - human_hp
                r_comb += float(np.clip(hp_delta * 0.01, -0.3, 0.3))
                r_combat = r_comb
                reward = r_comb

                self.replay_round_idx += 1
                self.rounds_survived += 1

            elif rinfo.is_pve or rinfo.is_carousel:
                self.player.add_gold(2)

            # Advance stage manager
            self.stage_manager.advance_round()
            self.actions_in_current_round = 0

            if self.player.alive:
                self.stage_manager.execute_round_start([self.player], pool=self.pool, rng=self.rng)

            # Check Termination
            placement: int | None = None
            if not self.player.alive or self.player.health <= 0:
                self.player.health = 0
                terminated = True
                placement = self._estimate_placement(self.rounds_survived)
                self.last_lp_delta = self.ranked_status.update_lp(placement)
                r_term = float((8 - placement) * 0.5)
                r_terminal += r_term
                reward += r_term

            # Check if survived the entire Top-1 match replay
            elif self.current_replay and self.replay_round_idx >= len(self.current_replay.rounds):
                terminated = True
                # Evaluate placement based on rounds survived vs total replay length
                # Only award 1st place if agent survived >= 22 rounds AND has positive health
                if self.rounds_survived >= 22 and self.player.health > 0:
                    placement = 1
                    reward += 5.0
                    r_terminal += 5.0
                else:
                    placement = self._estimate_placement(self.rounds_survived)
                    reward += float((8 - placement) * 0.5)
                    r_terminal += float((8 - placement) * 0.5)

                self.last_lp_delta = self.ranked_status.update_lp(placement)

        obs, mask = self._get_obs_and_masks()
        info = self._get_info(mask)
        info["ranked_status"] = self.ranked_status.display_rank
        info["ranked_tier"] = self.ranked_status.tier
        info["ranked_division"] = self.ranked_status.division
        info["ranked_lp"] = self.ranked_status.lp
        info["last_lp_delta"] = self.last_lp_delta
        info["reward_breakdown"] = {
            "r_combat": float(r_combat),
            "r_interest": float(r_interest),
            "r_terminal": float(r_terminal),
            "r_shaping": float(r_shaping),
            "r_micro": 0.0,
            "r_macro": 0.0,
            "r_env": float(reward),
        }
        if terminated and placement is not None:
            info["placement"] = placement
            info["final_placement"] = placement

        return obs, reward, terminated, truncated, info
