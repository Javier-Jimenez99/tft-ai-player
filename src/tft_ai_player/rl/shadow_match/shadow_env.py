"""Trace-Driven Shadow Match Environment for TFT.

In this environment, the RL agent plays a game of Teamfight Tactics where combat rounds
are evaluated against the real sequence of opponent boards faced by a human Challenger winner.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from ...simulation.actions import TOTAL_DISCRETE_ACTIONS, execute_action, get_action_mask
from ...simulation.combat import CombatResolver, HeuristicCombatResolver
from ...simulation.config import SetData, get_default_set17_data
from ...simulation.gym_env import TFTStateEncoder
from ...simulation.models import ChampionInstance, ChampionPool, ItemInstance, Player, UnitRole
from ...simulation.stage_manager import StageManager
from .replay_loader import ShadowMatchLoader, ShadowMatchReplay, ShadowMatchRound

import re
from ...embeddings.dataset import parse_loc_to_row_col

logger = logging.getLogger(__name__)

_SET18_SPECIAL_ALIASES: dict[str, str] = {
    "da_sentinel18": "TFT18_AncientSentinel",
    "da_18_sentry": "TFT18_AncientSentinel",
    "da_crimsonraptor18": "TFT18_Raptor",
    "da_18_gnarsmall": "TFT18_Gnar",
    "da_18_gnarbig": "TFT18_Gnar",
    "da_elderwood18_stonebarktree": "TFT18_Maokai",
    "da_elderwood18_lifeblossom": "TFT18_Maokai",
    "da_elderwood18_protector": "TFT18_Maokai",
    "da_18_elderdragon": "TFT18_ElderDragon",
    "da_18_yorick_spirit": "TFT18_Yorick",
    "da_18_elisespider": "TFT18_Elise",
    "da_scuttlecrab18": "TFT18_ScuttleCrab",
    "da_krug18_minikrug": "TFT18_Krug",
}


def resolve_canonical_champion(raw_name: str, set_data: SetData) -> str:
    """Resolve replay champion string (e.g. DA_18_Cassiopeia, DA_Vi18) to canonical catalog ID."""
    if not raw_name:
        return "TFT18_TrainingDummy" if "TFT18_TrainingDummy" in set_data.champions else next(iter(set_data.champions))
    if raw_name in set_data.champions:
        return raw_name
    raw_lower = raw_name.strip().lower()
    if raw_lower in _SET18_SPECIAL_ALIASES:
        target = _SET18_SPECIAL_ALIASES[raw_lower]
        if target in set_data.champions:
            return target

    cleaned = re.sub(r"^(tft18_|da_18_|da_)", "", raw_lower)
    cleaned_base = re.sub(r"(18|_ad|_ap|_spirit|_spider)$", "", cleaned)

    for cid in set_data.champions:
        clean_cid = cid.lower().replace("tft18_", "").replace("tft17_", "")
        if cleaned_base == clean_cid:
            return cid
        if cleaned.startswith(clean_cid + "_") or cleaned.startswith(clean_cid):
            return cid

    return "TFT18_TrainingDummy" if "TFT18_TrainingDummy" in set_data.champions else next(iter(set_data.champions))


def resolve_canonical_item(raw_name: str, set_data: SetData) -> str:
    """Resolve replay item string (e.g. DA_HextechGunblade) to canonical item ID."""
    if not raw_name:
        return ""
    if raw_name in set_data.items:
        return raw_name
    clean = re.sub(r"^(DA_|TFT18_|TFT_Item_)?(Component_)?", "", raw_name, flags=re.IGNORECASE)
    candidates = [
        f"TFT_Item_{clean}",
        f"TFT18_Item_{clean}",
        clean,
    ]
    for c in candidates:
        if c in set_data.items:
            return c
    return raw_name


def create_player_from_opponent_board(
    opponent_board: list[dict[str, Any]],
    opponent_level: int = 7,
    opponent_health: int = 100,
    player_id: int = 1,
    set_data: SetData | None = None,
) -> Player:
    """Reconstruct a Player object from an opponent_board dict list with full board and synergy fidelity."""
    sd = set_data or get_default_set17_data()
    p = Player(
        player_id=player_id,
        set_data=sd,
        name="Ranked Opponent",
    )
    p.health = opponent_health
    p.gold = 10
    p.level = max(1, min(10, opponent_level))

    occupied: set[tuple[int, int]] = set()

    for u_dict in opponent_board:
        raw_champ = u_dict.get("unit", "") or u_dict.get("champion", "")
        champ_id = resolve_canonical_champion(raw_champ, sd)
        star = max(1, min(3, int(u_dict.get("tier", 1) or 1)))
        raw_items = u_dict.get("items", []) or []
        items = [resolve_canonical_item(it, sd) for it in raw_items[:3] if it]
        loc = u_dict.get("loc")

        coords = parse_loc_to_row_col(loc)
        if coords is None or coords in occupied:
            # Hex collision fallback: allocate first unoccupied hex on 4x7 grid
            found = False
            for r in range(4):
                for c in range(7):
                    if (r, c) not in occupied:
                        coords = (r, c)
                        found = True
                        break
                if found:
                    break

        if coords is not None:
            occupied.add(coords)
            cdef = sd.champions.get(champ_id)
            cost = cdef.cost if cdef else 1
            inst = ChampionInstance(
                champion_id=champ_id,
                cost=cost,
                star_level=star,
                items=items,
                position=coords,
            )
            p.board[coords] = inst

    return p


class ShadowMatchEnv(gym.Env):
    """Gym environment simulating a TFT match against real Challenger replay opponent streams."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        replay_loader: ShadowMatchLoader | None = None,
        repository: ShadowMatchLoader | None = None,
        set_data: SetData | None = None,
        combat_resolver: CombatResolver | None = None,
        trunk: Any | None = None,
        world_model: torch.nn.Module | None = None,
        max_actions_per_round: int = 15,
        target_z: np.ndarray | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.set_data = set_data or get_default_set17_data()
        self.combat_resolver = combat_resolver or HeuristicCombatResolver()
        self.replay_loader = replay_loader or repository or ShadowMatchLoader()
        self.max_actions_per_round = max_actions_per_round
        self.target_z = target_z

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = TFTStateEncoder(set_data=self.set_data, trunk=trunk, device=self.device)
        self.world_model = world_model.to(self.device).eval() if world_model is not None else None

        self.action_space = spaces.Discrete(TOTAL_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(768,),
            dtype=np.float32,
        )

        self.pool = ChampionPool(set_data=self.set_data)
        self.stage_manager = StageManager(set_data=self.set_data)
        self.player = Player(player_id=0, set_data=self.set_data, name="AlphaStar Agent")
        self.player.health = 100
        self.player.gold = 10
        self.player.level = 3

        self.current_replay: ShadowMatchReplay | None = None
        self.replay_round_idx: int = 0
        self.rounds_survived: int = 0
        self.actions_in_current_round: int = 0
        self.rng = random.Random(42)

    @property
    def game(self) -> Any:
        """Compatibility proxy so LeagueTrainer, Planner and evaluators can interact with ShadowMatchEnv identically to TFTEnv."""
        env_self = self

        class _GameProxy:
            @property
            def pool(self) -> ChampionPool:
                return env_self.pool

            @property
            def stage_manager(self) -> StageManager:
                return env_self.stage_manager

            @property
            def is_over(self) -> bool:
                return not env_self.player.alive or env_self.player.health <= 0

            def get_focal_player(self) -> Player:
                return env_self.player

        return _GameProxy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)

        # 1. Reset Game State
        self.pool = ChampionPool(self.set_data)
        self.stage_manager = StageManager(self.set_data)
        self.player = Player(player_id=0, set_data=self.set_data, name="AlphaStar Agent")
        self.player.health = 100
        self.player.gold = 10
        self.player.level = 3
        self.stage_manager.execute_round_start([self.player], pool=self.pool, rng=self.rng)

        # 2. Sample or assign Shadow Match Replay
        if options and "replay" in options:
            self.current_replay = options["replay"]
        else:
            if len(self.replay_loader) == 0:
                self.replay_loader.build_or_load_index(max_matches=500)
            self.current_replay = self.replay_loader.sample_replay(rng=self.rng)

        self.replay_round_idx = 0
        self.rounds_survived = 0
        self.actions_in_current_round = 0

        obs, mask = self._get_obs_and_masks()
        info = self._get_info(mask)
        return obs, info

    def _get_obs_and_masks(self) -> tuple[np.ndarray, np.ndarray]:
        stage = self.stage_manager.stage
        round_in_stage = self.stage_manager.round_in_stage

        obs_vec, _, _ = self.encoder.extract_state_vector(
            player=self.player,
            stage=stage,
            round_in_stage=round_in_stage,
            target_z=self.target_z,
        )
        mask = get_action_mask(self.player, self.set_data)
        return obs_vec, mask

    def _get_info(self, mask: np.ndarray) -> dict[str, Any]:
        rinfo = self.stage_manager.get_current_round_info()
        human_hp = 100
        if self.current_replay and self.replay_round_idx < len(self.current_replay.rounds):
            human_hp = self.current_replay.rounds[self.replay_round_idx].focal_health

        return {
            "action_mask": mask,
            "stage": rinfo.stage_str,
            "round_type": rinfo.round_type.value,
            "health": self.player.health,
            "human_winner_health": human_hp,
            "health_delta": self.player.health - human_hp,
            "gold": self.player.gold,
            "level": self.player.level,
            "rounds_survived": self.rounds_survived,
            "placement": self._estimate_placement(self.rounds_survived),
            "alive": self.player.alive,
            "replay_match_id": self.current_replay.match_id if self.current_replay else "",
            "replay_focal_tier": self.current_replay.focal_tier if self.current_replay else "",
        }

    def _estimate_placement(self, rounds_survived: int) -> int:
        """Estimate final placement based on rounds survived in a Challenger lobby."""
        if rounds_survived < 14:  # Stage 3
            return 8
        elif rounds_survived < 18:  # Early Stage 4
            return 7
        elif rounds_survived < 22:  # Late Stage 4
            return 6
        elif rounds_survived < 26:  # Stage 5
            return 5
        elif rounds_survived < 30:  # Late Stage 5 (Top 4 threshold)
            return 4
        elif rounds_survived < 34:  # Stage 6
            return 3
        elif rounds_survived < 38:  # Late Stage 6
            return 2
        else:
            return 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0.0
        r_combat = 0.0
        r_interest = 0.0
        r_terminal = 0.0

        is_pass = (action == 0) or (self.actions_in_current_round >= self.max_actions_per_round)

        if not is_pass:
            # Step decision inside shop / bench
            gold_before = self.player.gold
            board_before = self.player.board_unit_count
            stars_before = sum(u.star_level for u in self.player.board.values())

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

                reward = r_int + r_stars
            else:
                reward = -0.02

        else:
            # End of planning phase -> Resolve Combat against Shadow Opponent
            rinfo = self.stage_manager.get_current_round_info()

            if rinfo.is_pvp:
                # Fetch opponent from real replay
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
                    # Fallback if beyond replay rounds
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
                    r_comb = 1.0  # Big reward for defeating real Challenger board
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
                # PvE or carousel round
                self.player.add_gold(2)

            # Advance stage manager
            self.stage_manager.advance_round()
            self.actions_in_current_round = 0

            # Passive round income & shop refresh via stage_manager
            if self.player.alive:
                self.stage_manager.execute_round_start([self.player], pool=self.pool, rng=self.rng)

            # Check Termination
            if not self.player.alive or self.player.health <= 0:
                self.player.health = 0
                terminated = True
                placement = self._estimate_placement(self.rounds_survived)
                # Terminal reward aligned with placement
                r_term = float((8 - placement) * 0.5)
                r_terminal += r_term
                reward += r_term

            # Check if finished entire replay
            if self.current_replay and self.replay_round_idx >= len(self.current_replay.rounds):
                terminated = True
                reward += 5.0  # Outlasted the entire Challenger match!
                r_terminal += 5.0

        obs, mask = self._get_obs_and_masks()
        info = self._get_info(mask)
        info["reward_breakdown"] = {
            "r_combat": float(r_combat),
            "r_interest": float(r_interest),
            "r_terminal": float(r_terminal),
            "r_micro": 0.0,
            "r_macro": 0.0,
            "r_env": float(reward),
        }
        if terminated:
            info["final_placement"] = self._estimate_placement(self.rounds_survived)

        return obs, reward, terminated, truncated, info
