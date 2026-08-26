"""Gymnasium reinforcement learning environment wrapper for Teamfight Tactics."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS, get_action_mask
from tft_ai_player.simulation.bots import BaseBot, StandardTempoBot
from tft_ai_player.simulation.combat import CombatResolver, HeuristicCombatResolver
from tft_ai_player.simulation.config import SetData, get_default_set17_data
from tft_ai_player.simulation.game import TFTGame
from tft_ai_player.simulation.observations import ObservationEncoder


class TFTEnv(gym.Env):
    """Reinforcement learning environment for training autonomous TFT agents.

    Compliant with the standard Gymnasium (v1.0+) interface with invalid action masking.
    """

    metadata = {"render_modes": ["human", "ansi", "text"]}

    def __init__(
        self,
        set_data: SetData | None = None,
        combat_resolver: CombatResolver | None = None,
        bot_factory: type[BaseBot] = StandardTempoBot,
        use_flat_obs: bool = False,
        render_mode: str | None = None,
        max_actions_per_round: int = 50,
    ) -> None:
        super().__init__()

        self.set_data = set_data or get_default_set17_data()
        self.combat_resolver = combat_resolver or HeuristicCombatResolver()
        self.bot_factory = bot_factory
        self.use_flat_obs = use_flat_obs
        self.render_mode = render_mode
        self.max_actions_per_round = max_actions_per_round

        self.encoder = ObservationEncoder(self.set_data)
        self.game = TFTGame(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            bot_factory=self.bot_factory,
        )

        self.actions_in_current_round: int = 0
        self.last_health: int = 100

        # Action Space: 1721 Discrete Actions
        self.action_space = spaces.Discrete(TOTAL_DISCRETE_ACTIONS)

        # Observation Space
        if self.use_flat_obs:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1000.0,
                shape=(self.encoder.flat_observation_dim,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "player_stats": spaces.Box(low=0.0, high=1000.0, shape=(9,), dtype=np.float32),
                    "board": spaces.Box(low=0.0, high=500.0, shape=(28, 5), dtype=np.float32),
                    "bench": spaces.Box(low=0.0, high=500.0, shape=(9, 5), dtype=np.float32),
                    "item_bench": spaces.Box(low=0.0, high=500.0, shape=(10,), dtype=np.float32),
                    "shop": spaces.Box(low=0.0, high=500.0, shape=(5,), dtype=np.float32),
                    "opponents": spaces.Box(low=0.0, high=1000.0, shape=(7, 8), dtype=np.float32),
                    "opponents_boards": spaces.Box(low=0.0, high=500.0, shape=(7, 28, 5), dtype=np.float32),
                    "opponents_benches": spaces.Box(low=0.0, high=500.0, shape=(7, 9, 5), dtype=np.float32),
                    "pool_counts": spaces.Box(low=0.0, high=100.0, shape=(self.encoder.num_champs,), dtype=np.float32),
                    "action_mask": spaces.Box(low=0, high=1, shape=(TOTAL_DISCRETE_ACTIONS,), dtype=bool),
                }
            )

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        focal = self.game.get_focal_player()
        opponents = self.game.get_opponents(focal.player_id)
        if self.use_flat_obs:
            return self.encoder.encode_flat(focal, opponents, self.game.pool, self.game.stage_manager)
        return self.encoder.encode_dict(focal, opponents, self.game.pool, self.game.stage_manager)

    def _get_info(self) -> dict[str, Any]:
        focal = self.game.get_focal_player()
        mask = get_action_mask(focal, self.set_data)
        rinfo = self.game.stage_manager.get_current_round_info()
        return {
            "action_mask": mask,
            "stage": rinfo.stage_str,
            "round_type": rinfo.round_type.value,
            "health": focal.health,
            "gold": focal.gold,
            "level": focal.level,
            "placement": focal.placement,
            "alive": focal.alive,
            "is_over": self.game.is_over,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray] | np.ndarray, dict[str, Any]]:
        """Reset the environment to the beginning of a fresh TFT match."""
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self.actions_in_current_round = 0
        self.last_health = self.game.get_focal_player().health

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode in ("human", "ansi", "text"):
            self.render()

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray] | np.ndarray, float, bool, bool, dict[str, Any]]:
        """Perform a discrete micro-action or advance the round on PASS (0)."""
        focal = self.game.get_focal_player()

        if not focal.alive or self.game.is_over:
            obs = self._get_obs()
            info = self._get_info()
            return obs, 0.0, True, False, info

        reward = 0.0
        self.actions_in_current_round += 1

        # Check if action is PASS (0) or exceeded action limit
        if action == 0 or self.actions_in_current_round >= self.max_actions_per_round:
            # 1. Opponent AI planning phase
            self.game.execute_bot_turns()

            # 2. Resolve round combat & advance
            health_before = focal.health
            combat_results = self.game.resolve_round_phase()
            health_after = focal.health
            delta_hp = health_after - health_before

            # Round outcome reward
            if delta_hp < 0:
                reward += (delta_hp / 100.0) * 0.5  # Negative reward for damage taken
            elif self.game.stage_manager.get_current_round_info().is_pvp:
                reward += 0.1  # Positive reward for winning PvP round

            # Small survival step reward
            if focal.alive:
                reward += 0.02

            # Tournament placement reward on game end / elimination
            if not focal.alive or self.game.is_over:
                placement = focal.placement or (1 if focal.alive else 8)
                # Map 1st -> +1.0, 2nd -> +0.6, 3rd -> +0.4, 4th -> +0.2, 5th -> -0.2, 6th -> -0.4, 7th -> -0.6, 8th -> -1.0
                placement_rewards = {
                    1: 1.0,
                    2: 0.6,
                    3: 0.4,
                    4: 0.2,
                    5: -0.2,
                    6: -0.4,
                    7: -0.6,
                    8: -1.0,
                }
                reward += placement_rewards.get(placement, 0.0)

            self.actions_in_current_round = 0
        else:
            # Execute micro-action (buy, sell, move, equip, roll, level)
            valid = self.game.step_player_action(focal.player_id, action)
            if not valid:
                reward -= 0.01  # Minor penalty for invalid action attempt

        terminated = not focal.alive or self.game.is_over
        truncated = False
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode in ("human", "ansi", "text"):
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render ASCII representation of current game state."""
        focal = self.game.get_focal_player()
        rinfo = self.game.stage_manager.get_current_round_info()

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(
            f" [TFT SIMULATION] | Stage: {rinfo.stage_str} ({rinfo.round_type.value}) | Round: {self.game.stage_manager.total_rounds_elapsed}"
        )
        lines.append("=" * 70)
        lines.append(
            f" HP: {focal.health}/100 | Gold: {focal.gold}g | Level: {focal.level} (XP: {focal.exp}/{self.set_data.level_exp.get(focal.level, 'MAX')}) | Streak: {focal.streak:+d}"
        )
        lines.append("-" * 70)

        # Board Grid
        lines.append(" FIELDED BOARD (4x7 Grid):")
        for r in range(self.set_data.board_rows):
            row_str = f"  Row {r}: "
            hex_cells: list[str] = []
            for c in range(self.set_data.board_cols):
                unit = focal.board.get((r, c))
                if unit:
                    star_str = "*" * unit.star_level
                    c_short = unit.champion_id.replace("TFT17_", "")[:7]
                    item_count = len(unit.items)
                    hex_cells.append(f"[{c_short} {star_str}|{item_count}i]")
                else:
                    hex_cells.append("[   .   ]")
            lines.append(row_str + " ".join(hex_cells))

        # Bench
        lines.append("-" * 70)
        bench_cells: list[str] = []
        for idx, slot in enumerate(focal.bench):
            if slot:
                star_str = "*" * slot.star_level
                c_short = slot.champion_id.replace("TFT17_", "")[:6]
                bench_cells.append(f"({idx}:{c_short} {star_str})")
            else:
                bench_cells.append(f"({idx}: - )")
        lines.append(" BENCH: " + " ".join(bench_cells))

        # Items
        item_names = [it.name for it in focal.item_bench]
        lines.append(f" ITEM BENCH ({len(focal.item_bench)}/10): {', '.join(item_names) if item_names else 'Empty'}")

        # Shop
        shop_cards: list[str] = []
        for idx, card in enumerate(focal.shop.slots):
            if card:
                cdef = self.set_data.champions.get(card)
                cost = cdef.cost if cdef else 1
                c_short = card.replace("TFT17_", "")
                shop_cards.append(f"[{idx}] {c_short} ({cost}g)")
            else:
                shop_cards.append(f"[{idx}] (Bought)")
        lock_str = "LOCKED" if focal.shop.locked else "Unlocked"
        lines.append(f" SHOP ({lock_str}): " + " | ".join(shop_cards))

        # Active Traits
        active_traits = focal.get_active_traits()
        active_str = ", ".join(f"{t}: Tier {tier}" for t, tier in active_traits.items() if tier > 0)
        lines.append(f" ACTIVE TRAITS: {active_str if active_str else 'None'}")

        # Opponents Standings
        lines.append("-" * 70)
        lines.append(" LOBBY STANDINGS:")
        for p in self.game.players:
            status = f"Dead (#{p.placement})" if not p.alive else f"HP: {p.health} | Lv:{p.level} | {p.gold}g | Board: {len(p.board)}u"
            is_focal_marker = "-> " if p.player_id == focal.player_id else "   "
            lines.append(f"  {is_focal_marker}Player {p.player_id}: {status}")

        lines.append("=" * 70)

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output
