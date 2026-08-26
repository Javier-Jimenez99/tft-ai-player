"""Baseline automated opponent policies for 8-player lobby simulation."""

from __future__ import annotations

import random
from typing import Protocol

from tft_ai_player.simulation.config import SetData, UnitRole
from tft_ai_player.simulation.models import ChampionInstance, ChampionPool, Player


class BaseBot(Protocol):
    """Protocol for automated player decision policies during the planning phase."""

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
    ) -> None:
        """Execute bot planning phase actions."""
        ...


class StandardTempoBot:
    """Standard competitive TFT tempo bot following meta leveling curves and synergy building."""

    def __init__(self) -> None:
        # Standard tempo targets: (stage, round) -> target level
        self.tempo_schedule: dict[tuple[int, int], int] = {
            (2, 1): 4,
            (2, 5): 5,
            (3, 2): 6,
            (4, 1): 7,
            (4, 2): 8,
            (5, 2): 9,
        }

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
    ) -> None:
        """Execute tempo-oriented decisions."""
        r = rng or random

        # 1. Leveling according to tempo schedule or excess econ above 50
        target_level = self.tempo_schedule.get((stage, round_in_stage))
        if target_level and player.level < target_level:
            # Spend gold to reach target level if affordable
            while player.level < target_level and player.gold >= set_data.exp_buy_cost:
                if not player.buy_exp():
                    break
        elif player.gold > 50 and player.level < set_data.max_level:
            # Dump surplus gold above 50 into EXP
            surplus = player.gold - 50
            buys = surplus // set_data.exp_buy_cost
            for _ in range(buys):
                if not player.buy_exp():
                    break

        # 2. Buy shop units that upgrade current units or match traits
        self._buy_shop_units(player, pool, set_data, stage)

        # 3. Roll Down Gold if in danger (HP <= 35) or surplus gold at Level 8+
        is_danger = player.health <= 35
        min_gold_threshold = 10 if is_danger else 50
        max_rerolls = 6 if is_danger else 2

        if (is_danger or player.level >= 8) and player.gold > min_gold_threshold:
            rerolls = 0
            while player.gold >= min_gold_threshold + set_data.reroll_cost and rerolls < max_rerolls:
                if not player.reroll_shop(pool):
                    break
                rerolls += 1
                self._buy_shop_units(player, pool, set_data, stage)

        # 4. Ensure Board is at Maximum Capacity
        self._fill_board_capacity(player, set_data)

        # 5. Item Equipping
        self._equip_items_intelligently(player, set_data)

        # 6. Bench cleanup if bench is overflowing
        if player.free_bench_slots == 0:
            self._sell_lowest_priority_bench_unit(player, pool, set_data)

    def _buy_shop_units(self, player: Player, pool: ChampionPool, set_data: SetData, stage: int) -> None:
        """Evaluate and buy beneficial champions from the current shop."""
        current_traits = set(player.get_active_traits().keys())
        owned_champ_ids = {u.champion_id for u in player.get_all_units()}

        for slot_idx in range(5):
            champ_id = player.shop.slots[slot_idx]
            if champ_id is None:
                continue

            cdef = set_data.champions.get(champ_id)
            if not cdef or player.gold < cdef.cost:
                continue

            shares_traits = any(t in current_traits for t in cdef.traits)
            is_upgrade_candidate = champ_id in owned_champ_ids

            # Buy if high synergy, upgrade, or surplus gold
            if is_upgrade_candidate or shares_traits or player.gold >= 30 or stage <= 2:
                player.buy_shop_slot(slot_idx, pool)

    def _fill_board_capacity(self, player: Player, set_data: SetData) -> None:
        """Move best units from bench to board if board is under capacity."""
        while player.board_unit_count < player.max_board_units:
            # Find strongest bench unit
            bench_candidates = [
                (idx, u) for idx, u in enumerate(player.bench) if u is not None
            ]
            if not bench_candidates:
                break

            # Sort by star level desc, cost desc
            bench_candidates.sort(key=lambda item: (item[1].star_level, item[1].cost), reverse=True)
            best_idx, best_unit = bench_candidates[0]

            # Find empty board hex
            placed = False
            for r in range(set_data.board_rows):
                for c in range(set_data.board_cols):
                    if (r, c) not in player.board:
                        player.move_unit(from_is_board=False, from_loc=best_idx, to_is_board=True, to_loc=(r, c))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                break

    def _equip_items_intelligently(self, player: Player, set_data: SetData) -> None:
        """Combine components and equip to appropriate role carries."""
        # First combine components on bench if possible
        if len(player.item_bench) >= 2:
            i = 0
            while i < len(player.item_bench) - 1:
                j = i + 1
                combined = False
                while j < len(player.item_bench):
                    if player.combine_items_on_bench(i, j, set_data):
                        combined = True
                        break
                    j += 1
                if not combined:
                    i += 1

        # Equip items onto board champions
        if not player.item_bench or not player.board:
            return

        # Prioritize 2-star / high-cost carries and tanks
        sorted_units = sorted(
            player.board.items(),
            key=lambda item: (item[1].star_level, item[1].cost),
            reverse=True,
        )

        for pos, unit in sorted_units:
            if len(unit.items) >= 3:
                continue
            if player.item_bench:
                player.equip_item(0, is_board=True, target_loc=pos)

    def _sell_lowest_priority_bench_unit(self, player: Player, pool: ChampionPool, set_data: SetData) -> None:
        """Sell the lowest value 1-star bench unit when space is constrained."""
        bench_units = [(idx, u) for idx, u in enumerate(player.bench) if u is not None and u.star_level == 1]
        if not bench_units:
            return

        # Sort by cost ascending
        bench_units.sort(key=lambda item: item[1].cost)
        lowest_idx, _ = bench_units[0]
        player.sell_unit(is_board=False, loc=lowest_idx, pool=pool)


class GreedyBankerBot:
    """Bot prioritizing maximum 50-gold interest at all costs."""

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
    ) -> None:
        """Execute greedy interest strategy."""
        # Always save until 50 gold
        if player.gold > 50:
            surplus = player.gold - 50
            buys = surplus // set_data.exp_buy_cost
            for _ in range(buys):
                player.buy_exp()

        # Buy any upgrade candidate that keeps gold >= 50
        for slot_idx in range(5):
            champ_id = player.shop.slots[slot_idx]
            if champ_id is None:
                continue
            cdef = set_data.champions.get(champ_id)
            if cdef and player.gold - cdef.cost >= (50 if stage >= 3 else 0):
                player.buy_shop_slot(slot_idx, pool)

        # Fill board
        while player.board_unit_count < player.max_board_units:
            bench_candidates = [(idx, u) for idx, u in enumerate(player.bench) if u is not None]
            if not bench_candidates:
                break
            best_idx, _ = bench_candidates[0]
            for r in range(set_data.board_rows):
                for c in range(set_data.board_cols):
                    if (r, c) not in player.board:
                        player.move_unit(from_is_board=False, from_loc=best_idx, to_is_board=True, to_loc=(r, c))
                        break


class RandomBot:
    """Random action bot used for baseline robustness testing."""

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
    ) -> None:
        """Execute random moves."""
        r = rng or random

        # Random shop buys
        for slot_idx in range(5):
            if r.random() < 0.4 and player.can_buy_champion(slot_idx):
                player.buy_shop_slot(slot_idx, pool)

        # Random level up
        if r.random() < 0.3 and player.gold >= set_data.exp_buy_cost:
            player.buy_exp()

        # Move to board
        while player.board_unit_count < player.max_board_units:
            bench_units = [(idx, u) for idx, u in enumerate(player.bench) if u is not None]
            if not bench_units:
                break
            idx, _ = r.choice(bench_units)
            empty_hexes = [
                (row, col)
                for row in range(set_data.board_rows)
                for col in range(set_data.board_cols)
                if (row, col) not in player.board
            ]
            if not empty_hexes:
                break
            target_hex = r.choice(empty_hexes)
            player.move_unit(from_is_board=False, from_loc=idx, to_is_board=True, to_loc=target_hex)
