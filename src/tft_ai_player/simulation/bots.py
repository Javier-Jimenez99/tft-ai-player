"""Baseline and benchmark opponent policies for 8-player lobby simulation.

Includes the three deterministic benchmark bots specified in AlphaStar evaluation:
- Bot Alpha (BotAlphaFast8): Rushes Level 8 at Stage 4-2, plays standard front-to-back 4-cost comps.
- Bot Beta (BotBetaHyperroll): Spends all economy at Stage 3-1 to 3-star early 1-cost / 2-cost units.
- Bot Gamma (BotGammaGreedy): Greedy economy open-fort, holds 50g at all costs to push fast Level 9.
"""

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


class BotAlphaFast8:
    """Bot Alpha (Standard 4-Cost Fast-8 Baseline).

    Standard competitive TFT tempo bot following standard meta leveling curves
    (Level 8 at 4-2) and front-to-back synergy construction.
    """

    def __init__(self) -> None:
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
        r = rng or random

        # 1. Leveling according to tempo schedule
        target_level = self.tempo_schedule.get((stage, round_in_stage))
        if target_level and player.level < target_level:
            while player.level < target_level and player.gold >= set_data.exp_buy_cost:
                if not player.buy_exp():
                    break
        elif player.gold > 50 and player.level < set_data.max_level:
            surplus = player.gold - 50
            buys = surplus // set_data.exp_buy_cost
            for _ in range(buys):
                if not player.buy_exp():
                    break

        # 2. Buy shop units
        self._buy_shop_units(player, pool, set_data, stage)

        # 3. Roll down if HP is low or reached Stage 4-2 Level 8
        if (player.health <= 35 or (stage >= 4 and player.level >= 8)) and player.gold > 20:
            for _ in range(4):
                if player.gold < 22:
                    break
                if not player.reroll_shop(pool, rng=r):
                    break
                self._buy_shop_units(player, pool, set_data, stage)

        # 4. Fill board capacity and equip items
        self._fill_board_capacity(player, set_data)
        self._equip_items(player, set_data)

    def _buy_shop_units(self, player: Player, pool: ChampionPool, set_data: SetData, stage: int) -> None:
        active_champs = {u.champion_id for u in player.get_all_units()}
        for slot_idx, champ_id in enumerate(player.shop.slots):
            if champ_id is None:
                continue
            cdef = set_data.champions.get(champ_id)
            if not cdef:
                continue
            # Buy if it upgrades existing units or matches higher cost threshold
            should_buy = champ_id in active_champs or (stage >= 3 and cdef.cost >= 3)
            if should_buy and player.can_buy_champion(slot_idx):
                player.buy_shop_slot(slot_idx, pool)

    def _fill_board_capacity(self, player: Player, set_data: SetData) -> None:
        while player.board_unit_count < player.max_board_units:
            bench_units = [(idx, u) for idx, u in enumerate(player.bench) if u is not None]
            if not bench_units:
                break
            bench_units.sort(key=lambda it: (it[1].star_level, it[1].cost), reverse=True)
            best_idx, _ = bench_units[0]
            # Find open hex
            for r in range(set_data.board_rows):
                for c in range(set_data.board_cols):
                    if (r, c) not in player.board:
                        player.move_unit(from_is_board=False, from_loc=best_idx, to_is_board=True, to_loc=(r, c))
                        break
                if (r, c) in player.board and player.board[(r, c)] == player.bench[best_idx]:
                    break
            player.layout_board_tactically()

    def _equip_items(self, player: Player, set_data: SetData) -> None:
        if not player.item_bench or not player.board:
            return
        carries = sorted(
            player.board.values(),
            key=lambda u: (u.star_level, u.cost),
            reverse=True,
        )
        for unit in carries:
            if len(unit.items) >= 3:
                continue
            while player.item_bench and len(unit.items) < 3:
                item_idx = len(player.item_bench) - 1
                if unit.position:
                    player.equip_item(item_idx, is_board=True, target_loc=unit.position)
                else:
                    break


class BotBetaHyperroll:
    """Bot Beta (Hyperroll 1-Cost Baseline).

    Spends all economy at Stage 3-1 to 3-star early game 1-cost and 2-cost units,
    playing high-tempo re-roll strategies.
    """

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
    ) -> None:
        r = rng or random

        # Leveling: Slow leveling, prefers holding lower levels for shop odds
        if stage >= 4 and player.gold > 40 and player.level < 7:
            player.buy_exp()

        # Buy 1-cost and 2-cost units
        for slot_idx, champ_id in enumerate(player.shop.slots):
            if champ_id is None:
                continue
            cdef = set_data.champions.get(champ_id)
            if cdef and cdef.cost <= 2 and player.can_buy_champion(slot_idx):
                player.buy_shop_slot(slot_idx, pool)

        # Stage 3-1 Hyperroll Spike: Dump gold down to 10g
        if stage == 3 and round_in_stage == 1:
            while player.gold >= 10:
                if not player.reroll_shop(pool, rng=r):
                    break
                for slot_idx, champ_id in enumerate(player.shop.slots):
                    if champ_id is None:
                        continue
                    cdef = set_data.champions.get(champ_id)
                    if cdef and cdef.cost <= 2 and player.can_buy_champion(slot_idx):
                        player.buy_shop_slot(slot_idx, pool)

        # Regular roll when above 50g
        if player.gold > 50:
            while player.gold > 50:
                if not player.reroll_shop(pool, rng=r):
                    break
                for slot_idx, champ_id in enumerate(player.shop.slots):
                    if champ_id is None:
                        continue
                    cdef = set_data.champions.get(champ_id)
                    if cdef and cdef.cost <= 2 and player.can_buy_champion(slot_idx):
                        player.buy_shop_slot(slot_idx, pool)

        self._fill_board_capacity(player, set_data)
        self._equip_items(player, set_data)

    def _fill_board_capacity(self, player: Player, set_data: SetData) -> None:
        while player.board_unit_count < player.max_board_units:
            bench_units = [(idx, u) for idx, u in enumerate(player.bench) if u is not None]
            if not bench_units:
                break
            bench_units.sort(key=lambda it: (it[1].star_level, it[1].cost), reverse=True)
            best_idx, _ = bench_units[0]
            for r in range(set_data.board_rows):
                for c in range(set_data.board_cols):
                    if (r, c) not in player.board:
                        player.move_unit(from_is_board=False, from_loc=best_idx, to_is_board=True, to_loc=(r, c))
                        break
            player.layout_board_tactically()

    def _equip_items(self, player: Player, set_data: SetData) -> None:
        if not player.item_bench or not player.board:
            return
        units = sorted(player.board.values(), key=lambda u: (u.star_level, -u.cost), reverse=True)
        for unit in units:
            while player.item_bench and len(unit.items) < 3:
                item_idx = len(player.item_bench) - 1
                if unit.position:
                    player.equip_item(item_idx, is_board=True, target_loc=unit.position)
                else:
                    break


class BotGammaGreedy:
    """Bot Gamma (Greedy Economy Open-Fort Baseline).

    Holds 50g at all costs to push fast level 9 and field high-cost legendary units.
    """

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
    ) -> None:
        # Never roll below 50g
        # Dump all gold above 50 into EXP
        if player.gold > 50 and player.level < set_data.max_level:
            surplus = player.gold - 50
            buys = surplus // set_data.exp_buy_cost
            for _ in range(buys):
                if not player.buy_exp():
                    break

        # Only buy units that cost >= 4 or if gold stays >= 50
        for slot_idx, champ_id in enumerate(player.shop.slots):
            if champ_id is None:
                continue
            cdef = set_data.champions.get(champ_id)
            if not cdef:
                continue
            if cdef.cost >= 4 or player.gold - cdef.cost >= 50:
                if player.can_buy_champion(slot_idx):
                    player.buy_shop_slot(slot_idx, pool)

        self._fill_board_capacity(player, set_data)
        self._equip_items(player, set_data)

    def _fill_board_capacity(self, player: Player, set_data: SetData) -> None:
        while player.board_unit_count < player.max_board_units:
            bench_units = [(idx, u) for idx, u in enumerate(player.bench) if u is not None]
            if not bench_units:
                break
            bench_units.sort(key=lambda it: (it[1].star_level, it[1].cost), reverse=True)
            best_idx, _ = bench_units[0]
            for r in range(set_data.board_rows):
                for c in range(set_data.board_cols):
                    if (r, c) not in player.board:
                        player.move_unit(from_is_board=False, from_loc=best_idx, to_is_board=True, to_loc=(r, c))
                        break
            player.layout_board_tactically()

    def _equip_items(self, player: Player, set_data: SetData) -> None:
        if not player.item_bench or not player.board:
            return
        units = sorted(player.board.values(), key=lambda u: (u.cost, u.star_level), reverse=True)
        for unit in units:
            while player.item_bench and len(unit.items) < 3:
                item_idx = len(player.item_bench) - 1
                if unit.position:
                    player.equip_item(item_idx, is_board=True, target_loc=unit.position)
                else:
                    break


class RandomBot:
    """Random baseline bot executing randomized actions."""

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
    ) -> None:
        r = rng or random
        from tft_ai_player.simulation.actions import execute_action, get_action_mask

        for _ in range(5):
            mask = get_action_mask(player, set_data)
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                break
            chosen = int(r.choice(valid_actions))
            if chosen == 0:
                break
            execute_action(player, pool, set_data, chosen, rng=r)


# Aliases for backwards compatibility and clean naming
StandardTempoBot = BotAlphaFast8
GreedyBankerBot = BotGammaGreedy
HyperrollBot = BotBetaHyperroll
