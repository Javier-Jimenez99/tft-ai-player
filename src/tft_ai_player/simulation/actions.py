"""Action spaces, discrete encoding/decoding, and high-performance invalid action masking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.models import ChampionPool, Player


class ActionType(IntEnum):
    """Categorical discrete action identifiers (306 total discrete actions)."""

    PASS = 0
    BUY_SHOP_0 = 1
    BUY_SHOP_1 = 2
    BUY_SHOP_2 = 3
    BUY_SHOP_3 = 4
    BUY_SHOP_4 = 5
    REROLL_SHOP = 6
    BUY_EXP = 7
    TOGGLE_LOCK_SHOP = 8
    SELL_BENCH = 9             # 9..17 (9 bench slots)
    SELL_BOARD = 18            # 18..29 (12 board team slots)
    DEPLOY_BENCH_TO_BOARD = 30 # 30..38 (9 bench slots)
    RECALL_BOARD_TO_BENCH = 39 # 39..50 (12 board team slots)
    EQUIP_ITEM_BOARD = 51      # 51..170 (10 item slots * 12 board slots = 120)
    EQUIP_ITEM_BENCH = 171     # 171..260 (10 item slots * 9 bench slots = 90)
    COMBINE_ITEMS = 261        # 261..305 (45 component pairs)


TOTAL_DISCRETE_ACTIONS = 306

ACTION_GROUP_NAMES = [
    "Pass",
    "Buy",
    "Reroll",
    "EXP",
    "Lock",
    "Sell",
    "Deploy",
    "Recall",
    "Equip",
    "Combine",
]


def get_action_group(action_id: int) -> str:
    """Map discrete action ID (0..305) to high-level action group name."""
    if action_id == 0:
        return "Pass"
    if 1 <= action_id <= 5:
        return "Buy"
    if action_id == 6:
        return "Reroll"
    if action_id == 7:
        return "EXP"
    if action_id == 8:
        return "Lock"
    if 9 <= action_id <= 29:
        return "Sell"
    if 30 <= action_id <= 38:
        return "Deploy"
    if 39 <= action_id <= 50:
        return "Recall"
    if 51 <= action_id <= 260:
        return "Equip"
    if 261 <= action_id <= 305:
        return "Combine"
    return "Unknown"


def hex_to_index(r: int, c: int, cols: int = 7) -> int:
    """Convert (row, col) to flattened board index (0..27)."""
    return r * cols + c


def index_to_hex(idx: int, cols: int = 7) -> tuple[int, int]:
    """Convert flattened board index (0..27) to (row, col)."""
    return divmod(idx, cols)


def execute_action(
    player: Player,
    pool: ChampionPool,
    set_data: SetData,
    action_id: int,
    rng: random.Random | None = None,
) -> bool:
    """Execute a discrete action (0..305) on the player's state. Returns True if action succeeded."""
    if action_id == 0:
        # PASS
        return True

    # 1. Buy shop slots (1..5)
    if 1 <= action_id <= 5:
        slot_idx = action_id - 1
        return player.buy_shop_slot(slot_idx, pool)

    # 2. Reroll shop (6)
    if action_id == 6:
        return player.reroll_shop(pool, rng=rng)

    # 3. Buy EXP (7)
    if action_id == 7:
        return player.buy_exp()

    # 4. Toggle lock shop (8)
    if action_id == 8:
        player.toggle_shop_lock()
        return True

    # 5. Sell Bench (9..17)
    if 9 <= action_id <= 17:
        bench_slot = action_id - 9
        return player.sell_unit(is_board=False, loc=bench_slot, pool=pool)

    # 6. Sell Board (18..29)
    if 18 <= action_id <= 29:
        board_slot = action_id - 18
        return player.sell_board_slot(board_slot, pool=pool)

    # 7. Deploy Bench to Board (30..38)
    if 30 <= action_id <= 38:
        bench_slot = action_id - 30
        return player.deploy_bench_slot(bench_slot)

    # 8. Recall Board to Bench (39..50)
    if 39 <= action_id <= 50:
        board_slot = action_id - 39
        return player.recall_board_slot(board_slot)

    # 9. Equip Item to Board Unit (51..170)
    if 51 <= action_id <= 170:
        offset = action_id - 51
        item_idx = offset // 12
        board_slot = offset % 12
        return player.equip_item_to_board_slot(item_idx, board_slot)

    # 10. Equip Item to Bench Unit (171..260)
    if 171 <= action_id <= 260:
        offset = action_id - 171
        item_idx = offset // 9
        bench_slot = offset % 9
        return player.equip_item(item_idx, is_board=False, target_loc=bench_slot)

    # 11. Combine Items on Bench (261..305)
    if 261 <= action_id <= 305:
        offset = action_id - 261
        comb_count = 0
        for i in range(10):
            for j in range(i + 1, 10):
                if comb_count == offset:
                    return player.combine_items_on_bench(i, j, set_data)
                comb_count += 1

    return False


def get_action_mask(player: Player, set_data: SetData) -> np.ndarray:
    """Compute a fast boolean vector of valid discrete actions for the player (306 actions)."""
    mask = np.zeros(TOTAL_DISCRETE_ACTIONS, dtype=bool)

    if not player.alive:
        mask[0] = True
        return mask

    # 0: PASS is always valid
    mask[0] = True

    # 1..5: Shop buys
    for slot_idx in range(5):
        if player.can_buy_champion(slot_idx):
            mask[1 + slot_idx] = True

    # 6: Reroll shop
    if player.gold >= set_data.reroll_cost:
        mask[6] = True

    # 7: Buy EXP
    if player.gold >= set_data.exp_buy_cost and player.level < set_data.max_level:
        mask[7] = True

    # 8: Toggle lock (only valid if shop is NOT already locked and contains champions)
    if not player.shop.locked and any(c is not None for c in player.shop.slots):
        mask[8] = True

    # 9..17: Sell Bench (0..8)
    for bench_idx, unit in enumerate(player.bench):
        if unit is not None:
            mask[9 + bench_idx] = True

    # 18..29: Sell Board (0..11)
    board_units = player.get_board_units_list()
    for board_idx in range(len(board_units)):
        if board_idx < 12:
            mask[18 + board_idx] = True

    # 30..38: Deploy Bench to Board (0..8)
    if player.board_unit_count < player.max_board_units:
        for bench_idx, unit in enumerate(player.bench):
            if unit is not None:
                mask[30 + bench_idx] = True

    # 39..50: Recall Board to Bench (0..11)
    if player.free_bench_slots > 0:
        for board_idx in range(len(board_units)):
            if board_idx < 12:
                mask[39 + board_idx] = True

    # 51..170: Equip Item to Board Unit (10 item slots * 12 board slots = 120)
    for it_idx, it in enumerate(player.item_bench):
        if it is not None and it_idx < 10:
            for board_idx, u in enumerate(board_units):
                if board_idx < 12 and len(u.items) < 3:
                    mask[51 + (it_idx * 12) + board_idx] = True

    # 171..260: Equip Item to Bench Unit (10 item slots * 9 bench slots = 90)
    for it_idx, it in enumerate(player.item_bench):
        if it is not None and it_idx < 10:
            for bench_idx, u in enumerate(player.bench):
                if u is not None and bench_idx < 9 and len(u.items) < 3:
                    mask[171 + (it_idx * 9) + bench_idx] = True

    # 261..305: Combine Items on Bench (45 component pairs)
    comb_count = 0
    for i in range(10):
        for j in range(i + 1, 10):
            if i < len(player.item_bench) and j < len(player.item_bench):
                it_a = player.item_bench[i]
                it_b = player.item_bench[j]
                if it_a and it_b and it_a.is_component and it_b.is_component:
                    if set_data.get_recipe_result(it_a.item_id, it_b.item_id) is not None:
                        mask[261 + comb_count] = True
            comb_count += 1

    return mask
