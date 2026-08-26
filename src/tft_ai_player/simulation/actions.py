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
    """Categorical discrete action identifiers."""

    PASS = 0
    BUY_SHOP_0 = 1
    BUY_SHOP_1 = 2
    BUY_SHOP_2 = 3
    BUY_SHOP_3 = 4
    BUY_SHOP_4 = 5
    REROLL_SHOP = 6
    BUY_EXP = 7
    TOGGLE_LOCK_SHOP = 8
    SELL_BENCH = 9            # 9..17 (9 slots)
    SELL_BOARD = 18           # 18..45 (28 hexes)
    MOVE_BENCH_TO_BOARD = 46  # 46..297 (9 * 28 = 252)
    MOVE_BOARD_TO_BENCH = 298 # 298..549 (28 * 9 = 252)
    MOVE_BOARD_TO_BOARD = 550 # 550..1305 (28 * 27 = 756)
    EQUIP_ITEM_BOARD = 1306   # 1306..1585 (10 * 28 = 280)
    EQUIP_ITEM_BENCH = 1586   # 1586..1675 (10 * 9 = 90)
    COMBINE_ITEMS = 1676      # 1676..1720 (45 combinations)


TOTAL_DISCRETE_ACTIONS = 1721


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
    """Execute a discrete action on the player's state. Returns True if action succeeded."""
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

    # 6. Sell Board (18..45)
    if 18 <= action_id <= 45:
        hex_idx = action_id - 18
        pos = index_to_hex(hex_idx, set_data.board_cols)
        return player.sell_unit(is_board=True, loc=pos, pool=pool)

    # 7. Move Bench to Board (46..297)
    if 46 <= action_id <= 297:
        offset = action_id - 46
        bench_slot = offset // 28
        hex_idx = offset % 28
        pos = index_to_hex(hex_idx, set_data.board_cols)
        return player.move_unit(from_is_board=False, from_loc=bench_slot, to_is_board=True, to_loc=pos)

    # 8. Move Board to Bench (298..549)
    if 298 <= action_id <= 549:
        offset = action_id - 298
        hex_idx = offset // 9
        bench_slot = offset % 9
        pos = index_to_hex(hex_idx, set_data.board_cols)
        return player.move_unit(from_is_board=True, from_loc=pos, to_is_board=False, to_loc=bench_slot)

    # 9. Move Board to Board (550..1305)
    if 550 <= action_id <= 1305:
        offset = action_id - 550
        src_hex_idx = offset // 27
        dst_relative_idx = offset % 27
        dst_hex_idx = dst_relative_idx if dst_relative_idx < src_hex_idx else dst_relative_idx + 1
        src_pos = index_to_hex(src_hex_idx, set_data.board_cols)
        dst_pos = index_to_hex(dst_hex_idx, set_data.board_cols)
        return player.move_unit(from_is_board=True, from_loc=src_pos, to_is_board=True, to_loc=dst_pos)

    # 10. Equip Item to Board Unit (1306..1585)
    if 1306 <= action_id <= 1585:
        offset = action_id - 1306
        item_idx = offset // 28
        hex_idx = offset % 28
        pos = index_to_hex(hex_idx, set_data.board_cols)
        return player.equip_item(item_idx, is_board=True, target_loc=pos)

    # 11. Equip Item to Bench Unit (1586..1675)
    if 1586 <= action_id <= 1675:
        offset = action_id - 1586
        item_idx = offset // 9
        bench_slot = offset % 9
        return player.equip_item(item_idx, is_board=False, target_loc=bench_slot)

    # 12. Combine Items on Bench (1676..1720)
    if 1676 <= action_id <= 1720:
        offset = action_id - 1676
        # Reconstruct (idx1, idx2) with idx1 < idx2 among 0..9
        comb_count = 0
        for i in range(10):
            for j in range(i + 1, 10):
                if comb_count == offset:
                    return player.combine_items_on_bench(i, j, set_data)
                comb_count += 1

    return False


def get_action_mask(player: Player, set_data: SetData) -> np.ndarray:
    """Compute a fast boolean vector of valid discrete actions for the player."""
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

    # 8: Toggle lock
    mask[8] = True

    # 9..17: Sell Bench
    for bench_idx, unit in enumerate(player.bench):
        if unit is not None:
            mask[9 + bench_idx] = True

    # 18..45: Sell Board
    for hex_idx in range(28):
        pos = index_to_hex(hex_idx, set_data.board_cols)
        if pos in player.board:
            mask[18 + hex_idx] = True

    # 46..297: Move Bench to Board
    board_count = player.board_unit_count
    max_board = player.max_board_units
    can_place_new = board_count < max_board

    for bench_idx, unit in enumerate(player.bench):
        if unit is not None:
            for hex_idx in range(28):
                pos = index_to_hex(hex_idx, set_data.board_cols)
                dest_occupied = pos in player.board
                if dest_occupied or can_place_new:
                    action_id = 46 + (bench_idx * 28) + hex_idx
                    mask[action_id] = True

    # 298..549: Move Board to Bench
    for hex_idx in range(28):
        pos = index_to_hex(hex_idx, set_data.board_cols)
        if pos in player.board:
            for bench_idx in range(9):
                action_id = 298 + (hex_idx * 9) + bench_idx
                mask[action_id] = True

    # 550..1305: Move Board to Board
    for src_hex_idx in range(28):
        src_pos = index_to_hex(src_hex_idx, set_data.board_cols)
        if src_pos in player.board:
            for dst_relative_idx in range(27):
                dst_hex_idx = dst_relative_idx if dst_relative_idx < src_hex_idx else dst_relative_idx + 1
                action_id = 550 + (src_hex_idx * 27) + dst_relative_idx
                mask[action_id] = True

    # Items availability
    num_items = len(player.item_bench)

    # 1306..1585: Equip Item to Board
    for item_idx in range(min(num_items, 10)):
        item_inst = player.item_bench[item_idx]
        for hex_idx in range(28):
            pos = index_to_hex(hex_idx, set_data.board_cols)
            target_unit = player.board.get(pos)
            if target_unit is not None:
                # Can equip if < 3 items, or can combine component
                if len(target_unit.items) < 3:
                    action_id = 1306 + (item_idx * 28) + hex_idx
                    mask[action_id] = True
                elif item_inst.is_component:
                    # Check if target has an uncombined component
                    has_comp = any(set_data.is_component(it) for it in target_unit.items)
                    if has_comp:
                        action_id = 1306 + (item_idx * 28) + hex_idx
                        mask[action_id] = True

    # 1586..1675: Equip Item to Bench
    for item_idx in range(min(num_items, 10)):
        item_inst = player.item_bench[item_idx]
        for bench_slot in range(9):
            target_unit = player.bench[bench_slot]
            if target_unit is not None:
                if len(target_unit.items) < 3:
                    action_id = 1586 + (item_idx * 9) + bench_slot
                    mask[action_id] = True
                elif item_inst.is_component:
                    has_comp = any(set_data.is_component(it) for it in target_unit.items)
                    if has_comp:
                        action_id = 1586 + (item_idx * 9) + bench_slot
                        mask[action_id] = True

    # 1676..1720: Combine Items on Bench
    if num_items >= 2:
        comb_count = 0
        for i in range(10):
            for j in range(i + 1, 10):
                if i < num_items and j < num_items:
                    item1 = player.item_bench[i]
                    item2 = player.item_bench[j]
                    if item1.is_component and item2.is_component:
                        if set_data.get_recipe_result(item1.item_id, item2.item_id) is not None:
                            mask[1676 + comb_count] = True
                comb_count += 1

    return mask
