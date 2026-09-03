"""Factorized discrete action space and high-performance invalid action masking.

Defines the canonical 111-action factorized discrete action space:
  - 0: PASS_ROUND (finalizes planning turn, triggers combat resolution)
  - 1..5: BUY_SHOP_SLOT (0..4)
  - 6: REROLL_SHOP (2 gold)
  - 7: BUY_XP (4 gold for 4 XP)
  - 8..16: SELL_BENCH (bench slot 0..8)
  - 17..44: SELL_BOARD (board hex 0..27)
  - 45..72: DEPLOY_UNIT (moves bench unit to board hex 0..27)
  - 73..100: MOVE_BOARD (swaps/moves unit to board hex 0..27)
  - 101..110: EQUIP_ITEM (equips item from item slot 0..9 onto primary unit)
"""

from __future__ import annotations

import random
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tft_ai_player.simulation.config import SetData
    from tft_ai_player.simulation.models import ChampionPool, Player


class ActionType(IntEnum):
    """Categorical discrete action identifiers (|A| = 111)."""

    PASS_ROUND = 0
    BUY_SHOP_0 = 1
    BUY_SHOP_1 = 2
    BUY_SHOP_2 = 3
    BUY_SHOP_3 = 4
    BUY_SHOP_4 = 5
    REROLL_SHOP = 6
    BUY_XP = 7
    SELL_BENCH = 8      # 8..16 (9 bench slots: 8 + slot)
    SELL_BOARD = 17     # 17..44 (28 board hexes: 17 + hex_idx)
    DEPLOY_UNIT = 45    # 45..72 (28 board hexes: 45 + hex_idx)
    MOVE_BOARD = 73     # 73..100 (28 board hexes: 73 + hex_idx)
    EQUIP_ITEM = 101    # 101..110 (10 item bench slots: 101 + item_slot)


TOTAL_DISCRETE_ACTIONS = 111

ACTION_CATEGORY_NAMES = [
    "PASS_ROUND",
    "BUY_SHOP",
    "REROLL_SHOP",
    "BUY_XP",
    "SELL_BENCH",
    "SELL_BOARD",
    "DEPLOY_UNIT",
    "MOVE_BOARD",
    "EQUIP_ITEM",
]


def get_action_category(action_id: int) -> str:
    """Map discrete action ID (0..110) to high-level action category name."""
    if action_id == 0:
        return "PASS_ROUND"
    if 1 <= action_id <= 5:
        return "BUY_SHOP"
    if action_id == 6:
        return "REROLL_SHOP"
    if action_id == 7:
        return "BUY_XP"
    if 8 <= action_id <= 16:
        return "SELL_BENCH"
    if 17 <= action_id <= 44:
        return "SELL_BOARD"
    if 45 <= action_id <= 72:
        return "DEPLOY_UNIT"
    if 73 <= action_id <= 100:
        return "MOVE_BOARD"
    if 101 <= action_id <= 110:
        return "EQUIP_ITEM"
    return "UNKNOWN"


def hex_to_index(r: int, c: int, cols: int = 7) -> int:
    """Convert (row, col) to flattened board index (0..27)."""
    return r * cols + c


def index_to_hex(idx: int, cols: int = 7) -> tuple[int, int]:
    """Convert flattened board index (0..27) to (row, col)."""
    return divmod(idx, cols)


def get_action_mask(player: Player, set_data: SetData) -> np.ndarray:
    """Compute boolean validity mask m in {0, 1}^111 for the current player state.

    Actions violating game rules (e.g. rolling with <2g, purchasing empty shop slots,
    deploying beyond board capacity) are strictly masked out (False).
    """
    mask = np.zeros(TOTAL_DISCRETE_ACTIONS, dtype=bool)

    # 0. PASS_ROUND is unconditionally legal
    mask[0] = True

    # 1. BUY_SHOP_SLOT (1..5)
    for slot_idx in range(5):
        action_id = 1 + slot_idx
        if player.can_buy_champion(slot_idx):
            mask[action_id] = True

    # 2. REROLL_SHOP (6)
    if player.gold >= 2 or player.free_rerolls > 0:
        mask[6] = True

    # 3. BUY_XP (7)
    if player.gold >= 4 and player.level < set_data.max_level:
        mask[7] = True

    # 4. SELL_BENCH (8..16)
    for bench_slot in range(min(9, len(player.bench))):
        action_id = 8 + bench_slot
        if player.bench[bench_slot] is not None:
            mask[action_id] = True

    # 5. SELL_BOARD (17..44)
    for hex_idx in range(28):
        action_id = 17 + hex_idx
        r, c = index_to_hex(hex_idx, set_data.board_cols)
        if (r, c) in player.board:
            mask[action_id] = True

    # 6. DEPLOY_UNIT (45..72)
    # Moving a bench unit onto board hex (r, c)
    has_bench_unit = any(u is not None for u in player.bench)
    if has_bench_unit:
        for hex_idx in range(28):
            action_id = 45 + hex_idx
            r, c = index_to_hex(hex_idx, set_data.board_cols)
            # Allowed if hex is empty and board not full, or if hex is occupied (swap)
            if (r, c) in player.board:
                mask[action_id] = True
            elif player.board_unit_count < player.max_board_units:
                mask[action_id] = True

    # 7. MOVE_BOARD (73..100)
    # Swapping or moving a board unit to target hex (r, c)
    has_board_unit = len(player.board) > 0
    if has_board_unit:
        for hex_idx in range(28):
            action_id = 73 + hex_idx
            mask[action_id] = True

    # 8. EQUIP_ITEM (101..110)
    # Equipping item from item bench slot 0..9 onto a unit with available item capacity
    units_with_capacity = [u for u in player.board.values() if len(u.items) < 3]
    if units_with_capacity and len(player.item_bench) > 0:
        for item_idx in range(min(10, len(player.item_bench))):
            action_id = 101 + item_idx
            mask[action_id] = True

    return mask


def execute_action(
    player: Player,
    pool: ChampionPool,
    set_data: SetData,
    action_id: int,
    rng: random.Random | None = None,
) -> bool:
    """Execute discrete action (0..110) on the player state.

    Returns True if the action executed successfully.
    """
    if action_id == 0:
        # PASS_ROUND
        return True

    # 1. Buy shop slots (1..5)
    if 1 <= action_id <= 5:
        slot_idx = action_id - 1
        return player.buy_shop_slot(slot_idx, pool)

    # 2. Reroll shop (6)
    if action_id == 6:
        return player.reroll_shop(pool, rng=rng)

    # 3. Buy XP (7)
    if action_id == 7:
        return player.buy_exp()

    # 4. Sell Bench (8..16)
    if 8 <= action_id <= 16:
        bench_slot = action_id - 8
        if 0 <= bench_slot < len(player.bench) and player.bench[bench_slot] is not None:
            return player.sell_unit(is_board=False, loc=bench_slot, pool=pool)
        return False

    # 5. Sell Board (17..44)
    if 17 <= action_id <= 44:
        hex_idx = action_id - 17
        r, c = index_to_hex(hex_idx, set_data.board_cols)
        if (r, c) in player.board:
            return player.sell_unit(is_board=True, loc=(r, c), pool=pool)
        return False

    # 6. Deploy Unit from Bench to Board (45..72)
    if 45 <= action_id <= 72:
        hex_idx = action_id - 45
        target_pos = index_to_hex(hex_idx, set_data.board_cols)
        # Find first available bench unit
        bench_candidates = [(idx, u) for idx, u in enumerate(player.bench) if u is not None]
        if not bench_candidates:
            return False
        # Sort to prioritize strongest bench candidate
        bench_candidates.sort(key=lambda it: (it[1].star_level, it[1].cost), reverse=True)
        best_bench_idx, _ = bench_candidates[0]

        if target_pos in player.board:
            # Swap board unit with bench unit
            board_unit = player.board[target_pos]
            bench_unit = player.bench[best_bench_idx]
            player.board[target_pos] = bench_unit
            if bench_unit is not None:
                bench_unit.position = target_pos
            player.bench[best_bench_idx] = board_unit
            if board_unit is not None:
                board_unit.position = None
            player.layout_board_tactically()
            return True
        elif player.board_unit_count < player.max_board_units:
            return player.move_unit(from_is_board=False, from_loc=best_bench_idx, to_is_board=True, to_loc=target_pos)
        return False

    # 7. Move Board Unit (73..100)
    if 73 <= action_id <= 100:
        hex_idx = action_id - 73
        target_pos = index_to_hex(hex_idx, set_data.board_cols)
        if not player.board:
            return False
        # Move first board unit that is not already at target_pos
        board_candidates = [pos for pos in player.board if pos != target_pos]
        if not board_candidates:
            return True  # Already placed
        from_pos = board_candidates[0]
        return player.move_unit(from_is_board=True, from_loc=from_pos, to_is_board=True, to_loc=target_pos)

    # 8. Equip Item (101..110)
    if 101 <= action_id <= 110:
        item_slot = action_id - 101
        if not (0 <= item_slot < len(player.item_bench)):
            return False
        # Find highest priority board unit with < 3 items
        eligible_units = [
            (pos, u) for pos, u in player.board.items() if len(u.items) < 3
        ]
        if not eligible_units:
            return False
        # Prioritize carry/tank units (higher star, higher cost)
        eligible_units.sort(key=lambda it: (it[1].star_level, it[1].cost), reverse=True)
        target_pos, _ = eligible_units[0]
        return player.equip_item(item_slot, is_board=True, target_loc=target_pos)

    return False
