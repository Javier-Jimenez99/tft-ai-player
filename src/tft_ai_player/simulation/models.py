"""Core entities and state containers for TFT simulation."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tft_ai_player.simulation.config import (
    TEAM_SIZE_EXPANDING_ITEMS,
    ChampionDef,
    ItemDef,
    SetData,
    UnitRole,
)


@dataclass(slots=True)
class ItemInstance:
    """An item held in an item bench or equipped on a champion."""

    item_id: str
    name: str
    is_component: bool = False

    def clone(self) -> ItemInstance:
        """Create an isolated clone of this item instance."""
        return ItemInstance(item_id=self.item_id, name=self.name, is_component=self.is_component)


@dataclass(slots=True)
class ChampionInstance:
    """An instance of a champion on a player's board or bench."""

    champion_id: str
    cost: int
    star_level: int = 1
    items: list[str] = field(default_factory=list)
    position: tuple[int, int] | None = None  # (row 0..3, col 0..6) if on board

    def clone(self) -> ChampionInstance:
        """Create an isolated clone of this champion instance."""
        return ChampionInstance(
            champion_id=self.champion_id,
            cost=self.cost,
            star_level=self.star_level,
            items=list(self.items),
            position=self.position,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching dataset / feature extractor format."""
        result: dict[str, Any] = {
            "unit": self.champion_id,
            "tier": self.star_level,
            "items": list(self.items),
        }
        if self.position is not None:
            result["row"] = self.position[0]
            result["col"] = self.position[1]
        return result


class ChampionPool:
    """Shared champion pool across all 8 players in a TFT lobby."""

    def __init__(self, set_data: SetData) -> None:
        self.set_data = set_data
        self.counts: dict[str, int] = {}
        self.champions_by_cost: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}

        for champ_id, cdef in set_data.champions.items():
            pool_size = set_data.pool_sizes.get(cdef.cost, 10)
            self.counts[champ_id] = pool_size
            if cdef.cost in self.champions_by_cost:
                self.champions_by_cost[cdef.cost].append(champ_id)

    def draw_champion(
        self,
        level: int,
        rng: random.Random | None = None,
        cost_tier_bonus: int = 0,
        candidates_pool: list[str] | None = None,
    ) -> str | None:
        """Sample a champion according to level shop odds and current pool availability."""
        r = rng or random

        if candidates_pool is not None:
            available = [c for c in candidates_pool if self.counts.get(c, 0) > 0]
            if not available:
                available = [c for c, count in self.counts.items() if count > 0]
                if not available:
                    return None
            weights = [self.counts[c] for c in available]
            chosen_champ = r.choices(available, weights=weights, k=1)[0]
            self.counts[chosen_champ] -= 1
            return chosen_champ

        effective_level = min(self.set_data.max_level, level + cost_tier_bonus)
        odds = self.set_data.shop_odds.get(effective_level, self.set_data.shop_odds.get(1, (1.0, 0, 0, 0, 0)))
        # Odds are for 1-cost, 2-cost, 3-cost, 4-cost, 5-cost
        cost_tiers = [1, 2, 3, 4, 5]
        chosen_cost = r.choices(cost_tiers, weights=odds, k=1)[0]

        # Available champions in this cost tier
        candidates = [c for c in self.champions_by_cost.get(chosen_cost, []) if self.counts.get(c, 0) > 0]
        if not candidates:
            # Fallback: try adjacent cost tiers if target tier is empty
            candidates = [c for c, count in self.counts.items() if count > 0]
            if not candidates:
                return None

        # Sample uniformly from available copies
        weights = [self.counts[c] for c in candidates]
        chosen_champ = r.choices(candidates, weights=weights, k=1)[0]
        self.counts[chosen_champ] -= 1
        return chosen_champ

    def return_champion(self, champion_id: str, star_level: int = 1) -> None:
        """Return copies of a champion back to the pool."""
        copies = 3 ** (star_level - 1)
        if champion_id in self.counts:
            self.counts[champion_id] += copies
        else:
            self.counts[champion_id] = copies

    def return_player_units(self, player: Player) -> None:
        """Return all units from a player's board, bench, and shop to the pool upon elimination."""
        for unit in player.board.values():
            self.return_champion(unit.champion_id, unit.star_level)
        for unit in player.bench:
            if unit is not None:
                self.return_champion(unit.champion_id, unit.star_level)
        for slot in player.shop.slots:
            if slot is not None:
                self.return_champion(slot, star_level=1)


    def clone(self) -> ChampionPool:
        """Create an isolated clone of the champion pool state."""
        cp = ChampionPool.__new__(ChampionPool)
        cp.set_data = self.set_data
        cp.counts = dict(self.counts)
        cp.champions_by_cost = self.champions_by_cost
        return cp


class Shop:
    """Player shop offering 5 champion choices each round."""

    def __init__(self) -> None:
        self.slots: list[str | None] = [None, None, None, None, None]
        self.locked: bool = False

    def clone(self) -> Shop:
        """Create an isolated clone of the shop state."""
        s = Shop()
        s.slots = list(self.slots)
        s.locked = self.locked
        return s

    def refresh(
        self,
        level: int,
        pool: ChampionPool,
        set_data: SetData,
        rng: random.Random | None = None,
        ignited_slots: Sequence[int] = (),
        candidates_pool: list[str] | None = None,
    ) -> None:
        """Return unpurchased cards and draw 5 new cards from the pool."""
        if self.locked:
            return

        # Return remaining cards to pool
        for slot in self.slots:
            if slot is not None:
                pool.return_champion(slot, star_level=1)

        # Draw 5 new cards
        new_slots: list[str | None] = []
        for slot_idx in range(5):
            cost_bonus = 1 if slot_idx in ignited_slots else 0
            new_slots.append(
                pool.draw_champion(
                    level=level,
                    rng=rng,
                    cost_tier_bonus=cost_bonus,
                    candidates_pool=candidates_pool,
                )
            )
        self.slots = new_slots

    def take(self, slot_idx: int) -> str | None:
        """Take card at slot_idx."""
        if 0 <= slot_idx < len(self.slots):
            card = self.slots[slot_idx]
            self.slots[slot_idx] = None
            return card
        return None


class Player:
    """Represents a single player in an 8-player TFT match."""

    def __init__(
        self,
        player_id: int,
        set_data: SetData,
        name: str | None = None,
    ) -> None:
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.set_data = set_data

        self.health: int = set_data.starting_hp
        self.gold: int = set_data.starting_gold
        self.level: int = 1
        self.exp: int = 0
        self.streak: int = 0  # +k for wins, -k for losses
        self.alive: bool = True
        self.placement: int | None = None

        self.board: dict[tuple[int, int], ChampionInstance] = {}
        self.bench: list[ChampionInstance | None] = [None] * set_data.max_bench_size
        self.item_bench: list[ItemInstance] = []
        self.shop: Shop = Shop()
        self.last_opponents: list[int] = []

        # Augments & Consumables
        self.augments: list[str] = []
        self.duplicators: int = 0
        self.reforgers: int = 0
        self.removers: int = 0
        self.extra_team_size: int = 0
        self.max_interest_cap: int = set_data.max_interest
        self.free_rerolls: int = 0

        # Economy & Loot Trait Tracking
        self.coven_essence: int = 0
        self.fae_pixies: int = 0
        self.draven_bounty_progress: int = 0
        self.rengar_takedowns: int = 0
        self.ignited_shop_slots: list[int] = []
        self.combats_since_overrun: int = 0

    def clone(self) -> Player:
        """Create a high-performance isolated clone of this player state."""
        p = Player(player_id=self.player_id, set_data=self.set_data, name=self.name)
        p.health = self.health
        p.gold = self.gold
        p.level = self.level
        p.exp = self.exp
        p.streak = self.streak
        p.alive = self.alive
        p.placement = self.placement
        p.board = {pos: unit.clone() for pos, unit in self.board.items()}
        p.bench = [u.clone() if u is not None else None for u in self.bench]
        p.item_bench = [it.clone() for it in self.item_bench]
        p.shop = self.shop.clone()
        p.last_opponents = list(self.last_opponents)
        p.augments = list(self.augments)
        p.duplicators = self.duplicators
        p.reforgers = self.reforgers
        p.removers = self.removers
        p.extra_team_size = self.extra_team_size
        p.max_interest_cap = self.max_interest_cap
        p.free_rerolls = self.free_rerolls
        p.coven_essence = self.coven_essence
        p.fae_pixies = self.fae_pixies
        p.draven_bounty_progress = self.draven_bounty_progress
        p.rengar_takedowns = self.rengar_takedowns
        p.ignited_shop_slots = list(self.ignited_shop_slots)
        p.combats_since_overrun = self.combats_since_overrun
        return p

    # -------------------------------------------------------------------------
    # Board & Unit Queries
    # -------------------------------------------------------------------------

    @property
    def board_unit_count(self) -> int:
        """Number of champions currently fielded on the board."""
        # Account for multi-slot champions (Elder Dragon takes 2 slots)
        count = 0
        for unit in self.board.values():
            if "ElderDragon" in unit.champion_id:
                count += 2
            else:
                count += 1
        return count

    @property
    def max_board_units(self) -> int:
        """Maximum number of champions allowed on the board (level + items + augments).

        Includes all team size expanding items (Tactician's Crown, Tactician's Shield,
        Tactician's Cape, etc.) equipped on board units, equipped on bench units,
        or held on the item bench, plus augment bonuses.
        """
        extra_slots = self.extra_team_size
        # Fielded board units
        for unit in self.board.values():
            for item_id in unit.items:
                if item_id in TEAM_SIZE_EXPANDING_ITEMS:
                    extra_slots += 1

        # Bench units holding items
        for unit in self.bench:
            if unit is not None:
                for item_id in unit.items:
                    if item_id in TEAM_SIZE_EXPANDING_ITEMS:
                        extra_slots += 1

        # Item bench inventory
        for item in self.item_bench:
            if item.item_id in TEAM_SIZE_EXPANDING_ITEMS:
                extra_slots += 1

        return self.level + extra_slots

    @property
    def free_bench_slots(self) -> int:
        """Number of unoccupied slots on the champion bench."""
        return sum(1 for slot in self.bench if slot is None)

    @property
    def free_item_slots(self) -> int:
        """Number of unoccupied slots on the item bench."""
        return max(0, self.set_data.max_item_bench - len(self.item_bench))

    def get_all_units(self) -> list[ChampionInstance]:
        """Return all units currently owned (board + bench)."""
        units = list(self.board.values())
        for b in self.bench:
            if b is not None:
                units.append(b)
        return units

    def get_board_value(self) -> float:
        """Calculate total gold value invested on the board."""
        total = 0.0
        for unit in self.board.values():
            cost = self.set_data.champions.get(unit.champion_id, ChampionDef(unit.champion_id, "", unit.cost)).cost
            total += cost * (3 ** (unit.star_level - 1))
        return total

    def get_active_traits(self) -> dict[str, int]:
        """Calculate active trait counts and tiers based on fielded unique champions."""
        trait_counts: Counter[str] = Counter()
        seen_champions: set[str] = set()

        for unit in self.board.values():
            if unit.champion_id in seen_champions:
                continue
            seen_champions.add(unit.champion_id)
            cdef = self.set_data.champions.get(unit.champion_id)
            if cdef:
                for trait in cdef.traits:
                    trait_counts[trait] += 1

        active_tiers: dict[str, int] = {}
        for trait, count in trait_counts.items():
            tdef = self.set_data.traits.get(trait)
            if tdef and tdef.thresholds:
                active_tier = 0
                for tier_idx, thresh in enumerate(tdef.thresholds, start=1):
                    if count >= thresh:
                        active_tier = tier_idx
                active_tiers[trait] = active_tier
            else:
                active_tiers[trait] = count
        return active_tiers

    # -------------------------------------------------------------------------
    # Economy & Income Calculations
    # -------------------------------------------------------------------------

    def calculate_interest_gold(self) -> int:
        """Calculate interest income for the current round based on banked gold."""
        interest = int(self.gold * self.set_data.interest_rate)
        return min(interest, self.max_interest_cap)

    def calculate_streak_gold(self) -> int:
        """Calculate streak bonus income."""
        abs_streak = abs(self.streak)
        for min_streak, bonus in self.set_data.streak_gold_thresholds:
            if abs_streak >= min_streak:
                return bonus
        return 0

    def add_gold(self, amount: int) -> None:
        """Add gold to player balance."""
        self.gold = max(0, self.gold + amount)

    # -------------------------------------------------------------------------
    # Cascading 3-in-1 Star-Up Combination Logic
    # -------------------------------------------------------------------------

    def _check_and_apply_star_ups(self, champion_id: str, pool: ChampionPool) -> bool:
        """Combine 3 copies of 1-star into 2-star, and 3 copies of 2-star into 3-star."""
        did_star_up = False

        for target_star in (1, 2):
            # Find all matching units of target_star level
            matching_board = [(pos, u) for pos, u in self.board.items() if u.champion_id == champion_id and u.star_level == target_star]
            matching_bench = [(idx, u) for idx, u in enumerate(self.bench) if u is not None and u.champion_id == champion_id and u.star_level == target_star]

            total_matching = len(matching_board) + len(matching_bench)
            if total_matching >= 3:
                # Merge 3 units into 1 upgraded unit
                # Select the 3 units to merge (prefer board units for keeping location)
                units_to_merge: list[tuple[bool, Any, ChampionInstance]] = []
                for pos, u in matching_board:
                    units_to_merge.append((True, pos, u))
                for idx, u in matching_bench:
                    units_to_merge.append((False, idx, u))

                merge_trio = units_to_merge[:3]
                primary_is_board, primary_loc, primary_unit = merge_trio[0]

                # Aggregate items from the 3 units
                collected_items: list[str] = []
                for _, _, u in merge_trio:
                    collected_items.extend(u.items)

                # Unit can carry at most 3 items; excess items returned to item bench
                upgraded_items = collected_items[:3]
                overflow_items = collected_items[3:]
                for item_id in overflow_items:
                    self.add_item(item_id)

                # Remove the other 2 units
                for is_board, loc, _ in merge_trio[1:]:
                    if is_board:
                        del self.board[loc]
                    else:
                        self.bench[loc] = None

                # Upgrade primary unit in-place
                primary_unit.star_level = target_star + 1
                primary_unit.items = upgraded_items

                did_star_up = True
                # Recursive check for subsequent star ups (e.g. 1-star -> 2-star -> 3-star)
                self._check_and_apply_star_ups(champion_id, pool)
                break

        return did_star_up

    # -------------------------------------------------------------------------
    # Champion Actions
    # -------------------------------------------------------------------------

    def add_champion_to_bench(
        self,
        champ: ChampionInstance,
        pool: ChampionPool,
        allow_board_overflow: bool = False,
    ) -> bool:
        """Add a champion to the bench (or merge if completing star-up) and check for star-ups."""
        # 1. Find first empty bench slot
        for idx, slot in enumerate(self.bench):
            if slot is None:
                champ.position = None
                self.bench[idx] = champ
                self._check_and_apply_star_ups(champ.champion_id, pool)
                return True

        # 2. If bench is full, check if adding this copy merges with 2 existing copies
        matching_board = [(pos, u) for pos, u in self.board.items() if u.champion_id == champ.champion_id and u.star_level == champ.star_level]
        matching_bench = [(idx, u) for idx, u in enumerate(self.bench) if u is not None and u.champion_id == champ.champion_id and u.star_level == champ.star_level]
        total_matching = len(matching_board) + len(matching_bench)

        if total_matching >= 2:
            # We can merge into an upgrade without allocating a new slot!
            if matching_board:
                primary_is_board, primary_loc, primary_unit = True, matching_board[0][0], matching_board[0][1]
                second_loc, second_is_board, second_unit = (matching_board[1][0], True, matching_board[1][1]) if len(matching_board) >= 2 else (matching_bench[0][0], False, matching_bench[0][1])
            else:
                primary_is_board, primary_loc, primary_unit = False, matching_bench[0][0], matching_bench[0][1]
                second_loc, second_is_board, second_unit = matching_bench[1][0], False, matching_bench[1][1]

            # Aggregate items from primary, second, and incoming champ
            collected_items = list(primary_unit.items) + list(second_unit.items) + list(champ.items)
            upgraded_items = collected_items[:3]
            overflow_items = collected_items[3:]
            for item_id in overflow_items:
                self.add_item(item_id)

            # Remove the second unit
            if second_is_board:
                del self.board[second_loc]
            else:
                self.bench[second_loc] = None

            # Upgrade primary unit
            primary_unit.star_level = champ.star_level + 1
            primary_unit.items = upgraded_items

            # Check if this new star level cascades into another upgrade
            self._check_and_apply_star_ups(champ.champion_id, pool)
            return True

        # 3. If allow_board_overflow is True (used by carousel draft) and board has open capacity
        if allow_board_overflow and self.board_unit_count < self.max_board_units:
            for r in range(self.set_data.board_rows):
                for c in range(self.set_data.board_cols):
                    if (r, c) not in self.board:
                        champ.position = (r, c)
                        self.board[(r, c)] = champ
                        self._check_and_apply_star_ups(champ.champion_id, pool)
                        return True

        return False

    def auto_fill_board_from_bench(self, pool: ChampionPool) -> int:
        """Automatically deploy strongest units from bench to board if board_unit_count < max_board_units.
        
        Matches official TFT behavior: when a round starts or player has empty board slots,
        units are automatically placed from the bench onto vacant board hexes up to team size.
        Returns the number of units deployed.
        """
        units_deployed = 0
        while self.board_unit_count < self.max_board_units:
            bench_candidates = [
                (idx, u) for idx, u in enumerate(self.bench) if u is not None
            ]
            if not bench_candidates:
                break

            # Prioritize strongest units: star level descending, cost descending
            bench_candidates.sort(key=lambda item: (item[1].star_level, item[1].cost), reverse=True)
            best_idx, _ = bench_candidates[0]

            # Find first available board hex (front row to back row)
            placed = False
            for r in range(self.set_data.board_rows):
                for c in range(self.set_data.board_cols):
                    if (r, c) not in self.board:
                        if self.move_unit(from_is_board=False, from_loc=best_idx, to_is_board=True, to_loc=(r, c)):
                            units_deployed += 1
                            placed = True
                            break
                if placed:
                    break

            if not placed:
                break

        self.layout_board_tactically()
        return units_deployed

    def get_board_units_list(self) -> list[ChampionInstance]:
        """Return list of active board units in stable sorted order (Star level desc, Cost desc, ID asc)."""
        units = list(self.board.values())
        units.sort(key=lambda u: (-u.star_level, -u.cost, str(u.champion_id)))
        return units

    def layout_board_tactically(self) -> None:
        """Deterministic tactical board positioning algorithm.

        Maps active board units to optimal (row, col) hexes based on role, durability, and items:
        - Row 0 (Frontline center/flanks): Main Tanks (UnitRole.TANK) and high-HP bruisers.
        - Row 1 (Second line): Melee Bruisers / Assassins (UnitRole.MELEE_DPS).
        - Row 2 (Third line): Utility / Supports / Short-range casters.
        - Row 3 (Backline corners/protected center): Primary Hyper-Carries (UnitRole.AD_CARRY, AP_CARRY).
        """
        units = list(self.board.values())
        if not units:
            self.board.clear()
            return

        # Hex priority preferences by row (center outward for tanks, protected corners for carries)
        row_hex_priorities: dict[int, list[tuple[int, int]]] = {
            0: [(0, 3), (0, 2), (0, 4), (0, 1), (0, 5), (0, 0), (0, 6)],  # Frontline center outward
            1: [(1, 3), (1, 2), (1, 4), (1, 1), (1, 5), (1, 0), (1, 6)],  # Second line center outward
            2: [(2, 3), (2, 2), (2, 4), (2, 1), (2, 5), (2, 0), (2, 6)],  # Third line
            3: [(3, 1), (3, 5), (3, 2), (3, 4), (3, 0), (3, 6), (3, 3)],  # Backline corners protected
        }

        # Categorize units into target row preference
        def get_preferred_row(u: ChampionInstance) -> int:
            cdef = self.set_data.champions.get(u.champion_id)
            role = cdef.role if cdef else UnitRole.UTILITY
            tank_items = sum(1 for it in u.items if any(k in it for k in ("Warmog", "Bramble", "Dragon", "Gargoyle", "Sunfire", "Steadfast", "Stoneplate", "Vow")))
            if tank_items >= 2 or role == UnitRole.TANK:
                return 0
            if role in (UnitRole.AD_CARRY, UnitRole.AP_CARRY):
                return 3
            return 1  # Utility / Front-mid default

        units_by_row: dict[int, list[ChampionInstance]] = {0: [], 1: [], 2: [], 3: []}
        for u in units:
            target_r = get_preferred_row(u)
            units_by_row[target_r].append(u)

        for r in units_by_row:
            units_by_row[r].sort(key=lambda u: (-u.star_level, -len(u.items), -u.cost))

        new_board: dict[tuple[int, int], ChampionInstance] = {}
        occupied_hexes: set[tuple[int, int]] = set()

        for target_r in [0, 3, 1, 2]:
            row_units = units_by_row[target_r]
            for u in row_units:
                placed = False
                for hex_pos in row_hex_priorities[target_r]:
                    if hex_pos not in occupied_hexes:
                        new_board[hex_pos] = u
                        occupied_hexes.add(hex_pos)
                        u.position = hex_pos
                        placed = True
                        break
                if not placed:
                    fallback_rows = [0, 1, 2, 3] if target_r < 2 else [3, 2, 1, 0]
                    for alt_r in fallback_rows:
                        for hex_pos in row_hex_priorities[alt_r]:
                            if hex_pos not in occupied_hexes:
                                new_board[hex_pos] = u
                                occupied_hexes.add(hex_pos)
                                u.position = hex_pos
                                placed = True
                                break
                        if placed:
                            break

        self.board = new_board

    def deploy_bench_slot(self, bench_slot: int) -> bool:
        """Deploy bench unit at bench_slot (0..8) onto active board with auto-layout."""
        if self.board_unit_count >= self.max_board_units:
            return False
        if not (0 <= bench_slot < len(self.bench)):
            return False
        unit = self.bench[bench_slot]
        if unit is None:
            return False

        self.bench[bench_slot] = None
        for r in range(self.set_data.board_rows):
            for c in range(self.set_data.board_cols):
                if (r, c) not in self.board:
                    unit.position = (r, c)
                    self.board[(r, c)] = unit
                    self.layout_board_tactically()
                    return True
        return False

    def recall_board_slot(self, board_slot: int) -> bool:
        """Recall board unit at board_slot (0..11) back to first free bench slot."""
        if self.free_bench_slots <= 0:
            return False
        units = self.get_board_units_list()
        if not (0 <= board_slot < len(units)):
            return False
        unit = units[board_slot]
        pos = unit.position
        if pos and pos in self.board:
            del self.board[pos]
        empty_bench_idx = next(i for i, s in enumerate(self.bench) if s is None)
        self.bench[empty_bench_idx] = unit
        unit.position = None
        self.layout_board_tactically()
        return True

    def sell_board_slot(self, board_slot: int, pool: ChampionPool) -> bool:
        """Sell board unit at board_slot (0..11)."""
        units = self.get_board_units_list()
        if not (0 <= board_slot < len(units)):
            return False
        unit = units[board_slot]
        pos = unit.position
        if pos and pos in self.board:
            res = self.sell_unit(is_board=True, loc=pos, pool=pool)
            self.layout_board_tactically()
            return res
        return False

    def equip_item_to_board_slot(self, item_bench_idx: int, board_slot: int) -> bool:
        """Equip item from item bench to board unit at board_slot (0..11)."""
        units = self.get_board_units_list()
        if not (0 <= board_slot < len(units)):
            return False
        unit = units[board_slot]
        pos = unit.position
        if pos and pos in self.board:
            res = self.equip_item(item_bench_idx, is_board=True, target_loc=pos)
            self.layout_board_tactically()
            return res
        return False

    def enforce_board_capacity(self, pool: ChampionPool) -> None:
        """Strictly enforce that board_unit_count <= max_board_units by benching or selling excess units."""
        while self.board_unit_count > self.max_board_units:
            # Sort board units by star level ascending, cost ascending
            sorted_board = sorted(
                self.board.items(),
                key=lambda item: (item[1].star_level, item[1].cost),
            )
            pos, _ = sorted_board[0]
            if self.free_bench_slots > 0:
                empty_bench_idx = next(i for i, s in enumerate(self.bench) if s is None)
                self.move_unit(from_is_board=True, from_loc=pos, to_is_board=False, to_loc=empty_bench_idx)
            else:
                self.sell_unit(is_board=True, loc=pos, pool=pool)

    def can_buy_champion(self, slot_idx: int) -> bool:
        """Check if the player can afford and store the shop card."""
        if not (0 <= slot_idx < len(self.shop.slots)):
            return False
        champ_id = self.shop.slots[slot_idx]
        if champ_id is None:
            return False

        cdef = self.set_data.champions.get(champ_id)
        cost = cdef.cost if cdef else 1
        if self.gold < cost:
            return False

        # If bench has free space, can always buy
        if self.free_bench_slots > 0:
            return True

        # If bench is full, player can still buy if buying completes a 3-in-1 star up!
        star1_copies = sum(
            1 for u in self.get_all_units()
            if u.champion_id == champ_id and u.star_level == 1
        )
        return star1_copies >= 2

    def buy_shop_slot(self, slot_idx: int, pool: ChampionPool) -> bool:
        """Purchase the unit at shop slot_idx."""
        if not self.can_buy_champion(slot_idx):
            return False

        champ_id = self.shop.take(slot_idx)
        if champ_id is None:
            return False

        cdef = self.set_data.champions.get(champ_id)
        cost = cdef.cost if cdef else 1
        self.gold -= cost

        unit = ChampionInstance(champion_id=champ_id, cost=cost, star_level=1)
        added = self.add_champion_to_bench(unit, pool)
        if not added:
            # Fallback if bench was full and star up didn't free space
            self.gold += cost
            pool.return_champion(champ_id, star_level=1)
            return False
        return True

    def sell_unit(
        self,
        is_board: bool,
        loc: int | tuple[int, int],
        pool: ChampionPool,
    ) -> bool:
        """Sell a unit from board or bench, refunding gold and popping items."""
        unit: ChampionInstance | None = None
        if is_board and isinstance(loc, tuple):
            unit = self.board.pop(loc, None)
        elif not is_board and isinstance(loc, int) and 0 <= loc < len(self.bench):
            unit = self.bench[loc]
            self.bench[loc] = None

        if unit is None:
            return False

        # Return items to item bench
        for item_id in unit.items:
            self.add_item(item_id)

        # Refund gold: cost * 3^(star - 1)
        refund_amount = unit.cost * (3 ** (unit.star_level - 1))
        # 1-cost 2-star sells for 2 gold in modern TFT (slight discount), but full refund for 1-stars
        if unit.cost == 1 and unit.star_level == 2:
            refund_amount = 2
        elif unit.cost == 1 and unit.star_level == 3:
            refund_amount = 4

        self.add_gold(refund_amount)
        pool.return_champion(unit.champion_id, unit.star_level)
        return True

    def move_unit(
        self,
        from_is_board: bool,
        from_loc: int | tuple[int, int],
        to_is_board: bool,
        to_loc: int | tuple[int, int],
    ) -> bool:
        """Move or swap a unit between board/bench locations."""
        if from_is_board == to_is_board and from_loc == to_loc:
            return False

        # Retrieve source unit
        source_unit: ChampionInstance | None = None
        if from_is_board and isinstance(from_loc, tuple):
            source_unit = self.board.get(from_loc)
        elif not from_is_board and isinstance(from_loc, int) and 0 <= from_loc < len(self.bench):
            source_unit = self.bench[from_loc]

        if source_unit is None:
            return False

        # Case 1: Move to Board
        if to_is_board and isinstance(to_loc, tuple):
            r, c = to_loc
            if not (0 <= r < self.set_data.board_rows and 0 <= c < self.set_data.board_cols):
                return False

            target_unit = self.board.get(to_loc)

            # If moving from bench to board and destination is empty, verify unit count limit
            if not from_is_board and target_unit is None:
                if self.board_unit_count >= self.max_board_units:
                    return False

            # Remove from source
            if from_is_board and isinstance(from_loc, tuple):
                del self.board[from_loc]
            elif not from_is_board and isinstance(from_loc, int):
                self.bench[from_loc] = target_unit  # Swap if target existed
                if target_unit:
                    target_unit.position = None

            # Place at target
            source_unit.position = to_loc
            self.board[to_loc] = source_unit
            return True

        # Case 2: Move to Bench
        if not to_is_board and isinstance(to_loc, int):
            if not (0 <= to_loc < len(self.bench)):
                return False

            target_unit = self.bench[to_loc]

            # Remove from source
            if from_is_board and isinstance(from_loc, tuple):
                del self.board[from_loc]
                if target_unit is not None:
                    # Put swapped target on board
                    target_unit.position = from_loc
                    self.board[from_loc] = target_unit
            elif not from_is_board and isinstance(from_loc, int):
                self.bench[from_loc] = target_unit

            source_unit.position = None
            self.bench[to_loc] = source_unit
            return True

        return False

    # -------------------------------------------------------------------------
    # Items & Crafting Actions
    # -------------------------------------------------------------------------

    def add_item(self, item_id: str) -> bool:
        """Add an item to the item bench."""
        if len(self.item_bench) >= self.set_data.max_item_bench:
            return False
        idef = self.set_data.items.get(item_id, ItemDef(item_id, item_id, is_component=True))
        self.item_bench.append(ItemInstance(item_id=item_id, name=idef.name, is_component=idef.is_component))
        return True

    def equip_item(
        self,
        item_bench_idx: int,
        is_board: bool,
        target_loc: int | tuple[int, int],
    ) -> bool:
        """Equip or combine an item onto a champion."""
        if not (0 <= item_bench_idx < len(self.item_bench)):
            return False

        unit: ChampionInstance | None = None
        if is_board and isinstance(target_loc, tuple):
            unit = self.board.get(target_loc)
        elif not is_board and isinstance(target_loc, int) and 0 <= target_loc < len(self.bench):
            unit = self.bench[target_loc]

        if unit is None:
            return False

        if len(unit.items) >= 3:
            return False

        item_inst = self.item_bench[item_bench_idx]

        # Check if we can combine two components
        if item_inst.is_component:
            # Check if unit has an uncombined component
            for idx, existing_item in enumerate(unit.items):
                if self.set_data.is_component(existing_item):
                    # Combine!
                    completed_item = self.set_data.get_recipe_result(existing_item, item_inst.item_id)
                    if completed_item:
                        unit.items[idx] = completed_item
                        self.item_bench.pop(item_bench_idx)
                        return True

        # Otherwise equip as a new item slot (if < 3 items)
        if len(unit.items) < 3:
            unit.items.append(item_inst.item_id)
            self.item_bench.pop(item_bench_idx)
            return True

        return False

    def combine_items_on_bench(self, idx1: int, idx2: int, set_data: SetData | None = None) -> bool:
        """Craft a completed item directly on the item bench from two components."""
        if idx1 == idx2 or not (0 <= idx1 < len(self.item_bench)) or not (0 <= idx2 < len(self.item_bench)):
            return False

        sdata = set_data or self.set_data
        item1 = self.item_bench[idx1]
        item2 = self.item_bench[idx2]
        if not item1.is_component or not item2.is_component:
            return False

        completed_id = sdata.get_recipe_result(item1.item_id, item2.item_id)
        if completed_id is None:
            return False

        # Remove both components and add completed item
        indices = sorted([idx1, idx2], reverse=True)
        self.item_bench.pop(indices[0])
        self.item_bench.pop(indices[1])

        idef = sdata.items.get(completed_id, ItemDef(completed_id, completed_id, is_component=False))
        self.item_bench.append(ItemInstance(item_id=completed_id, name=idef.name, is_component=False))
        return True

    # -------------------------------------------------------------------------
    # Experience & Rerolls
    # -------------------------------------------------------------------------

    def buy_exp(self) -> bool:
        """Buy 4 EXP for 4 Gold."""
        if self.gold < self.set_data.exp_buy_cost or self.level >= self.set_data.max_level:
            return False

        self.gold -= self.set_data.exp_buy_cost
        self.exp += self.set_data.exp_buy_amount

        while self.level < self.set_data.max_level:
            req_exp = self.set_data.level_exp.get(self.level, 0)
            if req_exp > 0 and self.exp >= req_exp:
                self.exp -= req_exp
                self.level += 1
            else:
                break
        return True

    def reroll_shop(self, pool: ChampionPool, rng: random.Random | None = None) -> bool:
        """Reroll shop for 2 Gold or using a free reroll token."""
        if self.free_rerolls > 0:
            self.free_rerolls -= 1
        elif self.gold >= self.set_data.reroll_cost:
            self.gold -= self.set_data.reroll_cost
        else:
            return False

        self.shop.refresh(self.level, pool, self.set_data, rng=rng, ignited_slots=self.ignited_shop_slots)
        self.ignited_shop_slots = []  # Consumed on roll
        return True

    def toggle_shop_lock(self) -> None:
        """Toggle shop lock status."""
        self.shop.locked = not self.shop.locked

    # -------------------------------------------------------------------------
    # Consumables Actions
    # -------------------------------------------------------------------------

    def use_duplicator(
        self,
        is_board: bool,
        loc: int | tuple[int, int],
        pool: ChampionPool,
    ) -> bool:
        """Use a Champion Duplicator on target champion to clone a 1-star copy."""
        if self.duplicators <= 0:
            return False

        unit: ChampionInstance | None = None
        if is_board and isinstance(loc, tuple):
            unit = self.board.get(loc)
        elif not is_board and isinstance(loc, int) and 0 <= loc < len(self.bench):
            unit = self.bench[loc]

        if unit is None:
            return False

        cdef = self.set_data.champions.get(unit.champion_id)
        cost = cdef.cost if cdef else unit.cost
        clone = ChampionInstance(champion_id=unit.champion_id, cost=cost, star_level=1)
        added = self.add_champion_to_bench(clone, pool=pool, allow_board_overflow=True)
        if added:
            self.duplicators -= 1
            return True
        return False

    def use_remover(
        self,
        is_board: bool,
        loc: int | tuple[int, int],
    ) -> bool:
        """Use a Magnetic Remover to pop all equipped items off a champion."""
        if self.removers <= 0:
            return False

        unit: ChampionInstance | None = None
        if is_board and isinstance(loc, tuple):
            unit = self.board.get(loc)
        elif not is_board and isinstance(loc, int) and 0 <= loc < len(self.bench):
            unit = self.bench[loc]

        if unit is None or not unit.items:
            return False

        # Pop all items to item bench
        items_to_pop = list(unit.items)
        unit.items.clear()
        for item_id in items_to_pop:
            self.add_item(item_id)

        self.removers -= 1
        return True

    def use_reforger(
        self,
        item_bench_idx: int,
        rng: random.Random | None = None,
    ) -> bool:
        """Use a Reforger to reroll an item on the bench into another random one."""
        if self.reforgers <= 0 or not (0 <= item_bench_idx < len(self.item_bench)):
            return False

        item_inst = self.item_bench[item_bench_idx]
        r = rng or random

        if item_inst.is_component:
            candidates = [c for c in self.set_data.components if c != item_inst.item_id]
            if candidates:
                new_item_id = r.choice(candidates)
                idef = self.set_data.items.get(new_item_id)
                self.item_bench[item_bench_idx] = ItemInstance(
                    item_id=new_item_id,
                    name=idef.name if idef else new_item_id,
                    is_component=True,
                )
                self.reforgers -= 1
                return True
        else:
            candidates = [i for i, idef in self.set_data.items.items() if not idef.is_component and i != item_inst.item_id]
            if candidates:
                new_item_id = r.choice(candidates)
                idef = self.set_data.items.get(new_item_id)
                self.item_bench[item_bench_idx] = ItemInstance(
                    item_id=new_item_id,
                    name=idef.name if idef else new_item_id,
                    is_component=False,
                )
                self.reforgers -= 1
                return True

        return False

    # -------------------------------------------------------------------------
    # State Serialization
    # -------------------------------------------------------------------------

    def to_feature_board(self) -> list[dict[str, Any]]:
        """Return list of champion dictionaries formatted for feature extractor."""
        return [unit.to_dict() for unit in self.board.values()]

    def to_dict_state(self) -> dict[str, Any]:
        """Convert full player state into an inspectable dictionary."""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "health": self.health,
            "gold": self.gold,
            "level": self.level,
            "exp": self.exp,
            "streak": self.streak,
            "alive": self.alive,
            "placement": self.placement,
            "board": [u.to_dict() for u in self.board.values()],
            "bench": [u.to_dict() for u in self.bench if u is not None],
            "item_bench": [it.item_id for it in self.item_bench],
            "shop": list(self.shop.slots),
            "shop_locked": self.shop.locked,
            "board_value": self.get_board_value(),
            "active_traits": self.get_active_traits(),
            "augments": list(self.augments),
            "duplicators": self.duplicators,
            "reforgers": self.reforgers,
            "removers": self.removers,
            "free_rerolls": self.free_rerolls,
            "coven_essence": self.coven_essence,
            "fae_pixies": self.fae_pixies,
        }
