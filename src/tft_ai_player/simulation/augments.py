"""Set 18 Augment System: Definitions, Registry, Offering Engine, and Effect Dispatcher."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tft_ai_player.simulation.config import SetData
    from tft_ai_player.simulation.models import ChampionPool, Player


class AugmentTier(StrEnum):
    """Augment rarity tier classification."""

    SILVER = "SILVER"
    GOLD = "GOLD"
    PRISMATIC = "PRISMATIC"


class AugmentCategory(StrEnum):
    """Categorization of augment mechanics."""

    ECONOMY = "ECONOMY"
    ITEM = "ITEM"
    LEVEL_XP = "LEVEL_XP"
    SHOP_REROLL = "SHOP_REROLL"
    TRAIT_EMBLEM = "TRAIT_EMBLEM"
    TEAM_SIZE = "TEAM_SIZE"
    COMBAT = "COMBAT"


@dataclass(frozen=True, slots=True)
class AugmentDef:
    """Specification of a TFT Augment."""

    augment_id: str
    name: str
    tier: AugmentTier
    category: AugmentCategory
    description: str
    instant_gold: int = 0
    instant_xp: int = 0
    instant_components: int = 0
    instant_items: tuple[str, ...] = ()
    instant_duplicators: int = 0
    instant_reforgers: int = 0
    instant_removers: int = 0
    instant_champions: tuple[tuple[str, int], ...] = ()  # ((champion_id, count), ...)
    instant_team_size: int = 0
    max_interest_cap: int | None = None
    free_rerolls_per_round: int = 0
    xp_per_round: int = 0
    gold_per_round: int = 0
    associated_traits: tuple[str, ...] = ()


from tft_ai_player.simulation.sets.set18.augments import SET18_AUGMENTS_CATALOG


# =============================================================================
# AUGMENT MANAGER & EVENT DISPATCHER
# =============================================================================

class AugmentManager:
    """Manages lobby augment offerings, player selection, and lifecycle event dispatches."""

    AUGMENT_STAGES: tuple[str, ...] = ("2-1", "3-2", "4-2")

    def __init__(self, catalog: dict[str, AugmentDef] | None = None) -> None:
        self.catalog: dict[str, AugmentDef] = catalog or dict(SET18_AUGMENTS_CATALOG)
        self.by_tier: dict[AugmentTier, list[AugmentDef]] = {
            AugmentTier.SILVER: [a for a in self.catalog.values() if a.tier == AugmentTier.SILVER],
            AugmentTier.GOLD: [a for a in self.catalog.values() if a.tier == AugmentTier.GOLD],
            AugmentTier.PRISMATIC: [a for a in self.catalog.values() if a.tier == AugmentTier.PRISMATIC],
        }

    def is_augment_round(self, stage_str: str) -> bool:
        """True if the given stage string is a standard augment offering round."""
        return stage_str in self.AUGMENT_STAGES

    def get_round_augment_tier(self, stage_str: str, rng: random.Random | None = None) -> AugmentTier:
        """Determine global lobby augment tier for a given stage."""
        r = rng or random
        if stage_str == "2-1":
            tier_weights = (0.50, 0.40, 0.10)
        elif stage_str == "3-2":
            tier_weights = (0.25, 0.60, 0.15)
        else:  # 4-2
            tier_weights = (0.15, 0.55, 0.30)

        return r.choices(
            [AugmentTier.SILVER, AugmentTier.GOLD, AugmentTier.PRISMATIC],
            weights=tier_weights,
            k=1,
        )[0]

    def generate_augment_choices(
        self,
        player: Player,
        stage_str: str,
        rng: random.Random | None = None,
        k: int = 3,
        target_tier: AugmentTier | None = None,
    ) -> list[AugmentDef]:
        """Sample k distinct augment candidates for a player of the specified tier."""
        r = rng or random

        tier = target_tier or self.get_round_augment_tier(stage_str, r)
        candidates = [a for a in self.by_tier[tier] if a.augment_id not in player.augments]
        if len(candidates) < k:
            candidates = [a for a in self.catalog.values() if a.augment_id not in player.augments]

        return r.sample(candidates, min(k, len(candidates)))

    def apply_augment(
        self,
        player: Player,
        augment_id: str,
        pool: ChampionPool,
        set_data: SetData,
        rng: random.Random | None = None,
    ) -> bool:
        """Equip an augment on a player and execute its instant reward payloads."""
        adef = self.catalog.get(augment_id)
        if adef is None:
            return False

        player.augments.append(augment_id)

        # 1. Instant Gold
        if adef.instant_gold > 0:
            player.add_gold(adef.instant_gold)

        # 2. Instant XP
        if adef.instant_xp > 0:
            player.exp += adef.instant_xp
            while player.level < set_data.max_level:
                req_exp = set_data.level_exp.get(player.level, 0)
                if req_exp > 0 and player.exp >= req_exp:
                    player.exp -= req_exp
                    player.level += 1
                else:
                    break

        # 3. Instant Components & Items
        if adef.instant_components > 0:
            comps = list(set_data.components)
            r = rng or random
            for _ in range(adef.instant_components):
                chosen_comp = r.choice(comps)
                player.add_item(chosen_comp)

        for item_id in adef.instant_items:
            player.add_item(item_id)

        # 4. Instant Consumables
        if adef.instant_duplicators > 0:
            player.duplicators += adef.instant_duplicators
        if adef.instant_reforgers > 0:
            player.reforgers += adef.instant_reforgers
        if adef.instant_removers > 0:
            player.removers += adef.instant_removers

        # 5. Instant Champions
        for champ_id, count in adef.instant_champions:
            cdef = set_data.champions.get(champ_id)
            cost = cdef.cost if cdef else 1
            for _ in range(count):
                from tft_ai_player.simulation.models import ChampionInstance
                cinst = ChampionInstance(champion_id=champ_id, cost=cost, star_level=1)
                player.add_champion_to_bench(cinst, pool=pool, allow_board_overflow=True)

        # 6. Max Interest Cap Modifier
        if adef.max_interest_cap is not None:
            player.max_interest_cap = adef.max_interest_cap

        # 7. Team Size Modifier
        if adef.instant_team_size > 0:
            player.extra_team_size += adef.instant_team_size

        return True

    def execute_round_start_augments(
        self,
        player: Player,
        set_data: SetData,
    ) -> None:
        """Apply passive per-round resources granted by active augments."""
        for aug_id in player.augments:
            adef = self.catalog.get(aug_id)
            if adef is None:
                continue

            # Per-round Gold
            if adef.gold_per_round > 0:
                player.add_gold(adef.gold_per_round)

            # Per-round Free Rerolls
            if adef.free_rerolls_per_round > 0:
                player.free_rerolls += adef.free_rerolls_per_round

            # Per-round XP
            if adef.xp_per_round > 0:
                player.exp += adef.xp_per_round
                while player.level < set_data.max_level:
                    req_exp = set_data.level_exp.get(player.level, 0)
                    if req_exp > 0 and player.exp >= req_exp:
                        player.exp -= req_exp
                        player.level += 1
                    else:
                        break
