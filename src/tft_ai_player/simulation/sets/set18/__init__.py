"""Set 18 'Enchanted Wilds' Package: Champions, Traits, Items, Augments, and Custom Mechanics."""

from __future__ import annotations

from tft_ai_player.simulation.sets.set18.augments import SET18_AUGMENTS_CATALOG
from tft_ai_player.simulation.sets.set18.mechanics import Set18MechanicsHandler
from tft_ai_player.simulation.sets.set18.profile import (
    SET18_CHAMPION_CATALOG,
    SET18_ITEM_RECIPES,
    SET18_ITEMS_CATALOG,
    SET18_TRAIT_CATALOG,
    Set18ChampionInfo,
    Set18ItemInfo,
    Set18Profile,
    Set18TraitInfo,
    export_set18_profile_json,
    get_set18_data,
)

__all__ = [
    "SET18_AUGMENTS_CATALOG",
    "SET18_CHAMPION_CATALOG",
    "SET18_ITEMS_CATALOG",
    "SET18_ITEM_RECIPES",
    "SET18_TRAIT_CATALOG",
    "Set18ChampionInfo",
    "Set18ItemInfo",
    "Set18MechanicsHandler",
    "Set18Profile",
    "Set18TraitInfo",
    "export_set18_profile_json",
    "get_set18_data",
]
