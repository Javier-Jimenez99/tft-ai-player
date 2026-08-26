"""Data-driven, set-agnostic configuration models for TFT simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class UnitRole(StrEnum):
    """High-level combat role for champions."""

    AD_CARRY = "AD_CARRY"
    AP_CARRY = "AP_CARRY"
    TANK = "TANK"
    UTILITY = "UTILITY"


@dataclass(frozen=True, slots=True)
class ChampionDef:
    """Static definition of a champion."""

    champion_id: str
    name: str
    cost: int
    traits: tuple[str, ...] = ()
    role: UnitRole = UnitRole.UTILITY


@dataclass(frozen=True, slots=True)
class TraitDef:
    """Static definition of a trait and its activation breakpoints."""

    trait_id: str
    name: str
    thresholds: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ItemDef:
    """Static definition of an item (component or completed)."""

    item_id: str
    name: str
    is_component: bool = False
    recipe: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class SetData:
    """Set-agnostic data bundle defining all champions, items, traits, odds, and rules."""

    set_name: str
    champions: dict[str, ChampionDef]
    traits: dict[str, TraitDef]
    items: dict[str, ItemDef]
    recipes: dict[frozenset[str], str] = field(default_factory=dict)
    shop_odds: dict[int, tuple[float, float, float, float, float]] = field(default_factory=dict)
    pool_sizes: dict[int, int] = field(default_factory=dict)
    level_exp: dict[int, int] = field(default_factory=dict)
    stage_base_damage: dict[int, int] = field(default_factory=dict)
    streak_gold_thresholds: tuple[tuple[int, int], ...] = (
        (6, 3),
        (5, 2),
        (3, 1),
    )
    passive_gold_schedule: dict[str, int] = field(default_factory=dict)
    exp_buy_cost: int = 4
    exp_buy_amount: int = 4
    reroll_cost: int = 2
    interest_rate: float = 0.1
    max_interest: int = 5
    max_bench_size: int = 9
    max_item_bench: int = 10
    board_rows: int = 4
    board_cols: int = 7
    starting_hp: int = 100
    starting_gold: int = 0
    max_level: int = 10

    def get_recipe_result(self, comp1: str, comp2: str) -> str | None:
        """Return the completed item resulting from combining two components."""
        key = frozenset([comp1, comp2])
        return self.recipes.get(key)

    def is_component(self, item_id: str) -> bool:
        """Return whether an item is a basic component."""
        item_def = self.items.get(item_id)
        return item_def.is_component if item_def else False


TEAM_SIZE_EXPANDING_ITEMS: frozenset[str] = frozenset([
    "TFT_Item_TacticiansCrown",
    "TFT_Item_ForceOfNature",
    "TFT_Item_TacticiansShield",
    "TFT_Item_TacticiansScepter",
    "TFT_Item_TacticiansCape",
    "TFT_Item_TacticiansRing",
    "TFT_Item_ChonccsCrown",
    "TFT_Item_ChonccsChalice",
])


# =============================================================================
# STANDARD TFT RULES & TABLES
# =============================================================================

STANDARD_SHOP_ODDS: dict[int, tuple[float, float, float, float, float]] = {
    1: (1.00, 0.00, 0.00, 0.00, 0.00),
    2: (1.00, 0.00, 0.00, 0.00, 0.00),
    3: (0.75, 0.25, 0.00, 0.00, 0.00),
    4: (0.55, 0.30, 0.15, 0.00, 0.00),
    5: (0.45, 0.33, 0.20, 0.02, 0.00),
    6: (0.30, 0.40, 0.25, 0.05, 0.00),
    7: (0.19, 0.30, 0.40, 0.10, 0.01),
    8: (0.15, 0.20, 0.32, 0.30, 0.03),
    9: (0.10, 0.17, 0.25, 0.33, 0.15),
    10: (0.05, 0.10, 0.20, 0.40, 0.25),
    11: (0.01, 0.02, 0.12, 0.50, 0.35),
}

STANDARD_POOL_SIZES: dict[int, int] = {
    1: 30,
    2: 25,
    3: 18,
    4: 10,
    5: 9,
}

STANDARD_LEVEL_EXP: dict[int, int] = {
    1: 2,   # 1 -> 2: 2 XP
    2: 2,   # 2 -> 3: 2 XP
    3: 6,   # 3 -> 4: 6 XP
    4: 10,  # 4 -> 5: 10 XP
    5: 20,  # 5 -> 6: 20 XP
    6: 36,  # 6 -> 7: 36 XP
    7: 60,  # 7 -> 8: 60 XP
    8: 68,  # 8 -> 9: 68 XP
    9: 68,  # 9 -> 10: 68 XP
    10: 0,  # Max level
}

STANDARD_STAGE_BASE_DAMAGE: dict[int, int] = {
    1: 0,
    2: 2,
    3: 6,
    4: 7,
    5: 10,
    6: 12,
    7: 17,
    8: 25,
}

STANDARD_PASSIVE_GOLD: dict[str, int] = {
    "1-1": 0,
    "1-2": 2,
    "1-3": 2,
    "1-4": 3,
    "2-1": 4,
    "default": 5,
}


# =============================================================================
# STANDARD TFT ITEM REGISTRY & RECIPES
# =============================================================================

STANDARD_COMPONENTS: list[tuple[str, str]] = [
    ("TFT_Item_BFSword", "B.F. Sword"),
    ("TFT_Item_RecurveBow", "Recurve Bow"),
    ("TFT_Item_NeedlesslyLargeRod", "Needlessly Large Rod"),
    ("TFT_Item_TearOfTheGoddess", "Tear of the Goddess"),
    ("TFT_Item_ChainVest", "Chain Vest"),
    ("TFT_Item_NegatronCloak", "Negatron Cloak"),
    ("TFT_Item_GiantsBelt", "Giant's Belt"),
    ("TFT_Item_SparringGloves", "Sparring Gloves"),
    ("TFT_Item_Spatula", "Spatula"),
]

# Standard recipe combinations (Component A, Component B -> Completed Item ID, Name)
STANDARD_COMPLETED_ITEMS_TABLE: list[tuple[str, str, str, str]] = [
    # BF Sword combos
    ("TFT_Item_BFSword", "TFT_Item_BFSword", "TFT_Item_Deathblade", "Deathblade"),
    ("TFT_Item_BFSword", "TFT_Item_RecurveBow", "TFT_Item_GiantSlayer", "Giant Slayer"),
    ("TFT_Item_BFSword", "TFT_Item_NeedlesslyLargeRod", "TFT_Item_HextechGunblade", "Hextech Gunblade"),
    ("TFT_Item_BFSword", "TFT_Item_TearOfTheGoddess", "TFT_Item_SpearOfShojin", "Spear of Shojin"),
    ("TFT_Item_BFSword", "TFT_Item_ChainVest", "TFT_Item_EdgeOfNight", "Edge of Night"),
    ("TFT_Item_BFSword", "TFT_Item_NegatronCloak", "TFT_Item_Bloodthirster", "Bloodthirster"),
    ("TFT_Item_BFSword", "TFT_Item_GiantsBelt", "TFT_Item_SteraksGage", "Sterak's Gage"),
    ("TFT_Item_BFSword", "TFT_Item_SparringGloves", "TFT_Item_InfinityEdge", "Infinity Edge"),
    ("TFT_Item_Spatula", "TFT_Item_Spatula", "TFT_Item_TacticiansCrown", "Tactician's Crown"),

    # Recurve Bow combos
    ("TFT_Item_RecurveBow", "TFT_Item_RecurveBow", "TFT_Item_RapidFireCannon", "Red Buff / Rapid Firecannon"),
    ("TFT_Item_RecurveBow", "TFT_Item_NeedlesslyLargeRod", "TFT_Item_GuinsoosRageblade", "Guinsoo's Rageblade"),
    ("TFT_Item_RecurveBow", "TFT_Item_TearOfTheGoddess", "TFT_Item_StatikkShiv", "Statikk Shiv"),
    ("TFT_Item_RecurveBow", "TFT_Item_ChainVest", "TFT_Item_TitansResolve", "Titan's Resolve"),
    ("TFT_Item_RecurveBow", "TFT_Item_NegatronCloak", "TFT_Item_RunaansHurricane", "Runaan's Hurricane"),
    ("TFT_Item_RecurveBow", "TFT_Item_GiantsBelt", "TFT_Item_NashorsTooth", "Nashor's Tooth"),
    ("TFT_Item_RecurveBow", "TFT_Item_SparringGloves", "TFT_Item_LastWhisper", "Last Whisper"),

    # Needlessly Large Rod combos
    ("TFT_Item_NeedlesslyLargeRod", "TFT_Item_NeedlesslyLargeRod", "TFT_Item_RabadonsDeathcap", "Rabadon's Deathcap"),
    ("TFT_Item_NeedlesslyLargeRod", "TFT_Item_TearOfTheGoddess", "TFT_Item_ArchangelsStaff", "Archangel's Staff"),
    ("TFT_Item_NeedlesslyLargeRod", "TFT_Item_ChainVest", "TFT_Item_Crownguard", "Crownguard"),
    ("TFT_Item_NeedlesslyLargeRod", "TFT_Item_NegatronCloak", "TFT_Item_IonicSpark", "Ionic Spark"),
    ("TFT_Item_NeedlesslyLargeRod", "TFT_Item_GiantsBelt", "TFT_Item_Morellonomicon", "Morellonomicon"),
    ("TFT_Item_NeedlesslyLargeRod", "TFT_Item_SparringGloves", "TFT_Item_JeweledGauntlet", "Jeweled Gauntlet"),

    # Tear of the Goddess combos
    ("TFT_Item_TearOfTheGoddess", "TFT_Item_TearOfTheGoddess", "TFT_Item_BlueBuff", "Blue Buff"),
    ("TFT_Item_TearOfTheGoddess", "TFT_Item_ChainVest", "TFT_Item_FrozenHeart", "Protector's Vow"),
    ("TFT_Item_TearOfTheGoddess", "TFT_Item_NegatronCloak", "TFT_Item_AdaptiveHelm", "Adaptive Helm"),
    ("TFT_Item_TearOfTheGoddess", "TFT_Item_GiantsBelt", "TFT_Item_Redemption", "Redemption"),
    ("TFT_Item_TearOfTheGoddess", "TFT_Item_SparringGloves", "TFT_Item_HandOfJustice", "Hand of Justice"),

    # Chain Vest combos
    ("TFT_Item_ChainVest", "TFT_Item_ChainVest", "TFT_Item_BrambleVest", "Bramble Vest"),
    ("TFT_Item_ChainVest", "TFT_Item_NegatronCloak", "TFT_Item_GargoyleStoneplate", "Gargoyle Stoneplate"),
    ("TFT_Item_ChainVest", "TFT_Item_GiantsBelt", "TFT_Item_SunfireCape", "Sunfire Cape"),
    ("TFT_Item_ChainVest", "TFT_Item_SparringGloves", "TFT_Item_NightHarvester", "Steadfast Heart"),

    # Negatron Cloak combos
    ("TFT_Item_NegatronCloak", "TFT_Item_NegatronCloak", "TFT_Item_DragonsClaw", "Dragon's Claw"),
    ("TFT_Item_NegatronCloak", "TFT_Item_GiantsBelt", "TFT_Item_Evenshroud", "Evenshroud"),
    ("TFT_Item_NegatronCloak", "TFT_Item_SparringGloves", "TFT_Item_Quicksilver", "Quicksilver"),

    # Giant's Belt combos
    ("TFT_Item_GiantsBelt", "TFT_Item_GiantsBelt", "TFT_Item_WarmogsArmor", "Warmog's Armor"),
    ("TFT_Item_GiantsBelt", "TFT_Item_SparringGloves", "TFT_Item_Guardbreaker", "Guardbreaker"),

    # Sparring Gloves combos
    ("TFT_Item_SparringGloves", "TFT_Item_SparringGloves", "TFT_Item_ThiefsGloves", "Thief's Gloves"),
]


def build_standard_items() -> tuple[dict[str, ItemDef], dict[frozenset[str], str]]:
    """Build item registry and recipe mapping."""
    items: dict[str, ItemDef] = {}
    recipes: dict[frozenset[str], str] = {}

    for item_id, name in STANDARD_COMPONENTS:
        items[item_id] = ItemDef(
            item_id=item_id,
            name=name,
            is_component=True,
            recipe=None,
        )

    for comp1, comp2, res_id, res_name in STANDARD_COMPLETED_ITEMS_TABLE:
        items[res_id] = ItemDef(
            item_id=res_id,
            name=res_name,
            is_component=False,
            recipe=(comp1, comp2),
        )
        recipes[frozenset([comp1, comp2])] = res_id

    return items, recipes


# =============================================================================
# SET 17 DEFAULT DATA FACTORY
# =============================================================================

def get_default_set17_data() -> SetData:
    """Instantiate standard Set 17 game configuration from official profile."""
    from tft_ai_player.simulation.sets.set17 import get_set17_data

    return get_set17_data()
