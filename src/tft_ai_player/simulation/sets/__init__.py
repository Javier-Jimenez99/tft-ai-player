"""Set data registry and set loader for TFT simulations."""

from __future__ import annotations

from typing import Any

from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.sets.set17 import (
    SET17_CHAMPION_CATALOG,
    SET17_TRAIT_CATALOG,
    Set17ChampionInfo,
    Set17Profile,
    Set17TraitInfo,
    export_set17_profile_json,
    get_set17_data,
)
from tft_ai_player.simulation.sets.set18 import (
    SET18_CHAMPION_CATALOG,
    SET18_TRAIT_CATALOG,
    Set18ChampionInfo,
    Set18Profile,
    Set18TraitInfo,
    export_set18_profile_json,
    get_set18_data,
)

AVAILABLE_SETS: list[str] = ["TFTSet17", "TFTSet18"]


def get_set_data(set_name: str = "TFTSet17") -> SetData:
    """Retrieve executable SetData instance for a specified TFT set."""
    normalized = set_name.replace(" ", "").replace("_", "").lower()
    if normalized in ("tftset17", "set17", "17"):
        return get_set17_data()
    if normalized in ("tftset18", "set18", "18"):
        return get_set18_data()

    raise ValueError(f"Unsupported TFT set '{set_name}'. Available sets: {AVAILABLE_SETS}")


def get_set_profile(set_name: str = "TFTSet17") -> Any:
    """Retrieve detailed Set profile metadata object for a specified TFT set."""
    normalized = set_name.replace(" ", "").replace("_", "").lower()
    if normalized in ("tftset17", "set17", "17"):
        return Set17Profile()
    if normalized in ("tftset18", "set18", "18"):
        return Set18Profile()

    raise ValueError(f"Unsupported TFT set '{set_name}'. Available sets: {AVAILABLE_SETS}")


__all__ = [
    "AVAILABLE_SETS",
    "SET17_CHAMPION_CATALOG",
    "SET17_TRAIT_CATALOG",
    "SET18_CHAMPION_CATALOG",
    "SET18_TRAIT_CATALOG",
    "Set17ChampionInfo",
    "Set17Profile",
    "Set17TraitInfo",
    "Set18ChampionInfo",
    "Set18Profile",
    "Set18TraitInfo",
    "export_set17_profile_json",
    "export_set18_profile_json",
    "get_set_data",
    "get_set_profile",
    "get_set17_data",
    "get_set18_data",
]
