"""Unit tests for Set 17 profile, champion roster, traits, recipes, and set loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tft_ai_player.round_winner.features import CHAMP_BASE_COSTS, SET17_TRAIT_CHAMPIONS
from tft_ai_player.simulation.sets import (
    AVAILABLE_SETS,
    Set17Profile,
    export_set17_profile_json,
    get_set_data,
    get_set_profile,
    get_set17_data,
)


def test_set_registry_loaders() -> None:
    """Verify set registry loader for Set 17 and rejection of unsupported sets."""
    assert "TFTSet17" in AVAILABLE_SETS

    # Valid lookups with case/format tolerance
    data1 = get_set_data("TFTSet17")
    data2 = get_set_data("set17")
    data3 = get_set_data("17")
    assert data1.set_name == "TFTSet17"
    assert len(data1.champions) == len(data2.champions) == len(data3.champions) == 63

    profile = get_set_profile("TFTSet17")
    assert isinstance(profile, Set17Profile)
    assert profile.set_name == "TFTSet17"

    with pytest.raises(ValueError, match="Unsupported TFT set"):
        get_set_data("TFTSet99")


def test_set17_champions_match_model_features() -> None:
    """Verify Set 17 champion IDs, costs, and traits match ML model feature extractor."""
    profile = Set17Profile()
    champ_map = {c.champion_id: c for c in profile.champions}

    assert len(profile.champions) == 63

    # All Set 17 champions exist in model feature extractor with identical costs
    for champ in profile.champions:
        assert champ.champion_id in CHAMP_BASE_COSTS, f"Missing champion {champ.champion_id} in CHAMP_BASE_COSTS"
        assert CHAMP_BASE_COSTS[champ.champion_id] == champ.cost, f"Cost mismatch for {champ.champion_id}"
        assert champ.name != ""

    # All traits have valid thresholds and member champions
    for trait_id, expected_members in SET17_TRAIT_CHAMPIONS.items():
        assert trait_id in profile.traits, f"Missing trait {trait_id} in Set 17 profile"
        t_info = profile.traits[trait_id]
        assert len(t_info.thresholds) >= 1
        for m in expected_members:
            assert m in champ_map, f"Missing member {m} in champion catalog"
            assert trait_id in champ_map[m].traits, f"Champion {m} should have trait {trait_id}"


def test_set17_items_and_recipes_completeness() -> None:
    """Verify Set 17 item registry contains all 36 completed item recipes and Spatula."""
    set_data = get_set17_data()

    # 9 components (8 standard + Spatula)
    components = [cid for cid, idef in set_data.items.items() if idef.is_component]
    assert len(components) == 9
    assert "TFT_Item_BFSword" in components
    assert "TFT_Item_Spatula" in components

    # 36 core completed items + Tactician's Crown + 8 Set 17 Emblems = 45 recipes
    assert len(set_data.recipes) == 45
    assert set_data.get_recipe_result("TFT_Item_BFSword", "TFT_Item_BFSword") == "TFT_Item_Deathblade"
    assert set_data.get_recipe_result("TFT_Item_BFSword", "TFT_Item_NeedlesslyLargeRod") == "TFT_Item_HextechGunblade"
    assert set_data.get_recipe_result("TFT_Item_Spatula", "TFT_Item_Spatula") == "TFT_Item_TacticiansCrown"
    assert set_data.get_recipe_result("TFT_Item_Spatula", "TFT_Item_BFSword") == "TFT17_Item_DarkStarEmblemItem"
    assert set_data.get_recipe_result("TFT_Item_Spatula", "TFT_Item_NeedlesslyLargeRod") == "TFT17_Item_StargazerEmblemItem"


def test_set17_json_profile_export(tmp_path: Path) -> None:
    """Verify JSON export of Set 17 profile."""
    out_file = tmp_path / "test_set17.json"
    exported_path = export_set17_profile_json(out_file)

    assert exported_path.exists()
    raw = json.loads(exported_path.read_text(encoding="utf-8"))

    assert raw["set_name"] == "TFTSet17"
    assert raw["stats"]["total_champions"] == 63
    assert raw["stats"]["total_traits"] == 35
    assert len(raw["champions"]) == 63
    assert len(raw["traits"]) == 35
    assert "shop_odds" in raw
    assert "level_exp" in raw
    assert "pool_sizes" in raw
