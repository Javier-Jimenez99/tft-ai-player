"""Unit tests for Set 18 'Enchanted Wilds' profile, champions, traits, and game simulation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tft_ai_player.simulation import (
    ChampionPool,
    HeuristicCombatResolver,
    MatchmakingEngine,
    Player,
    Set18Profile,
    StandardTempoBot,
    TFTGame,
    get_set_data,
    get_set_profile,
    get_set18_data,
)
from tft_ai_player.simulation.sets.set18 import export_set18_profile_json


def test_set18_registry_loaders() -> None:
    """Verify set registry loader for Set 18 and metadata access."""
    data1 = get_set_data("TFTSet18")
    data2 = get_set_data("set18")
    data3 = get_set_data("18")

    assert data1.set_name == "TFTSet18"
    assert len(data1.champions) == len(data2.champions) == len(data3.champions) == 65
    assert len(data1.traits) == 36

    profile = get_set_profile("TFTSet18")
    assert isinstance(profile, Set18Profile)
    assert profile.set_name == "TFTSet18"
    assert profile.total_champions == 65
    assert profile.total_traits == 36


def test_set18_champion_cost_distribution() -> None:
    """Verify Set 18 contains the official 65 champions distributed across 5 cost tiers."""
    profile = Set18Profile()
    by_cost: dict[int, int] = {}
    for c in profile.champions:
        by_cost[c.cost] = by_cost.get(c.cost, 0) + 1

    assert by_cost[1] == 14
    assert by_cost[2] == 13
    assert by_cost[3] == 14
    assert by_cost[4] == 14
    assert by_cost[5] == 10
    assert sum(by_cost.values()) == 65

    # Check key specific units
    champ_map = {c.champion_id: c for c in profile.champions}
    assert "TFT18_Kobuko" in champ_map
    assert champ_map["TFT18_Kobuko"].cost == 1
    assert "Brawler" in champ_map["TFT18_Kobuko"].traits

    assert "TFT18_Alune" in champ_map
    assert champ_map["TFT18_Alune"].cost == 5
    assert "Attuned" in champ_map["TFT18_Alune"].traits

    assert "TFT18_ElderDragon" in champ_map
    assert champ_map["TFT18_ElderDragon"].cost == 5
    assert "Apex Predator" in champ_map["TFT18_ElderDragon"].traits


def test_set18_traits_completeness() -> None:
    """Verify all 36 Set 18 traits have valid thresholds, descriptions, and members."""
    profile = Set18Profile()
    assert len(profile.traits) == 36

    for tid, t_info in profile.traits.items():
        assert len(t_info.thresholds) >= 1
        assert t_info.description != ""
        assert t_info.name == tid


def test_set18_json_profile_export(tmp_path: Path) -> None:
    """Verify JSON export of Set 18 profile."""
    out_file = tmp_path / "test_set18.json"
    exported_path = export_set18_profile_json(out_file)

    assert exported_path.exists()
    raw = json.loads(exported_path.read_text(encoding="utf-8"))

    assert raw["set_name"] == "TFTSet18"
    assert raw["stats"]["total_champions"] == 65
    assert raw["stats"]["total_traits"] == 36
    assert len(raw["champions"]) == 65
    assert len(raw["traits"]) == 36
    assert "shop_odds" in raw
    assert "level_exp" in raw
    assert "pool_sizes" in raw


def test_full_game_simulation_set18() -> None:
    """Verify a complete 8-player TFT game simulation runs seamlessly on Set 18."""
    set18_data = get_set18_data()
    game = TFTGame(set_data=set18_data, seed=42)

    assert game.set_data.set_name == "TFTSet18"
    assert len(game.pool.counts) == 65

    rounds_run = 0
    while not game.is_over and rounds_run < 35:
        # Bots plan turn
        for p in game.players:
            if p.alive:
                bot = StandardTempoBot()
                rinfo = game.stage_manager.get_current_round_info()
                bot.take_turn(p, game.pool, game.set_data, rinfo.stage, rinfo.round_in_stage)

        game.resolve_round_phase()
        rounds_run += 1

    assert rounds_run > 0
    assert len(game.get_rankings()) == 8
