"""Unit test suite for Set 18 Augments, Economy Traits, Consumables, and Loot Systems."""

from __future__ import annotations

import random
import pytest

from tft_ai_player.simulation.augments import (
    SET18_AUGMENTS_CATALOG,
    AugmentCategory,
    AugmentDef,
    AugmentManager,
    AugmentTier,
)
from tft_ai_player.simulation.combat import CombatResult
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.game import TFTGame
from tft_ai_player.simulation.models import (
    ChampionInstance,
    ChampionPool,
    ItemInstance,
    Player,
)
from tft_ai_player.simulation.sets import get_set18_data
from tft_ai_player.simulation.stage_manager import StageManager


@pytest.fixture
def set18_data() -> SetData:
    """Fixture providing complete Set 18 data."""
    return get_set18_data()


@pytest.fixture
def pool(set18_data: SetData) -> ChampionPool:
    """Fixture providing a fresh champion pool."""
    return ChampionPool(set18_data)


@pytest.fixture
def player(set18_data: SetData) -> Player:
    """Fixture providing a clean player instance."""
    return Player(player_id=0, set_data=set18_data, name="Test Player")


def test_augment_catalog_integrity():
    """Verify that Set 18 augment catalog is well-defined with appropriate tiers and categories."""
    assert len(SET18_AUGMENTS_CATALOG) >= 20
    for aug_id, adef in SET18_AUGMENTS_CATALOG.items():
        assert adef.augment_id == aug_id
        assert adef.name != ""
        assert adef.tier in (AugmentTier.SILVER, AugmentTier.GOLD, AugmentTier.PRISMATIC)
        assert isinstance(adef.category, AugmentCategory)


def test_augment_manager_offerings(player: Player):
    """Test that AugmentManager generates distinct choices and respects stage tiers."""
    mgr = AugmentManager()
    assert mgr.is_augment_round("2-1")
    assert mgr.is_augment_round("3-2")
    assert mgr.is_augment_round("4-2")
    assert not mgr.is_augment_round("2-2")

    choices_21 = mgr.generate_augment_choices(player, "2-1", rng=random.Random(42), k=3)
    assert len(choices_21) == 3
    assert len({c.augment_id for c in choices_21}) == 3


def test_augment_instant_effects(player: Player, pool: ChampionPool, set18_data: SetData):
    """Test instant effects of economy and item augments."""
    mgr = AugmentManager()

    # 1. Rich Get Richer (Gold + Interest Cap 7)
    starting_gold = player.gold
    applied = mgr.apply_augment(player, "DA_18_RichGetRicher", pool=pool, set_data=set18_data)
    assert applied
    assert player.gold == starting_gold + 12
    assert player.max_interest_cap == 7
    assert "DA_18_RichGetRicher" in player.augments

    # Check interest at 65 gold with 7g cap
    player.gold = 65
    assert player.calculate_interest_gold() == 6  # 65 * 0.1 = 6 (cap 7)
    player.gold = 75
    assert player.calculate_interest_gold() == 7  # 75 * 0.1 = 7 (cap 7)

    # 2. Big Grab Bag (3 components + 2 gold + 1 Reforger)
    items_before = len(player.item_bench)
    reforgers_before = player.reforgers
    applied = mgr.apply_augment(player, "DA_18_BigGrabBag", pool=pool, set_data=set18_data)
    assert applied
    assert len(player.item_bench) == items_before + 3
    assert player.reforgers == reforgers_before + 1

    # 3. New Recruit (+1 Team Size + 1 Duplicator)
    extra_size_before = player.extra_team_size
    duplicators_before = player.duplicators
    applied = mgr.apply_augment(player, "DA_18_NewRecruit", pool=pool, set_data=set18_data)
    assert applied
    assert player.extra_team_size == extra_size_before + 1
    assert player.duplicators == duplicators_before + 1


def test_consumable_duplicator(player: Player, pool: ChampionPool):
    """Test that Champion Duplicator clones a 1-star copy of a unit."""
    unit = ChampionInstance(champion_id="TFT18_Ahri", cost=4, star_level=2)
    player.board[(0, 0)] = unit
    player.duplicators = 1

    success = player.use_duplicator(is_board=True, loc=(0, 0), pool=pool)
    assert success
    assert player.duplicators == 0
    # Should have a 1-star Ahri on bench
    bench_ahri = [u for u in player.bench if u is not None and u.champion_id == "TFT18_Ahri"]
    assert len(bench_ahri) == 1
    assert bench_ahri[0].star_level == 1


def test_consumable_remover(player: Player):
    """Test that Magnetic Remover strips items off a unit to the item bench."""
    unit = ChampionInstance(
        champion_id="TFT18_Akali",
        cost=1,
        star_level=2,
        items=["TFT_Item_InfinityEdge", "TFT_Item_Bloodthirster"],
    )
    player.board[(0, 1)] = unit
    player.removers = 1

    success = player.use_remover(is_board=True, loc=(0, 1))
    assert success
    assert player.removers == 0
    assert len(unit.items) == 0
    assert len(player.item_bench) == 2
    bench_ids = [it.item_id for it in player.item_bench]
    assert "TFT_Item_InfinityEdge" in bench_ids
    assert "TFT_Item_Bloodthirster" in bench_ids


def test_consumable_reforger(player: Player, set18_data: SetData):
    """Test that Reforger rerolls an item on the bench into another random one."""
    player.add_item("TFT_Item_BFSword")
    player.reforgers = 1

    success = player.use_reforger(item_bench_idx=0, rng=random.Random(123))
    assert success
    assert player.reforgers == 0
    assert len(player.item_bench) == 1
    assert player.item_bench[0].is_component
    assert player.item_bench[0].item_id != "TFT_Item_BFSword"


def test_elder_dragon_two_slots(player: Player):
    """Test that Elder Dragon occupies 2 team slots toward board capacity."""
    player.level = 4
    dragon = ChampionInstance(champion_id="TFT18_ElderDragon", cost=5, star_level=1)
    player.board[(0, 0)] = dragon
    assert player.board_unit_count == 2

    akali = ChampionInstance(champion_id="TFT18_Akali", cost=1, star_level=1)
    player.board[(0, 1)] = akali
    assert player.board_unit_count == 3


def test_coven_and_draven_combat_loot(set18_data: SetData, pool: ChampionPool):
    """Test Coven essence accumulation and Draven bounty reward resolution."""
    sm = StageManager(set18_data)
    p0 = Player(player_id=0, set_data=set18_data, name="Winner")
    p1 = Player(player_id=1, set_data=set18_data, name="Loser")

    # Put Coven and Draven on Winner board
    p0.board[(0, 0)] = ChampionInstance(champion_id="TFT18_Camille", cost=1, star_level=2)
    p0.board[(0, 1)] = ChampionInstance(champion_id="TFT18_Elise", cost=2, star_level=2)
    p0.board[(0, 2)] = ChampionInstance(champion_id="TFT18_Cassiopeia", cost=3, star_level=2)
    p0.board[(0, 3)] = ChampionInstance(champion_id="TFT18_Draven", cost=5, star_level=1)

    # Put units on Loser board
    p1.board[(0, 0)] = ChampionInstance(champion_id="TFT18_Akali", cost=1, star_level=1)
    p1.board[(0, 1)] = ChampionInstance(champion_id="TFT18_Varus", cost=1, star_level=1)

    res = CombatResult(
        winner_id=0,
        loser_id=1,
        damage_dealt=10,
        win_prob_a=0.8,
        surviving_units=3,
    )

    gold_before = p0.gold
    sm.handle_combat_loot_and_traits([p0, p1], [res], pool=pool, rng=random.Random(42))

    # Coven essence and Draven bounties should have progressed
    assert p0.coven_essence > 0
    assert p0.draven_bounty_progress > 0

    # Simulate enough essence to trigger Coven cashout
    p0.coven_essence = 45
    sm.handle_combat_loot_and_traits([p0, p1], [res], pool=pool, rng=random.Random(42))
    assert p0.gold > gold_before
    assert p0.coven_essence == 0  # Consumed


def test_inferno_shop_ignition(set18_data: SetData, pool: ChampionPool):
    """Test that Inferno trait ignites shop slots to roll champions 1 tier higher."""
    sm = StageManager(set18_data)
    player = Player(player_id=0, set_data=set18_data, name="Inferno Player")
    # 3 Inferno units (Akali, Varus, Sett)
    player.board[(0, 0)] = ChampionInstance(champion_id="TFT18_Akali", cost=1, star_level=2)
    player.board[(0, 1)] = ChampionInstance(champion_id="TFT18_Varus", cost=1, star_level=2)
    player.board[(0, 2)] = ChampionInstance(champion_id="TFT18_Sett", cost=4, star_level=1)

    res = CombatResult(
        winner_id=0,
        loser_id=1,
        damage_dealt=6,
        win_prob_a=0.9,
        surviving_units=2,
    )
    sm.handle_combat_loot_and_traits([player, Player(1, set18_data)], [res], pool=pool)
    assert len(player.ignited_shop_slots) == 2


def test_full_set18_game_with_augments_and_traits(set18_data: SetData):
    """Run an entire 8-player TFTGame match to completion with Set 18 data and verify stability."""
    game = TFTGame(set_data=set18_data, seed=12345)
    max_rounds = 100
    rounds_run = 0

    while not game.is_over and rounds_run < max_rounds:
        game.execute_bot_turns()
        game.resolve_round_phase()
        rounds_run += 1

    assert game.is_over
    rankings = game.get_rankings()
    assert len(rankings) == 8
    # Winner must have 1st placement
    assert rankings[0][1].placement == 1
    # Check that alive or surviving players picked augments at 2-1, 3-2, 4-2
    for _, p in rankings:
        assert len(p.augments) >= 1
