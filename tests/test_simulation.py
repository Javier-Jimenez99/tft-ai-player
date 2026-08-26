"""Unit tests for set-agnostic TFT simulation engine and Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from tft_ai_player.simulation import (
    TOTAL_DISCRETE_ACTIONS,
    ChampionInstance,
    ChampionPool,
    HeuristicCombatResolver,
    ItemInstance,
    MatchmakingEngine,
    Player,
    SetData,
    StandardTempoBot,
    TFTEnv,
    TFTGame,
    get_action_mask,
    get_default_set17_data,
)


@pytest.fixture
def set_data() -> SetData:
    """Fixture providing standard Set 17 configuration."""
    return get_default_set17_data()


@pytest.fixture
def pool(set_data: SetData) -> ChampionPool:
    """Fixture providing shared champion pool."""
    return ChampionPool(set_data)


def test_set_data_initialization(set_data: SetData) -> None:
    """Verify SetData loads champions, traits, items, recipes, and rules correctly."""
    assert set_data.set_name == "TFTSet17"
    assert len(set_data.champions) > 40
    assert len(set_data.traits) >= 15
    assert len(set_data.items) >= 40
    assert len(set_data.recipes) >= 30

    # Verify recipe lookup
    gunblade = set_data.get_recipe_result("TFT_Item_BFSword", "TFT_Item_NeedlesslyLargeRod")
    assert gunblade == "TFT_Item_HextechGunblade"

    infinity_edge = set_data.get_recipe_result("TFT_Item_BFSword", "TFT_Item_SparringGloves")
    assert infinity_edge == "TFT_Item_InfinityEdge"


def test_champion_pool_mechanics(set_data: SetData, pool: ChampionPool) -> None:
    """Verify drawing and returning champion copies from the shared pool."""
    initial_count = pool.counts["TFT17_Aatrox"]
    assert initial_count == 30  # 1-cost size

    drawn_c = pool.draw_champion(level=1)
    assert drawn_c is not None
    assert pool.counts[drawn_c] == (initial_count - 1 if drawn_c == "TFT17_Aatrox" else pool.counts[drawn_c])

    # Return 1-star copy
    pool.return_champion("TFT17_Aatrox", star_level=1)
    # Return 2-star copy (should add 3 copies)
    count_before = pool.counts["TFT17_Aatrox"]
    pool.return_champion("TFT17_Aatrox", star_level=2)
    assert pool.counts["TFT17_Aatrox"] == count_before + 3


def test_player_economy(set_data: SetData) -> None:
    """Verify interest calculations, modern streak gold, and leveling XP mechanics."""
    player = Player(0, set_data)
    player.gold = 35
    assert player.calculate_interest_gold() == 3

    player.gold = 58
    assert player.calculate_interest_gold() == 5  # Max interest cap (50g)

    # Modern Streak thresholds: 0-2: 0g, 3-4: 1g, 5: 2g, 6+: 3g
    player.streak = 2
    assert player.calculate_streak_gold() == 0
    player.streak = 3
    assert player.calculate_streak_gold() == 1
    player.streak = 4
    assert player.calculate_streak_gold() == 1
    player.streak = 5
    assert player.calculate_streak_gold() == 2
    player.streak = 6
    assert player.calculate_streak_gold() == 3
    player.streak = -3
    assert player.calculate_streak_gold() == 1

    # Leveling XP
    player.gold = 20
    assert player.level == 1
    player.buy_exp()  # +4 XP (costs 4g). 1->2 needs 2 XP, 2->3 needs 2 XP.
    assert player.level == 3
    assert player.exp == 0
    assert player.gold == 16


def test_pve_loot_distribution(set_data: SetData) -> None:
    """Verify PvE monster rounds drop balanced, stage-appropriate items and gold."""
    from tft_ai_player.simulation.stage_manager import StageManager

    stage_mgr = StageManager(set_data)
    players = [Player(i, set_data) for i in range(8)]

    # Stage 1 Minions
    stage_mgr.stage = 1
    stage_mgr.round_in_stage = 2
    stage_mgr.handle_pve_loot(players)
    # Each player should have either a component or starting gold
    for p in players:
        assert len(p.item_bench) in (0, 1)

    # Stage 5-7 Rift Boss -> Should drop completed item
    stage_mgr.stage = 5
    stage_mgr.round_in_stage = 7
    stage_mgr.handle_pve_loot(players)
    for p in players:
        assert len(p.item_bench) >= 1
        assert p.gold >= 3


def test_cascading_star_ups(set_data: SetData, pool: ChampionPool) -> None:
    """Verify 3-in-1 cascading star up logic from 1-star to 2-star and 3-star."""
    player = Player(0, set_data)

    # Add 1st copy
    u1 = ChampionInstance(champion_id="TFT17_Aatrox", cost=1, star_level=1, items=["TFT_Item_BFSword"])
    player.add_champion_to_bench(u1, pool)
    assert sum(1 for u in player.bench if u is not None) == 1
    assert player.bench[0].star_level == 1

    # Add 2nd copy
    u2 = ChampionInstance(champion_id="TFT17_Aatrox", cost=1, star_level=1)
    player.add_champion_to_bench(u2, pool)
    assert sum(1 for u in player.bench if u is not None) == 2

    # Add 3rd copy -> Should merge into 1 2-star!
    u3 = ChampionInstance(champion_id="TFT17_Aatrox", cost=1, star_level=1, items=["TFT_Item_ChainVest"])
    player.add_champion_to_bench(u3, pool)

    units = [u for u in player.bench if u is not None]
    assert len(units) == 1
    assert units[0].star_level == 2
    # Combined items should be preserved
    assert set(units[0].items) == {"TFT_Item_BFSword", "TFT_Item_ChainVest"}


def test_item_combining_and_equipping(set_data: SetData) -> None:
    """Verify component combining directly on bench and equipping onto champions."""
    player = Player(0, set_data)
    player.add_item("TFT_Item_BFSword")
    player.add_item("TFT_Item_NeedlesslyLargeRod")
    assert len(player.item_bench) == 2

    # Combine on item bench
    success = player.combine_items_on_bench(0, 1, set_data)
    assert success is True
    assert len(player.item_bench) == 1
    assert player.item_bench[0].item_id == "TFT_Item_HextechGunblade"

    # Add unit to board
    unit = ChampionInstance(champion_id="TFT17_Karma", cost=4, star_level=2)
    player.board[(0, 0)] = unit

    # Equip completed item to unit
    equip_res = player.equip_item(0, is_board=True, target_loc=(0, 0))
    assert equip_res is True
    assert len(player.item_bench) == 0
    assert unit.items == ["TFT_Item_HextechGunblade"]


def test_matchmaking_pairings(set_data: SetData) -> None:
    """Verify PvP matchmaking handles 8 players and odd player ghost boards."""
    engine = MatchmakingEngine()
    players = [Player(i, set_data) for i in range(8)]

    # Even number of players (8)
    pairings_8 = engine.generate_pairings(players)
    assert len(pairings_8) == 4
    assert all(not m.is_ghost_b for m in pairings_8)

    # Odd number of players (7)
    players[7].alive = False
    pairings_7 = engine.generate_pairings(players)
    assert len(pairings_7) == 4
    ghost_count = sum(1 for m in pairings_7 if m.is_ghost_b)
    assert ghost_count == 1


def test_heuristic_combat_resolution(set_data: SetData) -> None:
    """Verify heuristic combat correctly evaluates board advantage."""
    resolver = HeuristicCombatResolver(stochastic=False)
    p_strong = Player(0, set_data)
    p_weak = Player(1, set_data)

    # Strong player: 2-star 4-cost with Hextech Gunblade
    p_strong.board[(0, 0)] = ChampionInstance(
        champion_id="TFT17_Karma",
        cost=4,
        star_level=2,
        items=["TFT_Item_HextechGunblade"],
    )

    # Weak player: 1-star 1-cost with no items
    p_weak.board[(0, 0)] = ChampionInstance(
        champion_id="TFT17_Aatrox",
        cost=1,
        star_level=1,
        items=[],
    )

    res = resolver.resolve(
        player_a=p_strong,
        player_b=p_weak,
        is_ghost_b=False,
        stage=3,
        stage_str="3-2",
        set_data=set_data,
    )
    assert res.winner_id == p_strong.player_id
    assert res.loser_id == p_weak.player_id
    assert res.win_prob_a > 0.8
    assert res.damage_dealt >= 6  # Stage 3 base damage (6) + surviving


def test_full_game_simulation() -> None:
    """Run an entire 8-player TFTGame simulation until a winner is crowned."""
    game = TFTGame(seed=42)
    assert game.is_over is False

    max_rounds = 100
    rounds_run = 0
    while not game.is_over and rounds_run < max_rounds:
        # Run bot turns for all players
        for p in game.players:
            if p.alive:
                bot = StandardTempoBot()
                rinfo = game.stage_manager.get_current_round_info()
                bot.take_turn(p, game.pool, game.set_data, rinfo.stage, rinfo.round_in_stage)

        game.resolve_round_phase()
        rounds_run += 1

        # Strict invariant: No player may have more units on board than max_board_units (level + bonuses)
        for p in game.players:
            if p.alive:
                assert p.board_unit_count <= p.max_board_units, (
                    f"Player {p.player_id} has {p.board_unit_count} units on board at level {p.level} (max: {p.max_board_units})"
                )

    assert game.is_over is True
    rankings = game.get_rankings()
    assert len(rankings) == 8
    # 1st place should be assigned
    assert rankings[0][0] == 1
    assert rankings[0][1].alive is True


def test_board_capacity_strict_limit(set_data: SetData, pool: ChampionPool) -> None:
    """Verify units bought when bench is full do not leak onto the board beyond capacity."""
    player = Player(0, set_data)
    player.level = 4
    player.gold = 50

    # Fill bench with 9 units (including two 1-star Aatrox)
    player.bench[0] = ChampionInstance(champion_id="TFT17_Aatrox", cost=1, star_level=1)
    player.bench[1] = ChampionInstance(champion_id="TFT17_Aatrox", cost=1, star_level=1)
    for i in range(2, 9):
        player.bench[i] = ChampionInstance(champion_id="TFT17_Poppy", cost=1, star_level=1)

    assert player.free_bench_slots == 0
    assert player.board_unit_count == 0

    # Buy a 3rd Aatrox -> Bench was full, so it should merge into 2-star on bench and NOT place on board!
    aatrox_3 = ChampionInstance(champion_id="TFT17_Aatrox", cost=1, star_level=1)
    success = player.add_champion_to_bench(aatrox_3, pool)
    assert success is True

    # Board must remain 0 units
    assert player.board_unit_count == 0
    # Bench slot should now be freed (was 9 units, merged 2 into 1)
    assert player.free_bench_slots == 1
    assert player.bench[0].star_level == 2


def test_team_size_expanding_items(set_data: SetData) -> None:
    """Verify Tactician's Crown, Shield, Cape properly expand max board capacity."""
    player = Player(0, set_data)
    player.level = 6
    assert player.max_board_units == 6

    # 1. Tactician's Crown in item bench
    crown_item = ItemInstance(item_id="TFT_Item_TacticiansCrown", name="Tactician's Crown")
    player.item_bench.append(crown_item)
    assert player.max_board_units == 7  # 6 + 1

    # 2. Tactician's Shield equipped on a fielded board unit
    unit = ChampionInstance(champion_id="TFT17_Aatrox", cost=1, star_level=1, items=["TFT_Item_TacticiansShield"])
    player.board[(0, 0)] = unit
    assert player.max_board_units == 8  # 6 + 1 + 1

    # 3. Tactician's Cape equipped on a bench unit
    bench_unit = ChampionInstance(champion_id="TFT17_Poppy", cost=1, star_level=1, items=["TFT_Item_TacticiansCape"])
    player.bench[0] = bench_unit
    assert player.max_board_units == 9  # 6 + 1 + 1 + 1

    # Remove all extra items -> Should drop back to base level
    player.item_bench.clear()
    player.board.clear()
    player.bench[0] = None
    assert player.max_board_units == 6


def test_gymnasium_env_compliance() -> None:
    """Verify TFTEnv standard Gymnasium compliance with observation shapes, step, reset, and action mask."""
    env = gym.make("TFT-v0")
    obs, info = env.reset(seed=123)

    assert isinstance(obs, dict)
    assert "player_stats" in obs
    assert "board" in obs
    assert "action_mask" in obs
    assert obs["action_mask"].shape == (TOTAL_DISCRETE_ACTIONS,)
    assert obs["action_mask"][0] == True  # PASS is always valid

    # Step PASS action
    next_obs, reward, terminated, truncated, next_info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "stage" in next_info

    # Test flat observation mode
    flat_env = TFTEnv(use_flat_obs=True)
    flat_obs, flat_info = flat_env.reset(seed=123)
    assert isinstance(flat_obs, np.ndarray)
    assert flat_obs.ndim == 1
    assert len(flat_obs) == flat_env.encoder.flat_observation_dim


def test_stage_aware_carousel_draft(set_data: SetData, pool: ChampionPool) -> None:
    """Verify carousel yields stage-appropriate champion costs (no legendaries in stage 1)."""
    from tft_ai_player.simulation.stage_manager import StageManager

    stage_mgr = StageManager(set_data)
    players = [Player(i, set_data) for i in range(8)]

    # Stage 1-1 Carousel
    stage_mgr.stage = 1
    stage_mgr.round_in_stage = 1
    events_1 = stage_mgr.handle_carousel(players, pool)

    assert len(events_1) == 8
    for ev in events_1:
        assert ev["cost"] in (1, 2)  # Stage 1 carousel MUST be 1-cost or 2-cost only!
        assert ev["item_id"].startswith("TFT_Item_")
        p = players[ev["player_id"]]
        # Unit was added and holds the item
        owned_units = p.get_all_units()
        assert any(u.champion_id == ev["champion_id"] and ev["item_id"] in u.items for u in owned_units)

    # Stage 5-4 Carousel
    stage_mgr.stage = 5
    stage_mgr.round_in_stage = 4
    # Set different healths to verify lowest HP draft priority
    players[0].health = 10
    players[1].health = 80
    events_5 = stage_mgr.handle_carousel(players, pool)
    assert len(events_5) == 8
    # First draft pick must be Player 0 (lowest health = 10)
    assert events_5[0]["player_id"] == 0
    # Costs in stage 5 should be 4 or 5-cost
    for ev in events_5:
        assert ev["cost"] in (4, 5)

