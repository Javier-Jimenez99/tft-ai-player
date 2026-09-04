from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from tft_ai_player.cli import main
from tft_ai_player.dataset.player_graph import GameManifestEntry, LobbyEdge, PlayerGraph, PlayerNode


def test_player_graph_add_and_priority() -> None:
    graph = PlayerGraph()

    # Add 5 Platinum players (1 app user, 4 non-app)
    for i in range(4):
        graph.add_player(
            riot_id=f"PlatNonApp{i}#NA1",
            region="na1",
            game_name=f"PlatNonApp{i}",
            tag_line="NA1",
            tier="PLATINUM",
            rank_text="PLATINUM IV 50 LP",
            app_matches=0,
            scanned=True,
        )
    p_plat_app = graph.add_player(
        riot_id="PlatApp#NA1",
        region="na1",
        game_name="PlatApp",
        tag_line="NA1",
        tier="PLATINUM",
        rank_text="PLATINUM I 10 LP",
        app_matches=25,
        scanned=False,
    )

    # Add 1 Gold app user (rare tier)
    p_gold_app = graph.add_player(
        riot_id="GoldApp#NA1",
        region="na1",
        game_name="GoldApp",
        tag_line="NA1",
        tier="GOLD",
        rank_text="GOLD I 20 LP",
        app_matches=10,
        scanned=False,
    )

    # Priority should pick GoldApp because GOLD is rarer in the graph than PLATINUM!
    next_p = graph.get_next_priority_player()
    assert next_p is not None
    assert next_p.riot_id == "GoldApp#NA1"

    # Mark GoldApp as scanned
    next_p.scanned = True

    # Next priority should be PlatApp (because it's an app user with matches)
    next_p2 = graph.get_next_priority_player()
    assert next_p2 is not None
    assert next_p2.riot_id == "PlatApp#NA1"


def test_player_graph_csv_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        graph = PlayerGraph()

        graph.add_player(
            riot_id="Javi#401",
            region="na1",
            game_name="Javi",
            tag_line="401",
            tier="PLATINUM",
            rank_text="PLATINUM III 85 LP",
            app_matches=42,
            depth=0,
            is_app_user=True,
            scanned=True,
        )
        graph.add_player(
            riot_id="Protos#Colin",
            region="na1",
            game_name="Protos",
            tag_line="Colin",
            tier="PLATINUM",
            rank_text="PLATINUM I 0 LP",
            app_matches=62,
            depth=1,
            is_app_user=True,
            scanned=False,
            discovered_from="match-uuid-1",
        )

        graph.add_game(
            match_uuid="match-uuid-1",
            timeline_url="https://matches3.metatft.com/match-uuid-1.json",
            focal_player_riot_id="Javi#401",
            tier="PLATINUM",
            avg_rating="PLATINUM III 26 LP",
            tft_set="TFTSet18",
        )

        graph.add_edge("Javi#401", "Protos#Colin", "match-uuid-1")

        # Save to CSV
        graph.save_csv(tmp_path)

        assert (tmp_path / "players.csv").exists()
        assert (tmp_path / "games.csv").exists()
        assert (tmp_path / "edges.csv").exists()

        # Load into new graph
        graph2 = PlayerGraph()
        graph2.load_csv(tmp_path)

        assert len(graph2.nodes) == 2
        assert len(graph2.games) == 1
        assert len(graph2.edges) == 1

        javi = graph2.nodes["Javi#401"]
        assert javi.rank_text == "PLATINUM III 85 LP"
        assert javi.is_app_user is True
        assert javi.scanned is True
        assert javi.app_matches == 42

        protos = graph2.nodes["Protos#Colin"]
        assert protos.discovered_from == "match-uuid-1"
        assert protos.scanned is False

        game = graph2.games["match-uuid-1"]
        assert game.timeline_url == "https://matches3.metatft.com/match-uuid-1.json"
        assert game.tier == "PLATINUM"


def test_tier_cap_skips_full_league_and_continues_others() -> None:
    """When a specific league reaches max_players_per_tier, the process should NOT stop;

    it should stop picking players from that full league and continue scanning players from other leagues.
    """
    graph = PlayerGraph()

    # Add 2 scanned Challenger players
    for i in range(2):
        graph.add_player(
            riot_id=f"ChalScanned{i}#NA1",
            region="na1",
            game_name=f"ChalScanned{i}",
            tag_line="NA1",
            tier="CHALLENGER",
            rank_text="CHALLENGER I 1000 LP",
            app_matches=10,
            is_app_user=True,
            scanned=True,
        )

    # Add 1 unscanned Challenger player
    graph.add_player(
        riot_id="ChalUnscanned#NA1",
        region="na1",
        game_name="ChalUnscanned",
        tag_line="NA1",
        tier="CHALLENGER",
        rank_text="CHALLENGER I 900 LP",
        app_matches=5,
        scanned=False,
    )

    # Add 1 unscanned Gold player
    graph.add_player(
        riot_id="GoldUnscanned#NA1",
        region="na1",
        game_name="GoldUnscanned",
        tag_line="NA1",
        tier="GOLD",
        rank_text="GOLD II 50 LP",
        app_matches=5,
        scanned=False,
    )

    # Add 1 unscanned Iron player
    graph.add_player(
        riot_id="IronUnscanned#NA1",
        region="na1",
        game_name="IronUnscanned",
        tag_line="NA1",
        tier="IRON",
        rank_text="IRON I 10 LP",
        app_matches=5,
        scanned=False,
    )

    # With max_players_per_tier = 2:
    # Challenger is already at 2, so it MUST NOT pick ChalUnscanned.
    # It should pick Iron or Gold next!
    next_p = graph.get_next_priority_player(max_players_per_tier=2)
    assert next_p is not None
    assert next_p.riot_id in ("IronUnscanned#NA1", "GoldUnscanned#NA1")
    assert next_p.tier != "CHALLENGER"

    # Mark it scanned
    next_p.scanned = True
    next_p.is_app_user = True

    # Next player should still NOT be Challenger
    next_p2 = graph.get_next_priority_player(max_players_per_tier=2)
    assert next_p2 is not None
    assert next_p2.tier != "CHALLENGER"
    assert next_p2.riot_id in ("IronUnscanned#NA1", "GoldUnscanned#NA1")

    # Mark next_p2 as scanned
    next_p2.scanned = True
    next_p2.is_app_user = True

    # Now both Iron and Gold have 1 (< 2 cap), but no unscanned Iron or Gold remain.
    # ChalUnscanned is the only unscanned player left, but Challenger is full (2 >= 2).
    # So get_next_priority_player should return None.
    assert graph.get_next_priority_player(max_players_per_tier=2) is None


def test_max_games_per_tier_priority_and_capping() -> None:
    """When a tier reaches max_games_per_tier (e.g. 1000 games), it should stop picking players

    from that tier and prioritize tiers with fewer games.
    """
    graph = PlayerGraph()

    # Add 1000 games in Emerald
    for i in range(1000):
        graph.add_game(
            match_uuid=f"emer-match-{i}",
            timeline_url=f"https://matches3.metatft.com/emer-{i}.json",
            focal_player_riot_id="EmerApp#NA1",
            tier="EMERALD",
        )

    # Add 50 games in Gold
    for i in range(50):
        graph.add_game(
            match_uuid=f"gold-match-{i}",
            timeline_url=f"https://matches3.metatft.com/gold-{i}.json",
            focal_player_riot_id="GoldApp#NA1",
            tier="GOLD",
        )

    # Add unscanned Emerald player
    graph.add_player(
        riot_id="EmerUnscanned#NA1",
        region="na1",
        game_name="EmerUnscanned",
        tag_line="NA1",
        tier="EMERALD",
        scanned=False,
    )

    # Add unscanned Gold player
    graph.add_player(
        riot_id="GoldUnscanned#NA1",
        region="na1",
        game_name="GoldUnscanned",
        tag_line="NA1",
        tier="GOLD",
        scanned=False,
    )

    # With max_games_per_tier = 1000:
    # Emerald has 1000 games (>= 1000), so EmerUnscanned must be skipped!
    # Gold has only 50 games (< 1000), so GoldUnscanned must be chosen.
    next_p = graph.get_next_priority_player(max_games_per_tier=1000)
    assert next_p is not None
    assert next_p.riot_id == "GoldUnscanned#NA1"
    assert next_p.tier == "GOLD"

    # Mark GoldUnscanned scanned
    next_p.scanned = True

    # Now no more unscanned Gold players, and Emerald is capped at 1000 games.
    # Should return None.
    assert graph.get_next_priority_player(max_games_per_tier=1000) is None


