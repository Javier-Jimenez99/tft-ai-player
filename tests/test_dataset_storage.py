from __future__ import annotations

import csv

from tft_ai_player.dataset.models import RoundObservation
from tft_ai_player.dataset.storage import PlayerCsvWriter


def test_writer_stores_and_appends_games_in_player_csv(tmp_path) -> None:
    writer = PlayerCsvWriter(tmp_path)
    obs_game1 = [_observation(match_id="game-1", stage="2-2"), _observation(match_id="game-1", stage="2-5")]
    obs_game2 = [_observation(match_id="game-2", stage="3-2")]

    path1 = writer.write_player_game(
        obs_game1,
        collected_from_riot_id="Focal#LAS",
        collected_from_region="la2",
        match_id_ow="internal-match-id-1",
    )
    path2 = writer.write_player_game(
        obs_game2,
        collected_from_riot_id="Focal#LAS",
        collected_from_region="la2",
        match_id_ow="internal-match-id-2",
    )

    expected_path = tmp_path / "players" / "la2_Focal_LAS.csv"
    assert path1 == expected_path
    assert path2 == expected_path
    assert writer.existing_match_ids(region="la2", riot_id="Focal#LAS") == {"game-1", "game-2"}

    with expected_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 3
    assert [row["round_stage"] for row in rows] == ["2-2", "2-5", "3-2"]
    assert [row["match_id"] for row in rows] == ["game-1", "game-1", "game-2"]
    assert {row["collected_from_riot_id"] for row in rows} == {"Focal#LAS"}
    assert {row["portal"] for row in rows} == {"TFT_Portals_Champions_ChampionStart"}
    assert {row["game_datetime"] for row in rows} == {"2026-08-23T12:07:19Z"}
    assert {row["focal_tier"] for row in rows} == {"CHALLENGER I 1756 LP"}
    assert {row["focal_rating_numeric"] for row in rows} == {"4556"}
    assert {row["avg_match_rating"] for row in rows} == {"GRANDMASTER I 937 LP"}
    assert {row["avg_match_rating_numeric"] for row in rows} == {"3737"}
    assert {row["focal_health"] for row in rows} == {"100"}
    assert {row["focal_level"] for row in rows} == {"4"}
    assert {row["focal_gold"] for row in rows} == {"20"}
    assert {row["focal_augments"] for row in rows} == {"TFT17_Augment_BigGains"}
    assert {row["focal_unit_count"] for row in rows} == {"1"}
    assert {row["focal_item_count"] for row in rows} == {"1"}
    assert {row["opponent_health"] for row in rows} == {"88"}
    assert {row["opponent_level"] for row in rows} == {"4"}
    assert {row["opponent_unit_count"] for row in rows} == {"1"}
    assert {row["opponent_item_count"] for row in rows} == {"0"}
    assert {row["metatft_win_prob"] for row in rows} == {"0.85"}
    assert list(tmp_path.rglob("*.csv")) == [expected_path]


def test_writer_skips_games_without_valid_pvp_rounds(tmp_path) -> None:
    assert PlayerCsvWriter(tmp_path).write_player_game([], collected_from_riot_id="Focal#LAS", collected_from_region="la2") is None
    assert not (tmp_path / "players").exists()


def test_writer_blacklist_management(tmp_path) -> None:
    writer = PlayerCsvWriter(tmp_path)
    assert writer.load_blacklist() == set()
    assert not writer.is_blacklisted("game-1")

    writer.add_to_blacklist("game-1", reason="no valid PVP rounds")
    writer.add_to_blacklist("game-2", reason="validation error")
    writer.add_to_blacklist("game-1", reason="duplicate attempt")

    assert writer.is_blacklisted("game-1")
    assert writer.is_blacklisted("game-2")
    assert not writer.is_blacklisted("game-3")
    assert writer.load_blacklist() == {"game-1", "game-2"}

    # Verify file content
    content = writer.blacklist_path().read_text(encoding="utf-8")
    assert "game-1\t# no valid PVP rounds" in content
    assert "game-2\t# validation error" in content
    assert content.count("game-1") == 1


def test_writer_player_blacklist_management(tmp_path) -> None:
    writer = PlayerCsvWriter(tmp_path)
    assert writer.load_player_blacklist() == set()
    assert not writer.is_player_blacklisted("Player#KR1")

    writer.add_player_to_blacklist("Player#KR1", reason="5 consecutive games with no valid PVP rounds")
    writer.add_player_to_blacklist("Player2#NA1", reason="custom reason")
    writer.add_player_to_blacklist("Player#KR1", reason="duplicate attempt")

    assert writer.is_player_blacklisted("Player#KR1")
    assert writer.is_player_blacklisted("Player2#NA1")
    assert not writer.is_player_blacklisted("Player3#EUW")
    assert writer.load_player_blacklist() == {"Player#KR1", "Player2#NA1"}

    content = writer.player_blacklist_path().read_text(encoding="utf-8")
    assert "Player#KR1\t# 5 consecutive games with no valid PVP rounds" in content
    assert "Player2#NA1\t# custom reason" in content
    assert content.count("Player#KR1") == 1


def test_write_game_compatibility_alias(tmp_path) -> None:
    writer = PlayerCsvWriter(tmp_path)
    # Empty observations should safely return None without IndexError
    assert writer.write_game([]) is None

    # Non-empty observations should derive focal_player and write successfully
    obs = [_observation(stage="2-2")]
    path = writer.write_game(obs)
    assert path is not None
    assert path.exists()
    assert "unknown_Focal.csv" in path.name


def test_writer_tier_partitioned_storage(tmp_path) -> None:
    writer = PlayerCsvWriter(tmp_path, tier_partitioned=True)
    obs_gold = [_observation(stage="2-1", match_id="gold-match-1", focal_tier="GOLD II 50 LP")]
    obs_challenger = [_observation(stage="2-1", match_id="chal-match-1", focal_tier="CHALLENGER I 995 LP")]

    path_gold = writer.write_player_game(
        obs_gold,
        collected_from_riot_id="GoldPlayer#NA1",
        collected_from_region="na1",
        match_id_ow="gold-ow-1",
    )
    path_chal = writer.write_player_game(
        obs_challenger,
        collected_from_riot_id="ChallengerPlayer#NA1",
        collected_from_region="na1",
        match_id_ow="chal-ow-1",
    )

    assert path_gold == tmp_path / "tiers" / "gold" / "players" / "na1_GoldPlayer_NA1.csv"
    assert path_chal == tmp_path / "tiers" / "challenger" / "players" / "na1_ChallengerPlayer_NA1.csv"

    # Verify existing match IDs across tier directories
    all_ids = writer.all_existing_match_ids()
    assert "gold-match-1" in all_ids
    assert "chal-match-1" in all_ids

    with path_gold.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["tier_category"] == "GOLD"


def test_normalize_tier() -> None:
    from tft_ai_player.dataset.models import normalize_tier
    assert normalize_tier("CHALLENGER I 995 LP") == "CHALLENGER"
    assert normalize_tier("Grandmaster I 400 LP") == "GRANDMASTER"
    assert normalize_tier("Master I 0 LP") == "MASTER"
    assert normalize_tier("Diamond IV 20 LP") == "DIAMOND"
    assert normalize_tier("Emerald II 55 LP") == "EMERALD"
    assert normalize_tier("Platinum I 90 LP") == "PLATINUM"
    assert normalize_tier("Gold III 10 LP") == "GOLD"
    assert normalize_tier("Silver IV 0 LP") == "SILVER"
    assert normalize_tier("Bronze I 75 LP") == "BRONZE"
    assert normalize_tier("Iron II 20 LP") == "IRON"
    assert normalize_tier("Unranked") == "UNKNOWN"
    assert normalize_tier(None) == "UNKNOWN"
    assert normalize_tier("") == "UNKNOWN"


def test_existing_match_ids_by_tier(tmp_path) -> None:
    writer = PlayerCsvWriter(tmp_path, tier_partitioned=True)
    obs_silver = [_observation(stage="2-1", match_id="match-silv-1", focal_tier="SILVER II 50 LP")]
    obs_bronze = [_observation(stage="2-1", match_id="match-bronze-1", focal_tier="BRONZE I 10 LP")]
    writer.write_player_game(obs_silver, collected_from_riot_id="P1#EUW", collected_from_region="euw1", tier="SILVER")
    writer.write_player_game(obs_bronze, collected_from_riot_id="P2#EUW", collected_from_region="euw1", tier="BRONZE")

    by_tier = writer.existing_match_ids_by_tier()
    assert by_tier.get("SILVER") == {"match-silv-1"}
    assert by_tier.get("BRONZE") == {"match-bronze-1"}
    assert writer.all_existing_match_ids() == {"match-silv-1", "match-bronze-1"}


def _observation(stage: str, match_id: str = "game-uuid", focal_tier: str = "CHALLENGER I 1756 LP") -> RoundObservation:
    return RoundObservation(
        match_id=match_id,
        game_datetime="2026-08-23T12:07:19Z",
        round_stage=stage,
        round_type="PVP",
        tft_set="TFTSet17",
        game_version="16.16",
        game_client_version="311.2.2",
        timeline_schema_version="1.0",
        portal="TFT_Portals_Champions_ChampionStart",
        focal_player="Focal",
        focal_tier=focal_tier,
        focal_rating_numeric=4556,
        avg_match_rating="GRANDMASTER I 937 LP",
        avg_match_rating_numeric=3737,
        focal_health=100,
        focal_level=4,
        focal_gold=20,
        focal_augments=["TFT17_Augment_BigGains"],
        opponent="Opponent",
        opponent_health=88,
        opponent_level=4,
        opponent_augments=[],
        outcome="victory",
        input_state={
            "focal_board": [{"unit": "TFT17_Aatrox", "tier": 2, "loc": "D1", "items": ["TFT_Item_WarmogsArmor"]}],
            "opponent_board": [{"unit": "TFT17_Veigar", "tier": 2, "loc": "A_1", "items": []}],
        },
        metatft_win_prob=0.85,
    )