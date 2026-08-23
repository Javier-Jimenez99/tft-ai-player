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


def _observation(stage: str, match_id: str = "game-uuid") -> RoundObservation:
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
        focal_tier="CHALLENGER I 1756 LP",
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