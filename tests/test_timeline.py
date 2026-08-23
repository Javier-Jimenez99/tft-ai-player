from __future__ import annotations

import json

import pytest

from tft_ai_player.dataset import TimelineValidationError, extract_pvp_rounds, parse_stage_data


def test_extract_pvp_rounds_excludes_pve_and_preserves_version_metadata() -> None:
    timeline = {
        "summoner_name": "Focal",
        "datetime": 1787486839194,
        "data_version": "1.0",
        "portal": "TFT_Portals_Champions_ChampionStart",
        "gep_internal": {"version_info": {"local_version": "311.2.2"}},
        "stage_data": json.dumps(
            [
                _snapshot(stage="2-1", round_name="PVE", outcome="victory"),
                _snapshot(stage="2-2", round_name="PVP", outcome="victory"),
            ]
        ),
    }

    observations = extract_pvp_rounds(
        timeline,
        match_id="LA2_123",
        tft_set="TFTSet17",
        game_version="16.16",
        focal_tier="CHALLENGER I 1756 LP",
        focal_rating_numeric=4556,
        avg_match_rating="GRANDMASTER I 937 LP",
        avg_match_rating_numeric=3737,
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.observation_id == "LA2_123:2-2:Focal"
    assert observation.label == 1
    assert observation.tft_set == "TFTSet17"
    assert observation.game_version == "16.16"
    assert observation.game_datetime == "2026-08-23T12:07:19Z"
    assert observation.focal_tier == "CHALLENGER I 1756 LP"
    assert observation.focal_rating_numeric == 4556
    assert observation.avg_match_rating == "GRANDMASTER I 937 LP"
    assert observation.avg_match_rating_numeric == 3737
    assert observation.game_client_version == "311.2.2"
    assert observation.timeline_schema_version == "1.0"
    assert observation.portal == "TFT_Portals_Champions_ChampionStart"
    assert observation.metatft_win_prob == 0.9

    assert observation.focal_player == "Focal"
    assert observation.focal_health == 90
    assert observation.focal_level == 4
    assert observation.focal_gold == 20
    assert observation.focal_augments == ["TFT17_Augment_BigGains"]
    assert observation.focal_unit_count == 1
    assert observation.focal_item_count == 1

    assert observation.opponent == "Opponent"
    assert observation.opponent_health == 88
    assert observation.opponent_level == 4
    assert observation.opponent_augments == []
    assert observation.opponent_unit_count == 1
    assert observation.opponent_item_count == 0

    assert observation.input_state["focal_board"][0] == {
        "unit": "TFT17_Aatrox",
        "tier": 2,
        "loc": "D1",
        "items": ["TFT_Item_WarmogsArmor"],
    }
    assert observation.input_state["opponent_board"][0] == {
        "unit": "TFT17_Veigar",
        "tier": 2,
        "loc": "A_1",
        "items": [],
    }

    assert "round" not in observation.input_state
    assert "battle_stats" not in observation.input_state
    assert "round_outcome" not in observation.input_state
    assert "winrate_info" not in observation.input_state


def test_extract_pvp_rounds_cleans_units_and_item_ids_fallback() -> None:
    timeline = {
        "summoner_name": "Focal",
        "data_version": "1.0",
        "stage_data": json.dumps(
            [
                {
                    "me": {"summoner_name": "Focal", "gold": 50, "xp": {"level": 7}},
                    "round_start_health": {
                        "player_status": {
                            "Focal": {"health": 75, "xp": 7},
                            "Opponent": {"health": 60, "xp": 7},
                        }
                    },
                    "match_info": {
                        "round_type": {"stage": "4-1", "name": "PVP", "type": "PVP"},
                        "opponent": {"name": "Opponent"},
                        "round_outcome": {"Focal": {"outcome": "defeat"}},
                    },
                    "matchup_boards": {
                        "player_board": [
                            {
                                "apiName": "TFT17_Belveth",
                                "name": "Bel'Veth",
                                "cost": 2,
                                "item_ids": ["TFT_Item_BFSword"],
                                "loc": "B7",
                                "tier": 2,
                            },
                            {
                                "unit": "TFT17_ShenProp",
                                "loc": "B4",
                                "tier": 1,
                            },
                        ],
                        "opponent_board": [
                            {
                                "unit": "TFT17_Jax",
                                "apiName": "TFT17_Jax",
                                "items": ["TFT17_Emblem_Vanguard"],
                                "item_ids": ["TFT17_Item_ShieldTankEmblemItem"],
                                "loc": "A_3",
                                "tier": 1,
                            }
                        ],
                    },
                }
            ]
        ),
    }

    observations = extract_pvp_rounds(
        timeline,
        match_id="TEST_1",
        tft_set="TFTSet17",
        game_version="16.16",
    )

    assert len(observations) == 1
    obs = observations[0]
    assert obs.focal_unit_count == 2
    assert obs.focal_item_count == 1
    assert obs.opponent_unit_count == 1
    assert obs.opponent_item_count == 1

    focal_board = obs.input_state["focal_board"]
    assert focal_board[0] == {
        "unit": "TFT17_Belveth",
        "tier": 2,
        "loc": "B7",
        "items": ["TFT_Item_BFSword"],
    }
    assert focal_board[1] == {
        "unit": "TFT17_ShenProp",
        "tier": 1,
        "loc": "B4",
        "items": [],
    }
    opp_board = obs.input_state["opponent_board"]
    assert opp_board[0] == {
        "unit": "TFT17_Jax",
        "tier": 1,
        "loc": "A_3",
        "items": ["TFT17_Emblem_Vanguard"],
    }


def test_parse_stage_data_rejects_a_non_array_payload() -> None:
    with pytest.raises(TimelineValidationError, match="must decode to a list"):
        parse_stage_data({"stage_data": "{}"})


def _snapshot(*, stage: str, round_name: str, outcome: str) -> dict[str, object]:
    return {
        "me": {"summoner_name": "Focal", "gold": "20", "xp": {"level": 4}},
        "augments": {"Focal": ["TFT17_Augment_BigGains"], "Opponent": []},
        "round_start_health": {
            "player_status": {
                "Focal": {"health": 90, "xp": 4, "rank": 0},
                "Opponent": {"health": 88, "xp": 4, "rank": 0},
            }
        },
        "match_info": {
            "round_type": {"stage": stage, "name": round_name, "type": round_name},
            "opponent": {"name": "Opponent"},
            "round_outcome": {"Focal": {"outcome": outcome}},
            "battle_stats": "must not enter the dataset",
        },
        "matchup_boards": {
            "player_board": [{"unit": "TFT17_Aatrox", "tier": 2, "loc": "D1", "items": ["TFT_Item_WarmogsArmor"]}],
            "opponent_board": [{"unit": "TFT17_Veigar", "tier": 2, "loc": "A_1", "items": []}],
        },
        "winrate_info": {"model_data": {"prediction": 0.9}},
    }