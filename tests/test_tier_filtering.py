from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd
import pytest

from tft_ai_player.dataset.models import RoundObservation, normalize_tier
from tft_ai_player.dataset.storage import PlayerCsvWriter
from tft_ai_player.embeddings.cluster import load_curated_endgame_snapshots
from tft_ai_player.embeddings.dataset import TFTPretrainDataset
from tft_ai_player.embeddings.transition import TransitionTrajectoryDataset
from tft_ai_player.metatft import MetaTftClient
from tft_ai_player.round_winner.trainer import RoundWinnerTrainer


def _make_round_observation(
    match_id: str,
    focal_player: str,
    stage: str,
    tier: str,
    label: int = 1,
) -> RoundObservation:
    return RoundObservation(
        match_id=match_id,
        game_datetime="2026-09-01T12:00:00Z",
        round_stage=stage,
        round_type="PVP",
        tft_set="TFTSet17",
        game_version="16.16",
        game_client_version="1.0",
        timeline_schema_version="1.0",
        portal=None,
        focal_player=focal_player,
        focal_tier=tier,
        focal_rating_numeric=1000,
        avg_match_rating=tier,
        avg_match_rating_numeric=1000,
        focal_health=100,
        focal_level=6,
        focal_gold=30,
        focal_augments=[],
        opponent="Opponent",
        opponent_health=80,
        opponent_level=6,
        opponent_augments=[],
        outcome="victory" if label == 1 else "defeat",
        input_state={
            "focal_board": [
                {"unit": "TFT17_Aatrox", "tier": 2, "loc": "D1", "items": ["TFT_Item_WarmogsArmor"]}
            ],
            "opponent_board": [
                {"unit": "TFT17_Veigar", "tier": 2, "loc": "A_1", "items": []}
            ],
        },
        metatft_win_prob=0.75,
        tier_category=normalize_tier(tier),
    )


def _populate_mixed_tier_dataset(data_dir: Path) -> None:
    writer = PlayerCsvWriter(data_dir)
    # Challenger games
    chal_obs = [
        _make_round_observation("chal-m1", "ChalPlayer", "2-1", "CHALLENGER I 800 LP"),
        _make_round_observation("chal-m1", "ChalPlayer", "2-5", "CHALLENGER I 800 LP"),
        _make_round_observation("chal-m1", "ChalPlayer", "3-2", "CHALLENGER I 800 LP"),
        _make_round_observation("chal-m1", "ChalPlayer", "4-2", "CHALLENGER I 800 LP"),
        _make_round_observation("chal-m1", "ChalPlayer", "5-2", "CHALLENGER I 800 LP"),
        _make_round_observation("chal-m1", "ChalPlayer", "5-6", "CHALLENGER I 800 LP"),
        _make_round_observation("chal-m1", "ChalPlayer", "6-1", "CHALLENGER I 800 LP"),
    ]
    writer.write_player_game(chal_obs, collected_from_riot_id="ChalPlayer#NA1", collected_from_region="na1")

    # Gold games
    gold_obs = [
        _make_round_observation("gold-m1", "GoldPlayer", "2-1", "GOLD II 50 LP"),
        _make_round_observation("gold-m1", "GoldPlayer", "2-5", "GOLD II 50 LP"),
        _make_round_observation("gold-m1", "GoldPlayer", "3-2", "GOLD II 50 LP"),
        _make_round_observation("gold-m1", "GoldPlayer", "4-2", "GOLD II 50 LP"),
        _make_round_observation("gold-m1", "GoldPlayer", "5-2", "GOLD II 50 LP"),
        _make_round_observation("gold-m1", "GoldPlayer", "5-6", "GOLD II 50 LP"),
        _make_round_observation("gold-m1", "GoldPlayer", "6-1", "GOLD II 50 LP"),
    ]
    writer.write_player_game(gold_obs, collected_from_riot_id="GoldPlayer#NA1", collected_from_region="na1")

    # Iron games
    iron_obs = [
        _make_round_observation("iron-m1", "IronPlayer", "2-1", "IRON IV 0 LP"),
        _make_round_observation("iron-m1", "IronPlayer", "2-5", "IRON IV 0 LP"),
    ]
    writer.write_player_game(iron_obs, collected_from_riot_id="IronPlayer#NA1", collected_from_region="na1")


def test_round_winner_trainer_tier_filter(tmp_path: Path) -> None:
    _populate_mixed_tier_dataset(tmp_path)
    trainer = RoundWinnerTrainer(random_state=42)

    # 1. Default should only load Challenger data
    df_chal = trainer.load_dataset(tmp_path)
    assert len(df_chal) == 7
    assert df_chal["match_id"].unique().tolist() == ["chal-m1"]
    assert (df_chal["tier_category"] == "CHALLENGER").all()

    # 2. Explicit filter for Gold
    df_gold = trainer.load_dataset(tmp_path, allowed_tiers=["GOLD"])
    assert len(df_gold) == 7
    assert df_gold["match_id"].unique().tolist() == ["gold-m1"]

    # 3. None allowed_tiers loads everything
    df_all = trainer.load_dataset(tmp_path, allowed_tiers=None)
    assert len(df_all) == 16


def test_tft_pretrain_dataset_tier_filter(tmp_path: Path) -> None:
    _populate_mixed_tier_dataset(tmp_path)

    # Default: Challenger only
    ds_chal = TFTPretrainDataset(data_dir=tmp_path)
    assert len(ds_chal) > 0
    for mid in ds_chal.match_ids:
        assert mid == "chal-m1"

    # All data allowed
    ds_all = TFTPretrainDataset(data_dir=tmp_path, allowed_tiers=None)
    assert len(ds_all) > len(ds_chal)


def test_transition_trajectory_dataset_tier_filter(tmp_path: Path) -> None:
    _populate_mixed_tier_dataset(tmp_path)

    # Default: Challenger only
    ds_chal = TransitionTrajectoryDataset(data_dir=tmp_path)
    assert len(ds_chal) > 0
    for mid in ds_chal.match_ids:
        assert mid == "chal-m1"

    # Explicit Gold tier
    ds_gold = TransitionTrajectoryDataset(data_dir=tmp_path, allowed_tiers=["GOLD"])
    assert len(ds_gold) > 0
    for mid in ds_gold.match_ids:
        assert mid == "gold-m1"


def test_load_curated_endgame_snapshots_tier_filter(tmp_path: Path) -> None:
    _populate_mixed_tier_dataset(tmp_path)

    # Default Challenger filter
    boards_chal, _, _, _ = load_curated_endgame_snapshots(data_dir=tmp_path, min_stage=5)
    assert len(boards_chal) == 3
    assert all(b.match_id == "chal-m1" for b in boards_chal)

    # Gold filter
    boards_gold, _, _, _ = load_curated_endgame_snapshots(data_dir=tmp_path, min_stage=5, allowed_tiers=["GOLD"])
    assert len(boards_gold) == 3
    assert all(b.match_id == "gold-m1" for b in boards_gold)


def test_metatft_extract_lobby_participants_and_tier() -> None:
    mock_timeline = {
        "board_players": [
            {
                "board": [
                    {"summoner": "Player1", "tag_line": "NA1"},
                    {"summoner": "Player2", "tag_line": "FREAK"},
                ]
            },
            {
                "board": [
                    {"summoner": "Player1", "tag_line": "NA1"},
                    {"summoner": "Player3", "tag_line": "KR1"},
                ]
            },
        ]
    }
    participants = MetaTftClient.extract_lobby_participants(mock_timeline)
    assert set(participants) == {("Player1", "NA1"), ("Player2", "FREAK"), ("Player3", "KR1")}

    mock_profile = {
        "ranked": {
            "rating_text": "CHALLENGER I 995 LP",
            "rating_numeric": 3795,
        }
    }
    assert MetaTftClient.extract_player_tier(mock_profile) == "CHALLENGER I 995 LP"
