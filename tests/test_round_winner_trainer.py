"""Unit tests for RoundWinnerTrainer and RoundWinnerPredictor."""

import json
from pathlib import Path
import pandas as pd
import pytest

from tft_ai_player.round_winner import RoundWinnerPredictor, RoundWinnerTrainer


def _make_dummy_dataset() -> pd.DataFrame:
    rows = []
    for match_id in range(10):
        for stage_num in range(2, 6):
            focal_board = [
                {"unit": "TFT17_Jinx", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge"]},
                {"unit": "TFT17_Nasus", "tier": 2, "loc": "A1", "items": ["TFT_Item_WarmogsArmor"]},
            ]
            opponent_board = [
                {"unit": "TFT17_Aatrox", "tier": 1, "loc": "A1", "items": []},
            ]
            rows.append({
                "match_id": f"match_{match_id}",
                "round_stage": f"{stage_num}-2",
                "focal_player": "PlayerA",
                "opponent_player": "PlayerB",
                "focal_level": 7,
                "opponent_level": 7,
                "focal_health": 80,
                "opponent_health": 80,
                "focal_gold": 20,
                "opponent_gold": 20,
                "focal_augments": json.dumps(["TFT17_Augment_SniperCrest"]),
                "opponent_augments": json.dumps([]),
                "label": 1 if match_id % 2 == 0 else 0,
                "outcome": "win" if match_id % 2 == 0 else "loss",
                "metatft_win_prob": 0.85 if match_id % 2 == 0 else 0.15,
                "input_state_json": json.dumps({
                    "focal_board": focal_board,
                    "opponent_board": opponent_board,
                }),
            })
    return pd.DataFrame(rows)


def test_trainer_fit_evaluate_and_predict(tmp_path: Path) -> None:
    df = _make_dummy_dataset()
    trainer = RoundWinnerTrainer(min_champ_freq=1, min_item_freq=1, min_aug_freq=1)

    metadata = trainer.fit_and_evaluate(df, test_size=0.30, val_size=0.20, verbose=False)
    assert "evaluation_metrics" in metadata
    assert "stress_tests" in metadata
    assert len(metadata["stress_tests"]) >= 3

    saved_model = trainer.save(tmp_path)
    assert saved_model.exists()
    assert (tmp_path / "metadata.json").exists()

    predictor = RoundWinnerPredictor.load(saved_model)
    prob = predictor.predict_proba(
        focal_board=[{"unit": "TFT17_Jinx", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge"]}],
        opponent_board=[{"unit": "TFT17_Aatrox", "tier": 1, "loc": "A1", "items": []}],
    )
    assert 0.0 <= prob <= 1.0

    win_prob_a, dmg_a, dmg_b = predictor.predict_combat(
        focal_board=[{"unit": "TFT17_Jinx", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge"]}],
        opponent_board=[{"unit": "TFT17_Aatrox", "tier": 1, "loc": "A1", "items": []}],
        round_stage="4-2",
    )
    assert 0.0 <= win_prob_a <= 1.0
    assert isinstance(dmg_a, int) and dmg_a >= 1
    assert isinstance(dmg_b, int) and dmg_b >= 1
