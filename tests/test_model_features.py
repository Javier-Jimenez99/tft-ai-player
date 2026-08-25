"""Unit tests for TFT board feature extraction, pipelines, and evaluation metrics."""

import json
import numpy as np
import pandas as pd
import pytest

from tft_ai_player.round_winner import (
    TFTBoardFeatureExtractor,
    compute_brier_metrics,
    compute_calibration_table,
    compute_classification_metrics,
    evaluate_probabilistic_model,
)


@pytest.fixture
def sample_rounds_df() -> pd.DataFrame:
    """Provide a minimal dataframe simulating round observations."""
    state_1 = {
        "focal_board": [
            {"unit": "TFT17_Jinx", "tier": 2, "items": ["TFT_Item_Guinsoo"], "loc": "D1"},
            {"unit": "TFT17_Nasus", "tier": 2, "items": ["TFT_Item_FrozenHeart"], "loc": "A1"},
        ],
        "opponent_board": [
            {"unit": "TFT17_Diana", "tier": 1, "items": [], "loc": "A2"},
            {"unit": "TFT17_Jinx", "tier": 1, "items": [], "loc": "D2"},
        ],
    }

    state_2 = {
        "focal_board": [
            {"unit": "TFT17_Diana", "tier": 1, "items": [], "loc": "A1"},
        ],
        "opponent_board": [
            {"unit": "TFT17_Jinx", "tier": 3, "items": ["TFT_Item_InfinityEdge", "TFT_Item_Guinsoo", "TFT_Item_GiantSlayer"], "loc": "D1"},
            {"unit": "TFT17_Nasus", "tier": 2, "items": ["TFT_Item_Gargoyle"], "loc": "A1"},
        ],
    }

    return pd.DataFrame(
        [
            {
                "match_id": "match_001",
                "round_stage": "2-1",
                "focal_level": 4,
                "opponent_level": 3,
                "focal_health": 100,
                "opponent_health": 90,
                "focal_gold": 10,
                "focal_augments": "Augment_A,Augment_B",
                "opponent_augments": "Augment_C",
                "label": 1,
                "metatft_win_prob": 0.85,
                "input_state_json": json.dumps(state_1),
            },
            {
                "match_id": "match_001",
                "round_stage": "3-2",
                "focal_level": 5,
                "opponent_level": 6,
                "focal_health": 80,
                "opponent_health": 95,
                "focal_gold": 25,
                "focal_augments": "Augment_A",
                "opponent_augments": "Augment_B,Augment_D",
                "label": 0,
                "metatft_win_prob": 0.12,
                "input_state_json": json.dumps(state_2),
            },
        ]
    )


def test_feature_extractor_fit_transform(sample_rounds_df: pd.DataFrame) -> None:
    extractor = TFTBoardFeatureExtractor(
        min_champ_freq=1,
        min_item_freq=1,
        min_aug_freq=1,
    )
    features = extractor.fit_transform(sample_rounds_df)
    names = extractor.get_feature_names_out()

    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 2
    assert features.shape[1] == len(names)
    assert len(extractor.champions_vocab_) > 0
    assert len(extractor.items_vocab_) > 0

    # Check Jinx differential in row 0: focal tier 2 - opponent tier 1 = +1
    jinx_idx = list(names).index("champ_diff__TFT17_Jinx")
    assert features[0, jinx_idx] == 1.0

    # In row 1: focal tier 0 - opponent tier 3 = -3
    assert features[1, jinx_idx] == -3.0

    # Check placement features: row 0 focal has Nasus in A1 (frontline) and Jinx in D1 (backline)
    front_idx = list(names).index("focal_front_units")
    back_idx = list(names).index("focal_back_units")
    assert features[0, front_idx] == 1.0
    assert features[0, back_idx] == 1.0

    # Check trait features: row 0 focal has Jinx (Sniper) and Nasus (Vanguard)
    sniper_diff_idx = list(names).index("diff_trait_count__Sniper")
    assert features[0, sniper_diff_idx] == 0.0  # 1 on focal, 1 on opponent (Jinx vs Jinx)



def test_brier_metrics_computation() -> None:
    y_true = np.array([1, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2])

    res = compute_brier_metrics(y_true, y_prob)
    assert "brier_score" in res
    assert "brier_skill_score" in res
    assert 0.0 <= res["brier_score"] <= 0.25
    assert res["brier_skill_score"] > 0.0


def test_calibration_table_computation() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.8, 0.9, 0.1, 0.2])

    df_cal, ece, mce = compute_calibration_table(y_true, y_prob, n_bins=5)
    assert isinstance(df_cal, pd.DataFrame)
    assert len(df_cal) == 5
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0


def test_classification_metrics() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.85, 0.2])

    res = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert res["accuracy"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["f1"] == 1.0


def test_evaluate_probabilistic_model_bundle() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.85, 0.15, 0.75, 0.3, 0.9])
    y_ref = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    res = evaluate_probabilistic_model(
        y_true,
        y_prob,
        model_name="TestModel",
        y_ref_prob=y_ref,
    )
    assert res["model"] == "TestModel"
    assert res["brier_score"] < 0.15
    assert res["brier_skill_score"] > 0.0
