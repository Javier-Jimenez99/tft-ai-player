"""Unit and property tests for TFT Elo Regressor and Solution Space OOD Detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tft_ai_player.elo_predictor.dataset import FullMatchEloDataset, parse_elo_from_record
from tft_ai_player.elo_predictor.features import (
    FEATURE_NAMES,
    extract_game_features,
    extract_neural_metrics,
    parse_stage_round_index,
)
from tft_ai_player.elo_predictor.model import (
    MatchEloRegressor,
    SolutionSpaceOODDetector,
    elo_to_tier_name,
)
from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.board_evaluator.model import BoardQualityNet


def test_elo_to_tier_mapping() -> None:
    """Verify continuous Elo to competitive tier mapping across all ranked thresholds."""
    cases = [
        (4300.0, "CHALLENGER"),
        (3800.0, "CHALLENGER"),
        (3500.0, "GRANDMASTER"),
        (3300.0, "GRANDMASTER"),
        (2900.0, "MASTER"),
        (2800.0, "MASTER"),
        (2500.0, "DIAMOND"),
        (2400.0, "DIAMOND"),
        (2100.0, "EMERALD"),
        (2000.0, "EMERALD"),
        (1700.0, "PLATINUM"),
        (1600.0, "PLATINUM"),
        (1300.0, "GOLD"),
        (1200.0, "GOLD"),
        (900.0, "SILVER"),
        (800.0, "SILVER"),
        (500.0, "BRONZE"),
        (400.0, "BRONZE"),
        (200.0, "IRON"),
        (0.0, "IRON"),
    ]
    for elo, expected_tier in cases:
        assert elo_to_tier_name(elo) == expected_tier, f"Failed for elo={elo}: got {elo_to_tier_name(elo)}"


def test_parse_elo_from_record() -> None:
    """Verify tier string and LP parsing into continuous Elo rating."""
    assert parse_elo_from_record("CHALLENGER 650 LP", None) == 4200.0 + 650.0
    assert parse_elo_from_record("MASTER 120 LP", None) == 3000.0 + 120.0
    assert parse_elo_from_record("DIAMOND I 50 LP", None) == 2600.0 - 150.0 + 300.0 + 50.0
    assert parse_elo_from_record("GOLD IV 20 LP", None) == 1400.0 - 150.0 + 0.0 + 20.0
    assert parse_elo_from_record(None, "2850.5") == 2850.5
    assert parse_elo_from_record(None, None) is None


def test_game_feature_extraction_dimensions_and_types() -> None:
    """Verify full 35D game feature extraction from raw round trajectory."""
    synthetic_traj = []
    stages = ["2-1", "2-3", "2-5", "3-1", "3-2", "3-5", "4-1", "4-2", "4-5", "5-1", "5-2", "5-5", "6-1"]

    for i, stg in enumerate(stages):
        synthetic_traj.append({
            "round_stage": stg,
            "focal_health": max(10, 100 - i * 6),
            "focal_level": min(9, 4 + i // 2),
            "focal_gold": 30 + (i % 4) * 10,
            "outcome": "victory" if i % 2 == 0 else "defeat",
            "focal_board": [
                {"unit": "TFT18_Irelia", "tier": 2, "cost": 1, "loc": "A1", "items": []},
                {"unit": "TFT18_Illaoi", "tier": 2, "cost": 3, "loc": "A2", "items": ["TFT_Item_WarmogsArmor"]},
                {"unit": "TFT18_Jinx", "tier": 2, "cost": 4, "loc": "D7", "items": ["TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade"]},
            ],
        })

    feats = extract_game_features(synthetic_traj, use_neural=False)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # Macro features sanity
    # Index 0: final_placement
    assert 1.0 <= feats[0] <= 8.0
    # Index 1: rounds_survived
    assert feats[1] == len(stages)
    # Index 3: final_health
    assert feats[3] == synthetic_traj[-1]["focal_health"]
    # Index 12: final_level
    assert feats[12] == synthetic_traj[-1]["focal_level"]
    # Index 23: three_item_carries_count
    assert feats[23] >= 1.0  # Jinx has 3 items


def test_game_feature_extraction_with_neural_evaluator() -> None:
    """Verify neural quality metrics extraction with active BoardQualityNet."""
    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    for u in ["TFT18_Irelia", "TFT18_Illaoi", "TFT18_Jinx"]:
        vocab.add_champion(u)
    for it in ["TFT_Item_WarmogsArmor", "TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade"]:
        item_vocab.add_item(it)

    trunk = MultiModalFusionTrunk(
        num_champs=len(vocab),
        num_items=len(item_vocab),
        num_traits=len(trait_vocab),
        board_feat_dim=256,
        fused_dim=320,
    )
    bq_net = BoardQualityNet(trunk=trunk, hidden_dim=64)

    synthetic_traj = [
        {
            "round_stage": "2-1",
            "focal_health": 100,
            "focal_level": 4,
            "focal_gold": 10,
            "focal_board": [{"unit": "TFT18_Irelia", "tier": 1, "loc": "A1"}],
        },
        {
            "round_stage": "5-1",
            "focal_health": 60,
            "focal_level": 8,
            "focal_gold": 40,
            "focal_board": [
                {"unit": "TFT18_Illaoi", "tier": 2, "loc": "A2", "items": ["TFT_Item_WarmogsArmor"]},
                {"unit": "TFT18_Jinx", "tier": 2, "loc": "D7", "items": ["TFT_Item_InfinityEdge"]},
            ],
        },
    ]

    neural_feats = extract_neural_metrics(
        synthetic_traj,
        board_evaluator=bq_net,
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
    )

    assert len(neural_feats) == 10
    assert not any(np.isnan(x) for x in neural_feats)
    # Mean quality should be within valid bounds [0.0, 10.0]
    assert 0.0 <= neural_feats[0] <= 10.0


def test_solution_space_ood_detector_inlier_outlier() -> None:
    """Verify OOD detector identifies normal human feature distributions vs aberrant bot outliers."""
    rng = np.random.default_rng(42)

    # 1. Generate human baseline distribution (60 games)
    n_feats = len(FEATURE_NAMES)
    human_X = rng.normal(loc=0.0, scale=1.0, size=(60, n_feats)).astype(np.float32)

    detector = SolutionSpaceOODDetector(nu=0.05)
    detector.fit(human_X)

    assert detector.is_fitted

    # Test inlier sample (near distribution mean)
    inlier_sample = np.mean(human_X, axis=0)
    inlier_result = detector.score_match(inlier_sample)

    assert inlier_result["is_in_distribution"]
    assert inlier_result["inlier_confidence_pct"] >= 50.0
    assert inlier_result["mahalanobis_distance"] < 6.0

    # Test aberrant bot outlier sample (15 standard deviations away)
    outlier_sample = inlier_sample + 15.0
    outlier_result = detector.score_match(outlier_sample)

    assert not outlier_result["is_in_distribution"]
    assert outlier_result["inlier_confidence_pct"] < 25.0
    assert outlier_result["mahalanobis_distance"] > 10.0
    assert len(outlier_result["top_divergent_features"]) == 3


def test_match_elo_regressor_save_load_roundtrip(tmp_path: Path) -> None:
    """Verify MatchEloRegressor fitting, prediction, and disk serialization."""
    rng = np.random.default_rng(42)
    n_samples = 40
    n_feats = len(FEATURE_NAMES)

    X = rng.normal(size=(n_samples, n_feats)).astype(np.float32)
    y = (3000.0 - X[:, 0] * 300.0 + rng.normal(scale=50.0, size=n_samples)).astype(np.float32)
    y = np.clip(y, 400.0, 4200.0)

    reg = MatchEloRegressor(n_estimators=15, max_depth=3, random_state=42)
    reg.fit(X, y)

    assert reg.is_fitted
    test_vec = X[0]
    pred_elo, pred_tier, ood_info = reg.predict_match(test_vec)

    assert 100.0 <= pred_elo <= 4600.0
    assert isinstance(pred_tier, str)
    assert "inlier_confidence_pct" in ood_info

    # Save and reload
    save_path = tmp_path / "elo_regressor_test.pkl"
    reg.save(save_path)
    assert save_path.exists()

    loaded = MatchEloRegressor.load(save_path)
    assert loaded.is_fitted
    loaded_elo, loaded_tier, loaded_ood = loaded.predict_match(test_vec)

    assert abs(pred_elo - loaded_elo) < 1e-4
    assert pred_tier == loaded_tier
    assert abs(ood_info["mahalanobis_distance"] - loaded_ood["mahalanobis_distance"]) < 1e-4


def test_elo_monotonicity_performance() -> None:
    """Verify that a high-performing Challenger match profile predicts higher Elo than an Iron profile."""
    rng = np.random.default_rng(42)
    n_samples = 80
    n_feats = len(FEATURE_NAMES)

    X = rng.normal(size=(n_samples, n_feats)).astype(np.float32)
    X[:, 0] = rng.uniform(1.0, 8.0, size=n_samples)
    X[:, 12] = rng.uniform(4.0, 9.0, size=n_samples)

    y = 2000.0 + (9.0 - X[:, 0]) * 250.0 + (X[:, 12] - 4.0) * 150.0
    y = np.clip(y, 200.0, 4400.0)

    reg = MatchEloRegressor(n_estimators=40, max_depth=3, random_state=42)
    reg.fit(X, y)

    # Challenger test profile: 1st place, level 9
    challenger_prof = np.zeros(n_feats, dtype=np.float32)
    challenger_prof[0] = 1.0
    challenger_prof[12] = 9.0

    # Iron test profile: 8th place, level 4
    iron_prof = np.zeros(n_feats, dtype=np.float32)
    iron_prof[0] = 8.0
    iron_prof[12] = 4.0

    elo_challenger, _, _ = reg.predict_match(challenger_prof)
    elo_iron, _, _ = reg.predict_match(iron_prof)

    assert elo_challenger > elo_iron + 500.0, (
        f"Expected Challenger Elo ({elo_challenger:.0f}) > Iron Elo ({elo_iron:.0f}) + 500"
    )
