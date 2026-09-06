"""Evaluation utility to predict player/agent Elo and verify solution space inlier status."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FEATURE_NAMES, extract_game_features
from .model import MatchEloRegressor, elo_to_tier_name


def compute_tft_sanity_invariants(
    trajectory: Sequence[Mapping[str, Any]],
    raw_features: Mapping[str, float],
) -> dict[str, Any]:
    """Verify core TFT gameplay invariants to detect unhuman/degenerate agent exploits.

    Returns:
        dict containing:
            - sanity_score: Score from 0.0 to 100.0 (higher = more human-coherent)
            - is_sane: Boolean True if score >= 70.0
            - violations: List of detected invariant violations
    """
    score = 100.0
    violations = []

    if not trajectory:
        return {"sanity_score": 0.0, "is_sane": False, "violations": ["Empty trajectory"]}

    final_round = trajectory[-1]
    final_hp = float(final_round.get("focal_health", 0) or 0)
    final_gold = float(final_round.get("focal_gold", 0) or 0)
    final_lvl = float(raw_features.get("final_level", 1.0))
    total_rounds = float(raw_features.get("rounds_survived", 1.0))
    board_cost = float(raw_features.get("final_board_cost", 0.0))
    items_count = float(raw_features.get("final_completed_items", 0.0))
    active_traits = float(raw_features.get("active_traits_count", 0.0))

    # Invariant 1: Dying rich ("Murió rico")
    # Human players in competitive play roll down when facing elimination
    if final_hp <= 0 and final_gold >= 40.0:
        penalty = min(35.0, (final_gold - 30.0) * 1.0)
        score -= penalty
        violations.append(
            f"Dying Rich: Eliminated with {final_gold:.0f} gold unspent (bot failed to roll down to stabilize)"
        )

    # Invariant 2: Phantom Board (Board cost too low for level)
    # At level 7+, a human board has at least 25-30g cost
    if final_lvl >= 7.0 and board_cost < 20.0:
        score -= 25.0
        violations.append(
            f"Phantom Board: Reached level {final_lvl:.0f} but board cost is only {board_cost:.0f}g"
        )

    # Invariant 3: Zero / Discarded Item Usage
    # Surviving past round 15 with fewer than 2 completed items on board
    if total_rounds >= 15.0 and items_count < 2.0:
        score -= 20.0
        violations.append(
            f"Item Disregard: Survived {total_rounds:.0f} rounds but equipped only {items_count:.0f} items"
        )

    # Invariant 4: Extreme Level Incoherence
    # Level 4 after 20 rounds, or level 9 with 0 gold and 10 HP at stage 3
    if total_rounds >= 20.0 and final_lvl <= 5.0:
        score -= 25.0
        violations.append(
            f"Level Stagnation: Survived {total_rounds:.0f} rounds but only reached level {final_lvl:.0f}"
        )

    # Invariant 5: Traitlessness
    if total_rounds >= 15.0 and active_traits < 2.0:
        score -= 15.0
        violations.append(
            f"No Synergies: Survived {total_rounds:.0f} rounds with only {active_traits:.0f} active traits"
        )

    final_score = float(np.clip(score, 0.0, 100.0))
    return {
        "sanity_score": final_score,
        "is_sane": bool(final_score >= 70.0),
        "violations": violations,
    }


def evaluate_game_trajectory(
    trajectory: Sequence[Mapping[str, Any]],
    model_path: str | Path = "models/elo_predictor/elo_regressor_player_level.pkl",
) -> dict[str, Any]:
    """Evaluate an entire match played by human or RL agent and predict competitive Elo.

    Combines:
      1. Machine Learning Elo Regressor (Continuous Elo & Tier)
      2. Out-of-Distribution (OOD) Solution Space Detector (Mahalanobis distance & One-Class SVM)
      3. TFT Domain Sanity Checker (Rule-based invariant verification)
      4. Composite Trust Score (Overall certainty that the prediction is genuine and not an exploit)

    Returns:
        dict containing:
            - predicted_elo: Continuous Elo rating (100 - 4500+)
            - predicted_tier: Competitive tier name (e.g. 'MASTER', 'EMERALD')
            - composite_trust_score: Aggregate trust score (0 - 100)%
            - trust_level: 'HIGH_CONFIDENCE', 'MODERATE_CONFIDENCE', or 'UNTRUSTED_OOD'
            - is_in_distribution: Boolean indicating if gameplay matches human manifold
            - inlier_confidence_pct: Statistical confidence score (0 - 100)%
            - mahalanobis_distance: Distance to human gameplay distribution centroid
            - sanity: Sanity score and list of domain violations
            - top_divergent_features: List of gameplay features deviating most from humans
            - raw_features: Dict of extracted 25D gameplay metrics
    """
    p = Path(model_path)
    if not p.exists():
        fallback_p = Path("models/elo_predictor/elo_regressor_match_level.pkl")
        if fallback_p.exists():
            p = fallback_p
        else:
            raise FileNotFoundError(f"Trained Elo model not found at {model_path}")

    model = MatchEloRegressor.load(p)
    feat_vec = extract_game_features(trajectory)
    pred_elo, pred_tier, ood_info = model.predict_match(feat_vec)

    raw_feat_dict = {FEATURE_NAMES[i]: float(feat_vec[i]) for i in range(len(FEATURE_NAMES))}
    sanity_info = compute_tft_sanity_invariants(trajectory, raw_feat_dict)

    # Composite trust score combining OOD manifold proximity and TFT domain sanity
    # Weights: 50% OOD Inlier Confidence + 50% Sanity Score
    ood_conf = float(ood_info["inlier_confidence_pct"])
    sanity_conf = float(sanity_info["sanity_score"])
    composite_trust = float(np.clip(0.50 * ood_conf + 0.50 * sanity_conf, 0.0, 100.0))

    if composite_trust >= 75.0 and ood_info["is_in_distribution"] and sanity_info["is_sane"]:
        trust_level = "HIGH_CONFIDENCE"
    elif composite_trust >= 45.0:
        trust_level = "MODERATE_CONFIDENCE"
    else:
        trust_level = "UNTRUSTED_OOD"

    return {
        "predicted_elo": pred_elo,
        "predicted_tier": pred_tier,
        "composite_trust_score": composite_trust,
        "trust_level": trust_level,
        "is_in_distribution": ood_info["is_in_distribution"],
        "inlier_confidence_pct": ood_conf,
        "mahalanobis_distance": ood_info["mahalanobis_distance"],
        "svm_decision_margin": ood_info["svm_decision_margin"],
        "sanity": sanity_info,
        "top_divergent_features": ood_info["top_divergent_features"],
        "raw_features": raw_feat_dict,
    }

