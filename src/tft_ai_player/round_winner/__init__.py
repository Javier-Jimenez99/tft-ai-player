"""Round winner prediction module for TFT AI Player."""

from .features import (
    ALL_SET17_TRAITS,
    CHAMP_BASE_COSTS,
    SET17_TRAIT_CHAMPIONS,
    SET17_TRAIT_THRESHOLDS,
    TFTBoardFeatureExtractor,
)
from .metrics import (
    compute_brier_metrics,
    compute_calibration_table,
    compute_classification_metrics,
    evaluate_probabilistic_model,
)
from .pipeline import (
    build_calibrated_pipeline,
    build_catboost_pipeline,
    build_lightgbm_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
)
from .trainer import RoundWinnerPredictor, RoundWinnerTrainer

__all__ = [
    "ALL_SET17_TRAITS",
    "CHAMP_BASE_COSTS",
    "SET17_TRAIT_CHAMPIONS",
    "SET17_TRAIT_THRESHOLDS",
    "RoundWinnerPredictor",
    "RoundWinnerTrainer",
    "TFTBoardFeatureExtractor",
    "build_calibrated_pipeline",
    "build_catboost_pipeline",
    "build_lightgbm_pipeline",
    "build_random_forest_pipeline",
    "build_xgboost_pipeline",
    "compute_brier_metrics",
    "compute_calibration_table",
    "compute_classification_metrics",
    "evaluate_probabilistic_model",
]
