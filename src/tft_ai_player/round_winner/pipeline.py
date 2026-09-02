"""Pipeline factory builders for TFT round winner models."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

from .features import TFTBoardFeatureExtractor


def build_hist_gradient_boosting_pipeline(
    *,
    feature_extractor: TFTBoardFeatureExtractor | None = None,
    max_iter: int = 400,
    max_depth: int = 10,
    max_leaf_nodes: int = 63,
    learning_rate: float = 0.04,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.1,
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline with feature extraction and HistGradientBoosting Classifier."""
    extractor = feature_extractor or TFTBoardFeatureExtractor()
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
        **kwargs,
    )
    return Pipeline([("features", extractor), ("model", clf)])


def build_xgboost_pipeline(
    *,
    feature_extractor: TFTBoardFeatureExtractor | None = None,
    n_estimators: int = 600,
    max_depth: int = 8,
    learning_rate: float = 0.03,
    subsample: float = 0.85,
    colsample_bytree: float = 0.75,
    reg_alpha: float = 0.05,
    reg_lambda: float = 0.5,
    tree_method: str = "hist",
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline with feature extraction and tuned XGBoost Classifier."""
    try:
        from xgboost import XGBClassifier
    except (ImportError, Exception):
        return build_hist_gradient_boosting_pipeline(feature_extractor=feature_extractor, random_state=random_state)

    extractor = feature_extractor or TFTBoardFeatureExtractor()
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method=tree_method,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
    return Pipeline([("features", extractor), ("model", clf)])


def build_lightgbm_pipeline(
    *,
    feature_extractor: TFTBoardFeatureExtractor | None = None,
    n_estimators: int = 600,
    max_depth: int = 10,
    num_leaves: int = 127,
    learning_rate: float = 0.03,
    min_child_samples: int = 20,
    subsample: float = 0.85,
    colsample_bytree: float = 0.75,
    reg_alpha: float = 0.05,
    reg_lambda: float = 0.5,
    random_state: int = 42,
    verbose: int = -1,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline with feature extraction and tuned LightGBM Classifier."""
    try:
        from lightgbm import LGBMClassifier
        extractor = feature_extractor or TFTBoardFeatureExtractor()
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=verbose,
            n_jobs=-1,
            **kwargs,
        )
        return Pipeline([("features", extractor), ("model", clf)])
    except (ImportError, Exception):
        return build_hist_gradient_boosting_pipeline(
            feature_extractor=feature_extractor,
            max_iter=n_estimators,
            max_depth=max_depth,
            max_leaf_nodes=num_leaves,
            learning_rate=learning_rate,
            min_samples_leaf=min_child_samples,
            random_state=random_state,
        )


def build_catboost_pipeline(
    *,
    feature_extractor: TFTBoardFeatureExtractor | None = None,
    iterations: int = 600,
    depth: int = 7,
    learning_rate: float = 0.04,
    l2_leaf_reg: float = 2.5,
    random_seed: int = 42,
    verbose: int = 0,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline with feature extraction and CatBoost Classifier."""
    try:
        from catboost import CatBoostClassifier
        extractor = feature_extractor or TFTBoardFeatureExtractor()
        clf = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=verbose,
            thread_count=-1,
            **kwargs,
        )
        return Pipeline([("features", extractor), ("model", clf)])
    except (ImportError, Exception):
        return build_hist_gradient_boosting_pipeline(
            feature_extractor=feature_extractor,
            max_iter=iterations,
            max_depth=depth,
            learning_rate=learning_rate,
            random_state=random_seed,
        )


def build_random_forest_pipeline(
    *,
    feature_extractor: TFTBoardFeatureExtractor | None = None,
    n_estimators: int = 250,
    max_depth: int | None = 18,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline with feature extraction and Random Forest Classifier."""
    extractor = feature_extractor or TFTBoardFeatureExtractor()
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
    return Pipeline([("features", extractor), ("model", clf)])


def build_calibrated_pipeline(
    estimator: Any,
    *,
    method: str = "isotonic",
    cv: int | str = 3,
) -> CalibratedClassifierCV:
    """Wrap an estimator with probability calibration (isotonic or sigmoid)."""
    return CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=cv,
    )
