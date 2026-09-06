"""Elo Regressor and Solution Space Outlier (OOD) Detector.

Predicts competitive player Elo rating from complete match trajectories and
computes whether an AI/bot trajectory lies within the human gameplay solution manifold.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .features import FEATURE_NAMES, extract_game_features


TIER_THRESHOLDS = [
    (3800.0, "CHALLENGER"),
    (3300.0, "GRANDMASTER"),
    (2800.0, "MASTER"),
    (2400.0, "DIAMOND"),
    (2000.0, "EMERALD"),
    (1600.0, "PLATINUM"),
    (1200.0, "GOLD"),
    (800.0, "SILVER"),
    (400.0, "BRONZE"),
    (0.0, "IRON"),
]


def elo_to_tier_name(elo: float) -> str:
    """Map continuous numeric Elo to TFT competitive tier name."""
    for threshold, tier in TIER_THRESHOLDS:
        if elo >= threshold:
            return tier
    return "IRON"


class SolutionSpaceOODDetector:
    """Out-of-Distribution (OOD) and Solution Space Inlier Detector for full TFT games.

    Uses One-Class SVM and Mahalanobis statistical distance to verify whether an
    analyzed match comes from the human gameplay distribution or represents an
    aberrant, out-of-distribution bot trajectory.
    """

    def __init__(self, nu: float = 0.05, gamma: str = "scale") -> None:
        self.scaler = StandardScaler()
        self.oc_svm = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf")
        self.cov_estimator = EmpiricalCovariance()
        self.is_fitted = False
        self.feature_means: np.ndarray = np.zeros(len(FEATURE_NAMES))
        self.feature_stds: np.ndarray = np.ones(len(FEATURE_NAMES))

    def fit(self, X: np.ndarray) -> SolutionSpaceOODDetector:
        """Fit inlier boundary on human match feature vectors."""
        X_scaled = self.scaler.fit_transform(X)
        self.oc_svm.fit(X_scaled)
        self.cov_estimator.fit(X_scaled)

        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds[self.feature_stds < 1e-6] = 1.0
        self.is_fitted = True
        return self

    def score_match(self, feat_vec: np.ndarray) -> dict[str, Any]:
        """Evaluate whether a game feature vector lies within the human solution space."""
        if not self.is_fitted:
            raise RuntimeError("SolutionSpaceOODDetector must be fitted before scoring.")

        if feat_vec.ndim == 1:
            feat_vec = feat_vec.reshape(1, -1)

        feat_scaled = self.scaler.transform(feat_vec)

        # 1. One-Class SVM decision score (positive = inlier, negative = outlier)
        svm_score = float(self.oc_svm.decision_function(feat_scaled)[0])
        is_svm_inlier = bool(self.oc_svm.predict(feat_scaled)[0] == 1)

        # 2. Mahalanobis distance to human centroid
        mahalanobis_dist = float(np.sqrt(self.cov_estimator.mahalanobis(feat_scaled)[0]))

        # Normalize inlier confidence score to [0.0, 100.0]%
        # A Mahalanobis dist < 5 on 25D is typically within the 99% chi-squared ellipsoid
        inlier_confidence = float(np.clip(100.0 / (1.0 + np.exp((mahalanobis_dist - 5.5) * 0.8)), 0.0, 100.0))

        # 3. Identify most divergent features (z-score distance)
        z_scores = (feat_vec[0] - self.feature_means) / self.feature_stds
        divergent_indices = np.argsort(np.abs(z_scores))[::-1][:3]
        top_divergences = [
            {
                "feature": FEATURE_NAMES[idx],
                "value": float(feat_vec[0, idx]),
                "human_mean": float(self.feature_means[idx]),
                "z_score": float(z_scores[idx]),
            }
            for idx in divergent_indices
        ]

        return {
            "is_in_distribution": bool(inlier_confidence >= 40.0),
            "inlier_confidence_pct": inlier_confidence,
            "mahalanobis_distance": mahalanobis_dist,
            "svm_decision_margin": svm_score,
            "top_divergent_features": top_divergences,
        }


class MatchEloRegressor:
    """Predicts competitive Elo rating from a complete match trajectory."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=16,
                random_state=random_state,
                verbose=-1,
            )
            self._backend = "lightgbm"
        except ImportError:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )
            self._backend = "sklearn"

        self.ood_detector = SolutionSpaceOODDetector()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> MatchEloRegressor:
        """Train Elo regressor and fit OOD solution space detector."""
        self.model.fit(X, y)
        self.ood_detector.fit(X)
        self.is_fitted = True
        return self

    def predict_match(self, feat_vec: np.ndarray) -> tuple[float, str, dict[str, Any]]:
        """Predict Elo, predicted Tier, and OOD solution space metrics."""
        if not self.is_fitted:
            raise RuntimeError("MatchEloRegressor must be fitted before predict.")

        if feat_vec.ndim == 1:
            feat_vec = feat_vec.reshape(1, -1)

        raw_pred = float(self.model.predict(feat_vec)[0])
        pred_elo = float(np.clip(raw_pred, 100.0, 4600.0))
        tier_name = elo_to_tier_name(pred_elo)
        ood_info = self.ood_detector.score_match(feat_vec)

        return pred_elo, tier_name, ood_info

    def get_feature_importances(self) -> list[tuple[str, float]]:
        """Return ranking of features by model importance."""
        if not self.is_fitted:
            return []
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        norm_imp = importances / np.sum(importances)
        ranked = sorted(zip(FEATURE_NAMES, norm_imp), key=lambda x: x[1], reverse=True)
        return ranked

    def save(self, path: str | Path) -> None:
        """Serialize model bundle to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> MatchEloRegressor:
        """Load serialized model bundle from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
