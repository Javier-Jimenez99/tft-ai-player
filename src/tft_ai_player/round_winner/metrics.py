"""Evaluation metrics for TFT round winner probability predictions and MetaTFT comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_brier_metrics(
    y_true: np.ndarray | pd.Series | list[int],
    y_prob: np.ndarray | pd.Series | list[float],
    y_ref_prob: np.ndarray | pd.Series | list[float] | None = None,
) -> dict[str, float]:
    """Calculate Brier score and Brier Skill Score (BSS).

    Brier score is the mean squared difference between predicted probability and actual 0/1 outcome:
    BS = (1/N) * sum((p_i - y_i)^2). Lower is better (0 = perfect, 0.25 = uninformative coin toss).

    Brier Skill Score:
    BSS = 1 - (BS_model / BS_ref). Higher is better (1 = perfect, > 0 = skill over reference baseline).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    bs = float(brier_score_loss(y_t, y_p))

    # Reference Brier score: climatology (mean win rate) or provided reference (e.g., MetaTFT)
    if y_ref_prob is not None:
        y_ref = np.clip(np.asarray(y_ref_prob, dtype=float), 0.0, 1.0)
        bs_ref = float(brier_score_loss(y_t, y_ref))
    else:
        p_base = float(np.mean(y_t))
        bs_ref = float(np.mean((y_t - p_base) ** 2))

    bss = 1.0 - (bs / bs_ref) if bs_ref > 0 else 0.0

    return {
        "brier_score": bs,
        "ref_brier_score": bs_ref,
        "brier_skill_score": bss,
    }


def compute_calibration_table(
    y_true: np.ndarray | pd.Series | list[int],
    y_prob: np.ndarray | pd.Series | list[float],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float, float]:
    """Compute reliability calibration table, Expected Calibration Error (ECE), and MCE.

    Returns:
    - DataFrame with bin ranges, mean predicted probability, observed empirical win rate, and sample count.
    - ECE (Expected Calibration Error).
    - MCE (Maximum Calibration Error).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_t)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_p, bin_edges, right=True)

    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0

    for i in range(1, n_bins + 1):
        mask = bin_assignments == i
        count = int(np.sum(mask))
        if count > 0:
            mean_prob = float(np.mean(y_p[mask]))
            empirical_winrate = float(np.mean(y_t[mask]))
            gap = abs(empirical_winrate - mean_prob)
            ece += (count / n) * gap
            mce = max(mce, gap)
        else:
            mean_prob = (bin_edges[i - 1] + bin_edges[i]) / 2.0
            empirical_winrate = np.nan
            gap = 0.0

        rows.append(
            {
                "bin": f"[{bin_edges[i-1]:.2f}, {bin_edges[i]:.2f}]",
                "count": count,
                "mean_pred_prob": mean_prob,
                "empirical_win_rate": empirical_winrate,
                "calibration_gap": gap,
            }
        )

    df_cal = pd.DataFrame(rows)
    return df_cal, float(ece), float(mce)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series | list[int],
    y_prob: np.ndarray | pd.Series | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute standard classification metrics based on probability decision threshold."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_p >= threshold).astype(int)

    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_t, y_p))
    except ValueError:
        auc = 0.5

    try:
        loss = float(log_loss(y_t, y_p))
    except ValueError:
        loss = np.nan

    cm = confusion_matrix(y_t, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "log_loss": loss,
        "confusion_matrix": cm,
    }


def evaluate_probabilistic_model(
    y_true: np.ndarray | pd.Series | list[int],
    y_prob: np.ndarray | pd.Series | list[float],
    *,
    model_name: str = "Model",
    y_ref_prob: np.ndarray | pd.Series | list[float] | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Comprehensive evaluation bundle for round-winner probabilistic models."""
    brier = compute_brier_metrics(y_true, y_prob, y_ref_prob=y_ref_prob)
    _, ece, mce = compute_calibration_table(y_true, y_prob, n_bins=n_bins)
    clf = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    return {
        "model": model_name,
        "brier_score": brier["brier_score"],
        "brier_skill_score": brier["brier_skill_score"],
        "ece": ece,
        "mce": mce,
        "log_loss": clf["log_loss"],
        "roc_auc": clf["roc_auc"],
        "accuracy": clf["accuracy"],
        "precision": clf["precision"],
        "recall": clf["recall"],
        "f1": clf["f1"],
        "confusion_matrix": clf["confusion_matrix"],
    }
