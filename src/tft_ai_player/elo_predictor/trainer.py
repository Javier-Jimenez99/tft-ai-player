"""Training and evaluation pipeline for full-match Elo regression and OOD analysis.

Compares two distinct Elo labeling objectives:
  - Target A: Match Lobby Elo (`match_elo`)
  - Target B: Player Peak / Latest Elo (`player_last_elo`)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .dataset import FullMatchEloDataset
from .model import MatchEloRegressor, SolutionSpaceOODDetector, elo_to_tier_name


def evaluate_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Perform 5-fold cross validation and return regression metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes = []
    rmses = []
    r2s = []
    spearmans = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        reg = MatchEloRegressor(n_estimators=160, learning_rate=0.08, max_depth=4)
        reg.fit(X_train, y_train)

        preds = np.clip(reg.model.predict(X_val), 100.0, 4600.0)
        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        r2s.append(r2_score(y_val, preds))
        corr, _ = spearmanr(y_val, preds)
        spearmans.append(corr)

    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
        "spearman": float(np.mean(spearmans)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Full-Match Elo Regressor and OOD Solution Space Detector")
    parser.add_argument("--data-dir", type=str, default="D:/tft-winner-data/set18/players")
    parser.add_argument("--output-dir", type=str, default="models/elo_predictor")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default="tft-elo-predictor")
    parser.add_argument("--run-name", type=str, default="elo_hybrid_neural_regression")
    parser.add_argument("--entity", type=str, default="javier-jimenez99")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" [TFT FULL-MATCH ELO REGRESSOR & OOD SOLUTION SPACE PIPELINE]")
    print(f" Source Directory: {args.data_dir} | Output: {out_dir}")
    print("=" * 80)

    t0 = time.time()
    dataset = FullMatchEloDataset(data_dir=args.data_dir, max_files=args.max_files)
    dt_load = time.time() - t0
    print(f" [+] Extracted {len(dataset):,} full matches from dataset in {dt_load:.1f}s!")

    if len(dataset) < 50:
        print("[!] Not enough matches extracted to train.", file=sys.stderr)
        return 1

    X = dataset.feature_matrix
    y_match = dataset.match_elos
    y_player = dataset.player_last_elos

    # 1. Evaluate Target A: Match Lobby Elo
    print("\n" + "-" * 80)
    print(" [*] Evaluating Target A: MATCH LOBBY ELO (avg_match_rating / focal_tier)...")
    res_match = evaluate_kfold(X, y_match)
    print(f"     MAE: {res_match['mae']:.1f} ELO | RMSE: {res_match['rmse']:.1f} | R^2: {res_match['r2']:.3f} | Spearman: {res_match['spearman']:.3f}")

    # 2. Evaluate Target B: Player Latest / Peak Elo
    print("\n" + "-" * 80)
    print(" [*] Evaluating Target B: PLAYER PEAK/LATEST ELO (highest/final recorded rating)...")
    res_player = evaluate_kfold(X, y_player)
    print(f"     MAE: {res_player['mae']:.1f} ELO | RMSE: {res_player['rmse']:.1f} | R^2: {res_player['r2']:.3f} | Spearman: {res_player['spearman']:.3f}")

    # Comparison summary
    print("\n" + "=" * 80)
    print(" [TARGET COMPARISON SUMMARY: MATCH ELO vs PLAYER PEAK ELO]")
    print(f" {'Metric':20} | {'Target A (Match Elo)':25} | {'Target B (Player Peak Elo)':25}")
    print("-" * 80)
    print(f" {'Mean Absolute Error':20} | {res_match['mae']:10.1f} ELO             | {res_player['mae']:10.1f} ELO")
    print(f" {'RMSE':20} | {res_match['rmse']:10.1f} ELO             | {res_player['rmse']:10.1f} ELO")
    print(f" {'R^2 Variance Exp':20} | {res_match['r2']*100:10.1f}%                 | {res_player['r2']*100:10.1f}%")
    print(f" {'Spearman Rank Corr':20} | {res_match['spearman']:10.3f}                 | {res_player['spearman']:10.3f}")
    print("=" * 80)

    # 3. Fit Final Production Models
    print("\n [*] Training final full-dataset models and fitting OOD solution space detector...")
    model_match = MatchEloRegressor(n_estimators=180, learning_rate=0.07, max_depth=4)
    model_match.fit(X, y_match)
    path_match = out_dir / "elo_regressor_match_level.pkl"
    model_match.save(path_match)
    print(f" [+] Saved Match-Level Model: {path_match}")

    model_player = MatchEloRegressor(n_estimators=180, learning_rate=0.07, max_depth=4)
    model_player.fit(X, y_player)
    path_player = out_dir / "elo_regressor_player_level.pkl"
    model_player.save(path_player)
    print(f" [+] Saved Player-Level Model: {path_player}")

    backup_dir = Path("D:/tft-winner-data/set18/models/elo_predictor")
    if backup_dir.exists():
        model_match.save(backup_dir / "elo_regressor_match_level.pkl")
        model_player.save(backup_dir / "elo_regressor_player_level.pkl")
        print(f" [+] Saved backup copies to {backup_dir}")

    # 4. Feature Importance Analysis
    print("\n" + "-" * 80)
    print(" [TOP 8 FEATURE IMPORTANCES FOR PREDICTING TFT ELO]")
    print("-" * 80)
    importances = model_match.get_feature_importances()
    for rank, (feat, imp) in enumerate(importances[:8], 1):
        print(f"  {rank}. {feat:28}: {imp*100:5.1f}%")
    print("-" * 80)

    # 5. Out-of-Distribution (OOD) Solution Space Verification Demonstration
    print("\n" + "=" * 80)
    print(" [SOLUTION SPACE OUTLIER (OOD) DEMONSTRATION]")
    print("=" * 80)
    # Test sample from real human dataset
    real_sample = X[0]
    real_elo, real_tier, real_ood = model_match.predict_match(real_sample)
    print(f"  1. Human Match Sample:")
    print(f"     -> Predicted Elo: {real_elo:.0f} ({real_tier})")
    print(f"     -> Inlier Confidence: {real_ood['inlier_confidence_pct']:.1f}%")
    print(f"     -> Mahalanobis Distance: {real_ood['mahalanobis_distance']:.2f} (Status: {'IN-DISTRIBUTION' if real_ood['is_in_distribution'] else 'OUTLIER'})")

    # Construct an aberrant bot outlier (e.g. stays lvl 4 at stage 6-1, hoards 95 gold, 0 completed items)
    bot_sample = real_sample.copy()
    bot_sample[0] = 8.0  # 8th place
    bot_sample[7] = 95.0 # 95 avg gold (refused to spend)
    bot_sample[12] = 4.0 # never leveled up
    bot_sample[17] = 8.0 # 8 gold board cost
    bot_sample[22] = 0.0 # 0 items
    bot_elo, bot_tier, bot_ood = model_match.predict_match(bot_sample)
    print(f"\n  2. Aberrant Bot Trajectory (Outlier test):")
    print(f"     -> Raw Regressor Prediction: {bot_elo:.0f} ({bot_tier})")
    print(f"     -> Inlier Confidence: {bot_ood['inlier_confidence_pct']:.1f}%")
    print(f"     -> Mahalanobis Distance: {bot_ood['mahalanobis_distance']:.2f} (Status: {'IN-DISTRIBUTION' if bot_ood['is_in_distribution'] else 'OUTLIER FLAGGED!'})")
    print(f"     -> Top Divergent Features:")
    for d in bot_ood["top_divergent_features"]:
        print(f"        * {d['feature']}: value={d['value']:.1f} (human mean={d['human_mean']:.1f}, z={d['z_score']:.1f})")
    print("=" * 80 + "\n")

    # 6. WandB Logging
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                entity=args.entity,
                config={
                    "num_matches": len(dataset),
                    "features": len(X[0]),
                    "backend": model_match._backend,
                },
            )
            payload = {
                "match_elo/mae": res_match["mae"],
                "match_elo/rmse": res_match["rmse"],
                "match_elo/r2": res_match["r2"],
                "match_elo/spearman": res_match["spearman"],
                "player_elo/mae": res_player["mae"],
                "player_elo/rmse": res_player["rmse"],
                "player_elo/r2": res_player["r2"],
                "player_elo/spearman": res_player["spearman"],
            }
            for feat, imp in importances:
                payload[f"importance/{feat}"] = float(imp)
            wandb.log(payload)
            wandb.finish()
            print(" [+] Successfully logged benchmark telemetry to WandB.")
        except Exception as e:
            print(f" [!] WandB logging warning: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
