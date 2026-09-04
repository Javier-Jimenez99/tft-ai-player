"""Standalone CLI script to train, evaluate, and serialize the TFT Round Winner model."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .trainer import RoundWinnerPredictor, RoundWinnerTrainer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and serialize the TFT Round Winner Prediction Model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\set18\players") if Path(r"D:\tft-winner-data\set18\players").exists() else Path(r"D:\tft-winner-data\players"),
        help="Path to directory containing player round CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\set18\models\round_winner") if Path(r"D:\tft-winner-data\set18\models").exists() else Path("models/round_winner"),
        help="Directory where model bundle and metadata will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of matches held out for unbiased testing.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        default=False,
        help="Also train XGBoost and ensemble (slower, ~50s extra).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap on number of player CSV files to load (for rapid testing).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data splitting and model initialization.",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        default=None,
        help="Optional custom filename for saved model bundle (e.g. round_winner_ensemble.joblib).",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default=None,
        help="Optional custom filename for saved metadata (e.g. metadata_ensemble.json).",
    )
    parser.add_argument(
        "--allowed-tiers",
        nargs="+",
        default=["CHALLENGER"],
        help="Allowed tiers for training dataset (default: CHALLENGER). Pass ALL to use all data.",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    if not data_dir.exists():
        fallback = Path("data/players")
        if fallback.exists():
            data_dir = fallback
        else:
            print(f"Error: Data directory not found at '{data_dir}' or '{fallback}'.", file=sys.stderr)
            return 1

    model_filename = args.model_filename or ("round_winner_ensemble.joblib" if args.include_xgboost else "round_winner_model.joblib")
    metadata_filename = args.metadata_filename or ("metadata_ensemble.json" if args.include_xgboost else "metadata.json")

    print(f"=== TFT 3-Output Multi-Task Round Winner & Combat Damage Model Training ===")
    print(f"Data source: {data_dir.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Target model file: {model_filename}")
    print(f"Target metadata file: {metadata_filename}\n")

    trainer = RoundWinnerTrainer(random_state=args.random_state)
    
    # Load dataset with tier isolation
    allowed_tiers = None if "ALL" in [t.upper() for t in args.allowed_tiers] else args.allowed_tiers
    df_clean = trainer.load_dataset(data_dir, allowed_tiers=allowed_tiers)

    print(f"Loaded {len(df_clean):,} clean round observations across {df_clean['match_id'].nunique():,} unique matches.")

    metadata = trainer.fit_and_evaluate(
        df_clean,
        test_size=args.test_size,
        include_xgboost=args.include_xgboost,
        verbose=True,
    )

    print("\n--- Model Leaderboard: Win/Loss Classification ---")
    metrics = metadata["evaluation_metrics"]
    for name, res in metrics.items():
        if res and isinstance(res, dict) and "model" in res:
            print(
                f"{res['model']:<25} | Brier: {res['brier_score']:.4f} | "
                f"ROC-AUC: {res['roc_auc']:.4f} | Acc: {res['accuracy']*100:.2f}% | ECE: {res['ece']*100:.2f}%"
            )

    if "combat_damage_regression" in metrics:
        dmg_eval = metrics["combat_damage_regression"]
        print(f"\n--- Model Leaderboard: Combat Damage Regression (Heads 2 & 3) ---")
        print(f"Test Loss Rounds Evaluated: {dmg_eval['test_loss_samples']:,}")
        print(f"Mean Absolute Error (MAE) : {dmg_eval['damage_mae']:.3f} HP")
        print(f"Root Mean Squared Error   : {dmg_eval['damage_rmse']:.3f} HP")

    print("\n--- Tactical Stress Sanity Tests ---")
    for st in metadata["stress_tests"]:
        status = "PASSED" if st["passed"] else "FAILED"
        print(f"[{status}] {st['scenario']:<60} -> P(Win): {st['predicted_win_prob']*100:.1f}% (Expected: {st['expected']})")

    saved_path = trainer.save(
        args.output_dir,
        model_filename=model_filename,
        metadata_filename=metadata_filename,
    )
    print(f"\n[SUCCESS] 3-Output Multi-Task model saved to: {saved_path.resolve()}")
    print(f"[SUCCESS] Training metadata saved to: {(args.output_dir / metadata_filename).resolve()}")

    # 3-Output Verification Check
    predictor = RoundWinnerPredictor.load(saved_path)
    is_set18 = any(c.startswith("DA_") or "18" in c for c in predictor.feature_names_)
    if is_set18:
        p_win, dmg_a, dmg_b = predictor.predict_combat(
            focal_board=[{"unit": "DA_18_Xayah", "tier": 2, "loc": "D1", "items": ["DA_InfinityEdge", "DA_LastWhisper"]}],
            opponent_board=[{"unit": "DA_18_Camille", "tier": 1, "loc": "A1", "items": []}],
            round_stage="4-2",
        )
        print(f"[VERIFY 3-OUTPUT] 2-Star Xayah vs 1-Star Camille -> P(Win A) = {p_win*100:.1f}% | Loss Damage to A: {dmg_a} HP | Loss Damage to B: {dmg_b} HP")
    else:
        p_win, dmg_a, dmg_b = predictor.predict_combat(
            focal_board=[{"unit": "TFT17_Jinx", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge"]}],
            opponent_board=[{"unit": "TFT17_Aatrox", "tier": 1, "loc": "A1", "items": []}],
            round_stage="4-2",
        )
        print(f"[VERIFY 3-OUTPUT] 2-Star Jinx vs 1-Star Aatrox -> P(Win A) = {p_win*100:.1f}% | Loss Damage to A: {dmg_a} HP | Loss Damage to B: {dmg_b} HP")

    return 0


if __name__ == "__main__":
    sys.exit(main())
