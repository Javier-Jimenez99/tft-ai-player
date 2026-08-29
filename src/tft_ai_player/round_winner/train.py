"""Standalone CLI script to train, evaluate, and serialize the TFT Round Winner model."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .trainer import RoundWinnerPredictor, RoundWinnerTrainer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and serialize the TFT Round Winner Prediction Model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\players"),
        help="Path to directory containing player round CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/round_winner"),
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

    print(f"=== TFT Round Winner Model Training Pipeline ===")
    print(f"Data source: {data_dir.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Target model file: {model_filename}")
    print(f"Target metadata file: {metadata_filename}\n")

    trainer = RoundWinnerTrainer(random_state=args.random_state)
    
    # Load dataset
    path = Path(data_dir)
    csv_files = list(path.glob("*.csv"))
    if not csv_files and (path / "players").exists():
        path = path / "players"
        csv_files = list(path.glob("*.csv"))
    if not csv_files:
        csv_files = list(path.rglob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in '{data_dir}' or subdirectories.", file=sys.stderr)
        return 1

    if args.max_files:
        csv_files = csv_files[:args.max_files]
        print(f"Subsampled {len(csv_files)} CSV files for fast execution.")
    else:
        print(f"Found {len(csv_files)} CSV files to process.")
    
    df_list = []
    for file_path in csv_files:
        try:
            df_list.append(pd.read_csv(file_path, encoding="utf-8"))
        except Exception:
            df_list.append(pd.read_csv(file_path, encoding="latin-1"))

    df_raw = pd.concat(df_list, ignore_index=True)
    df_clean = df_raw.dropna(subset=["label"]).copy()
    df_clean["label"] = df_clean["label"].astype(int)
    df_clean = df_clean.drop_duplicates(subset=["match_id", "round_stage", "focal_player"]).reset_index(drop=True)

    print(f"Loaded {len(df_clean):,} clean round observations across {df_clean['match_id'].nunique():,} unique matches.")

    metadata = trainer.fit_and_evaluate(
        df_clean,
        test_size=args.test_size,
        include_xgboost=args.include_xgboost,
        verbose=True,
    )

    print("\n--- Model Leaderboard vs MetaTFT Baseline ---")
    metrics = metadata["evaluation_metrics"]
    for name, res in metrics.items():
        if res:
            print(
                f"{res['model']:<25} | Brier: {res['brier_score']:.4f} | "
                f"ROC-AUC: {res['roc_auc']:.4f} | Acc: {res['accuracy']*100:.2f}% | ECE: {res['ece']*100:.2f}%"
            )

    print("\n--- Tactical Stress Sanity Tests ---")
    for st in metadata["stress_tests"]:
        status = "PASSED" if st["passed"] else "FAILED"
        print(f"[{status}] {st['scenario']:<60} -> P(Win): {st['predicted_win_prob']*100:.1f}% (Expected: {st['expected']})")

    saved_path = trainer.save(
        args.output_dir,
        model_filename=model_filename,
        metadata_filename=metadata_filename,
    )
    print(f"\n[SUCCESS] Model bundle saved to: {saved_path.resolve()}")
    print(f"[SUCCESS] Training metadata saved to: {(args.output_dir / metadata_filename).resolve()}")

    # Quick test of loaded predictor
    predictor = RoundWinnerPredictor.load(saved_path)
    is_set18 = any(c.startswith("DA_") or "18" in c for c in predictor.feature_names_)
    if is_set18:
        sample_prob = predictor.predict_proba(
            focal_board=[{"unit": "DA_18_Xayah", "tier": 2, "loc": "D1", "items": ["DA_InfinityEdge", "DA_LastWhisper"]}],
            opponent_board=[{"unit": "DA_18_Camille", "tier": 1, "loc": "A1", "items": []}],
        )
        print(f"[VERIFY] Inference Sanity Check (2-Star 2-Item Xayah vs 1-Star Camille): P(Focal Win) = {sample_prob*100:.1f}%")
    else:
        sample_prob = predictor.predict_proba(
            focal_board=[{"unit": "TFT17_Jinx", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge"]}],
            opponent_board=[{"unit": "TFT17_Aatrox", "tier": 1, "loc": "A1", "items": []}],
        )
        print(f"[VERIFY] Inference Sanity Check (2-Star Jinx vs 1-Star Aatrox): P(Focal Win) = {sample_prob*100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
