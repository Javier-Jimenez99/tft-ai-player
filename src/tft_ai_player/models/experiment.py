"""High-capacity ensembling and super-learner benchmark vs MetaTFT."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from tft_ai_player.models.features import TFTBoardFeatureExtractor
from tft_ai_player.models.metrics import evaluate_probabilistic_model, compute_brier_metrics, compute_classification_metrics

def load_and_clean_data(data_dir: Path) -> pd.DataFrame:
    csv_files = list(data_dir.glob("*.csv"))
    print(f"Ingesting {len(csv_files)} CSV files from {data_dir.resolve()}...")
    df_list = []
    for p in csv_files:
        try:
            df_list.append(pd.read_csv(p, encoding="utf-8"))
        except UnicodeDecodeError:
            df_list.append(pd.read_csv(p, encoding="latin-1"))
    df_raw = pd.concat(df_list, ignore_index=True)
    df_clean = df_raw.drop_duplicates(subset=["match_id", "round_stage", "focal_player"]).copy()
    print(f"Loaded {len(df_clean):,} clean observations across {df_clean['match_id'].nunique():,} matches.")
    return df_clean

def run_experimentation() -> None:
    data_dir = Path(r"D:\tft-winner-data\players")
    if not data_dir.exists():
        data_dir = Path("data/players")

    df = load_and_clean_data(data_dir)

    # GroupShuffleSplit strictly by match_id
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["match_id"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = df_train["label"].values
    y_test = df_test["label"].values

    print("\n--- Fitting Advanced High-Resolution Feature Extractor ---")
    extractor = TFTBoardFeatureExtractor(
        include_champions=True,
        include_items=True,
        include_augments=True,
        include_placement=True,
        include_traits=True,
        include_absolute=True,
        min_champ_freq=8,
        min_item_freq=8,
        min_aug_freq=8,
    )
    extractor.fit(df_train)
    print(f"Total features extracted: {len(extractor.get_feature_names_out())}")

    print("Transforming training and test sets...")
    X_train = extractor.transform(df_train)
    X_test = extractor.transform(df_test)

    print("\n--- Training Next-Gen Ensemble Candidates ---")

    # 1. High-Capacity LightGBM (127 leaves, deep trees)
    print("1. Fitting High-Capacity LightGBM (127 leaves, 850 trees)...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=850,
        learning_rate=0.025,
        num_leaves=127,
        max_depth=10,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train, y_train)

    # 2. High-Capacity XGBoost (Hist, max_depth 8)
    print("2. Fitting High-Capacity XGBoost (Hist, depth 8, 850 trees)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=850,
        learning_rate=0.025,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.5,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)

    # 3. High-Capacity CatBoost (depth 7, 800 iterations)
    print("3. Fitting High-Capacity CatBoost (depth 7, 800 trees)...")
    cb_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=2.5,
        random_seed=42,
        thread_count=-1,
        verbose=0,
    )
    cb_model.fit(X_train, y_train)

    # 4. Probability Calibration (3-fold CV)
    print("4. Fitting 3-Fold Isotonic Calibrated Models...")
    cal_lgbm = CalibratedClassifierCV(estimator=lgbm_model, method="isotonic", cv=3)
    cal_lgbm.fit(X_train, y_train)

    cal_xgb = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=3)
    cal_xgb.fit(X_train, y_train)

    cal_cb = CalibratedClassifierCV(estimator=cb_model, method="isotonic", cv=3)
    cal_cb.fit(X_train, y_train)

    # Test Predictions
    preds_lgbm = cal_lgbm.predict_proba(X_test)[:, 1]
    preds_xgb = cal_xgb.predict_proba(X_test)[:, 1]
    preds_cb = cal_cb.predict_proba(X_test)[:, 1]

    # Ensembles
    blend_equal = (preds_lgbm + preds_xgb + preds_cb) / 3.0
    blend_weighted = 0.50 * preds_lgbm + 0.35 * preds_xgb + 0.15 * preds_cb

    all_models = {
        "Calibrated LightGBM (127 leaves)": preds_lgbm,
        "Calibrated XGBoost (depth 8)": preds_xgb,
        "Calibrated CatBoost (depth 7)": preds_cb,
        "Ensemble Equal Blend": blend_equal,
        "Ensemble Weighted Super-Blend": blend_weighted,
    }

    # MetaTFT Comparison
    mask_meta = df_test["metatft_win_prob"].notna()
    df_test_meta = df_test[mask_meta].reset_index(drop=True)
    y_test_meta = y_test[mask_meta]
    meta_probs = df_test_meta["metatft_win_prob"].values

    print(f"\n========================================================")
    print(f"BENCHMARK RESULTS ON {len(df_test_meta):,} TEST ROUNDS (vs MetaTFT)")
    print(f"========================================================")

    results = []
    meta_res = evaluate_probabilistic_model(y_test_meta, meta_probs, model_name="MetaTFT Model")
    meta_res["brier_delta_vs_meta"] = 0.0
    results.append(meta_res)

    for name, p_all in all_models.items():
        p_sub = p_all[mask_meta]
        res = evaluate_probabilistic_model(y_test_meta, p_sub, model_name=f"Our {name}")
        res["brier_delta_vs_meta"] = res["brier_score"] - meta_res["brier_score"]
        results.append(res)

    df_res = pd.DataFrame(results).drop(columns=["confusion_matrix"])
    print(df_res.sort_values(by="brier_score").to_string(index=False))

    # Stage Breakdown
    df_test_meta["stage_major"] = df_test_meta["round_stage"].apply(
        lambda s: f"Stage {s.split('-')[0]}" if isinstance(s, str) and "-" in s else "Other"
    )
    best_probs = blend_weighted[mask_meta]

    print("\n--- STAGE-BY-STAGE BREAKDOWN (Best Weighted Super-Blend vs MetaTFT) ---")
    stage_rows = []
    for stage, grp in df_test_meta.groupby("stage_major"):
        if len(grp) < 30:
            continue
        g_idx = grp.index.values
        y_g = y_test_meta[g_idx]
        m_g = meta_probs[g_idx]
        o_g = best_probs[g_idx]

        bs_m = compute_brier_metrics(y_g, m_g)["brier_score"]
        bs_o = compute_brier_metrics(y_g, o_g)["brier_score"]
        acc_m = compute_classification_metrics(y_g, m_g)["accuracy"]
        acc_o = compute_classification_metrics(y_g, o_g)["accuracy"]

        stage_rows.append({
            "Stage": stage,
            "Rounds": len(grp),
            "Meta Brier": bs_m,
            "Our Brier": bs_o,
            "Meta Acc": acc_m,
            "Our Acc": acc_o,
            "Brier Delta (Ours - Meta)": bs_o - bs_m,
        })
    df_stage = pd.DataFrame(stage_rows).sort_values(by="Stage")
    print(df_stage.to_string(index=False))

if __name__ == "__main__":
    run_experimentation()
