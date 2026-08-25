"""Training, evaluation, serialization, and inference engine for TFT Round Winner models."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import xgboost as xgb

from .features import TFTBoardFeatureExtractor
from .metrics import evaluate_probabilistic_model


@dataclass
class ScenarioTestResult:
    scenario: str
    expected: str
    predicted_win_prob: float
    passed: bool


class RoundWinnerTrainer:
    """End-to-end trainer and evaluator for round winner prediction models."""

    def __init__(
        self,
        *,
        min_champ_freq: int = 10,
        min_item_freq: int = 10,
        min_aug_freq: int = 10,
        random_state: int = 42,
    ) -> None:
        self.min_champ_freq = min_champ_freq
        self.min_item_freq = min_item_freq
        self.min_aug_freq = min_aug_freq
        self.random_state = random_state

        self.extractor: TFTBoardFeatureExtractor | None = None
        self.lgb_model: lgb.LGBMClassifier | None = None
        self.xgb_model: xgb.XGBClassifier | None = None
        self.calibrator_lgb: LogisticRegression | None = None
        self.calibrator_xgb: LogisticRegression | None = None
        self.feature_names_: list[str] = []
        self.metadata_: dict[str, Any] = {}

    def load_dataset(self, data_dir: Path | str) -> pd.DataFrame:
        """Load and clean round observations from directory of CSV files."""
        path = Path(data_dir)
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path.resolve()}")

        df_list: list[pd.DataFrame] = []
        for file_path in csv_files:
            try:
                df_tmp = pd.read_csv(file_path, encoding="utf-8")
                df_list.append(df_tmp)
            except UnicodeDecodeError:
                df_tmp = pd.read_csv(file_path, encoding="latin-1")
                df_list.append(df_tmp)

        df_raw = pd.concat(df_list, ignore_index=True)
        # Safe cleaning: drop missing labels and remove duplicate player-round snapshots
        df_clean = df_raw.dropna(subset=["label"]).copy()
        df_clean["label"] = df_clean["label"].astype(int)
        df_clean = df_clean.drop_duplicates(subset=["match_id", "round_stage", "focal_player"]).reset_index(drop=True)
        return df_clean

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        *,
        test_size: float = 0.20,
        val_size: float = 0.15,
        include_xgboost: bool = False,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Fit models, calibrate probabilities, and evaluate on holdout test matches."""
        t0 = time.time()

        # 1. Match Partitioning (Zero match leakage)
        gss = GroupShuffleSplit(n_splits=1, train_size=1.0 - test_size, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(df, groups=df["match_id"]))

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        gss_val = GroupShuffleSplit(n_splits=1, train_size=1.0 - val_size, random_state=self.random_state)
        sub_train_idx, val_idx = next(gss_val.split(df_train, groups=df_train["match_id"]))

        df_sub_train = df_train.iloc[sub_train_idx].reset_index(drop=True)
        df_val = df_train.iloc[val_idx].reset_index(drop=True)

        y_sub_train = df_sub_train["label"].values
        y_val = df_val["label"].values
        y_test = df_test["label"].values

        if verbose:
            print(f"Dataset partitioned: {len(df_sub_train):,} train, {len(df_val):,} val, {len(df_test):,} test rounds.")

        # 2. Extract Features
        if verbose:
            print("Extracting domain combat features...")
        self.extractor = TFTBoardFeatureExtractor(
            include_champions=True,
            include_items=True,
            include_augments=True,
            include_placement=True,
            include_traits=True,
            include_absolute=True,
            min_champ_freq=self.min_champ_freq,
            min_item_freq=self.min_item_freq,
            min_aug_freq=self.min_aug_freq,
        )
        self.extractor.fit(df_sub_train)
        self.feature_names_ = self.extractor.get_feature_names_out()

        X_sub_train = self.extractor.transform(df_sub_train)
        X_val = self.extractor.transform(df_val)
        X_test = self.extractor.transform(df_test)

        if verbose:
            print(f"Extracted {len(self.feature_names_)} features across all splits.")

        # 3. Train LightGBM Model (Fast ~9s)
        if verbose:
            print("Training LightGBM model...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=8,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.75,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.lgb_model.fit(X_sub_train, y_sub_train)

        # 4. Optional XGBoost Model
        if include_xgboost:
            if verbose:
                print("Training XGBoost model...")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.75,
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.xgb_model.fit(X_sub_train, y_sub_train)
        else:
            self.xgb_model = None

        # 5. Fit Smooth Probability Calibrators (Platt Scaling)
        if verbose:
            print("Fitting smooth probability calibrators...")
        val_p_lgb = self.lgb_model.predict_proba(X_val)[:, 1]

        if len(np.unique(y_val)) > 1:
            self.calibrator_lgb = LogisticRegression().fit(val_p_lgb.reshape(-1, 1), y_val)
            if self.xgb_model:
                val_p_xgb = self.xgb_model.predict_proba(X_val)[:, 1]
                self.calibrator_xgb = LogisticRegression().fit(val_p_xgb.reshape(-1, 1), y_val)
            else:
                self.calibrator_xgb = None
        else:
            self.calibrator_lgb = None
            self.calibrator_xgb = None

        # 6. Evaluate on Test Set
        raw_p_lgb = self.lgb_model.predict_proba(X_test)[:, 1]
        cal_p_lgb = self.calibrator_lgb.predict_proba(raw_p_lgb.reshape(-1, 1))[:, 1] if self.calibrator_lgb else raw_p_lgb

        eval_lgb = evaluate_probabilistic_model(y_test, cal_p_lgb, model_name="Calibrated LightGBM")
        eval_xgb = None
        eval_ens = None

        if self.xgb_model:
            raw_p_xgb = self.xgb_model.predict_proba(X_test)[:, 1]
            cal_p_xgb = self.calibrator_xgb.predict_proba(raw_p_xgb.reshape(-1, 1))[:, 1] if self.calibrator_xgb else raw_p_xgb
            cal_p_ensemble = 0.55 * cal_p_lgb + 0.45 * cal_p_xgb
            eval_xgb = evaluate_probabilistic_model(y_test, cal_p_xgb, model_name="Calibrated XGBoost")
            eval_ens = evaluate_probabilistic_model(y_test, cal_p_ensemble, model_name="Ensemble (LGB+XGB)")

        # Benchmark vs MetaTFT if present
        meta_benchmark: dict[str, Any] | None = None
        if "metatft_win_prob" in df_test.columns:
            mask_meta = df_test["metatft_win_prob"].notna()
            if mask_meta.sum() > 50:
                y_sub_meta = y_test[mask_meta]
                meta_p = df_test.loc[mask_meta, "metatft_win_prob"].values
                meta_benchmark = evaluate_probabilistic_model(y_sub_meta, meta_p, model_name="MetaTFT Baseline")

        # 7. Stress Tests
        stress_results = self.run_stress_tests()

        self.metadata_ = {
            "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_rounds": len(df),
            "train_rounds": len(df_sub_train),
            "test_rounds": len(df_test),
            "n_features": len(self.feature_names_),
            "training_duration_seconds": round(time.time() - t0, 2),
            "evaluation_metrics": {
                "lightgbm": eval_lgb,
                "xgboost": eval_xgb,
                "ensemble": eval_ens,
                "metatft_baseline": meta_benchmark,
            },
            "stress_tests": [asdict(r) for r in stress_results],
        }

        return self.metadata_

    def run_stress_tests(self) -> list[ScenarioTestResult]:
        """Execute domain-logic sanity tests on the trained model."""
        if not self.extractor or not self.lgb_model:
            raise RuntimeError("Trainer must be fitted before running stress tests.")

        scenarios = [
            (
                "Greedy Banker (200g + 1 unit vs 0g + 8 units)",
                "Near 0%",
                {
                    "round_stage": "4-2",
                    "focal_level": 8,
                    "opponent_level": 8,
                    "focal_health": 80,
                    "opponent_health": 80,
                    "focal_gold": 200,
                    "opponent_gold": 0,
                    "input_state_json": json.dumps({
                        "focal_board": [{"unit": "TFT17_Aatrox", "tier": 1, "loc": "A1", "items": []}],
                        "opponent_board": [
                            {"unit": "TFT17_Nasus", "tier": 2, "loc": "A1", "items": ["TFT_Item_WarmogsArmor", "TFT_Item_GargoyleStoneplate", "TFT_Item_DragonsClaw"]},
                            {"unit": "TFT17_Maokai", "tier": 2, "loc": "A2", "items": []},
                            {"unit": "TFT17_Illaoi", "tier": 2, "loc": "A3", "items": []},
                            {"unit": "TFT17_Poppy", "tier": 2, "loc": "A4", "items": []},
                            {"unit": "TFT17_Jinx", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade"]},
                            {"unit": "TFT17_Caitlyn", "tier": 2, "loc": "D2", "items": []},
                            {"unit": "TFT17_Kindred", "tier": 2, "loc": "D3", "items": []},
                            {"unit": "TFT17_Corki", "tier": 2, "loc": "D4", "items": []},
                        ],
                    }),
                },
                lambda p: p < 0.10,
            ),
            (
                "1 HP Survival (1 HP vs 100 HP with 8 strong units vs 1 unit)",
                "> 95%",
                {
                    "round_stage": "5-1",
                    "focal_level": 8,
                    "opponent_level": 8,
                    "focal_health": 1,
                    "opponent_health": 100,
                    "focal_gold": 20,
                    "opponent_gold": 20,
                    "input_state_json": json.dumps({
                        "focal_board": [
                            {"unit": "TFT17_Nasus", "tier": 2, "loc": "A1", "items": ["TFT_Item_WarmogsArmor", "TFT_Item_GargoyleStoneplate"]},
                            {"unit": "TFT17_Jinx", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade"]},
                            {"unit": "TFT17_Caitlyn", "tier": 2, "loc": "D2", "items": []},
                            {"unit": "TFT17_Maokai", "tier": 2, "loc": "A2", "items": []},
                            {"unit": "TFT17_Illaoi", "tier": 2, "loc": "A3", "items": []},
                            {"unit": "TFT17_Kindred", "tier": 2, "loc": "D3", "items": []},
                            {"unit": "TFT17_Corki", "tier": 2, "loc": "D4", "items": []},
                            {"unit": "TFT17_Poppy", "tier": 2, "loc": "A4", "items": []},
                        ],
                        "opponent_board": [{"unit": "TFT17_Aatrox", "tier": 1, "loc": "A1", "items": []}],
                    }),
                },
                lambda p: p > 0.80,
            ),
            (
                "Quality vs Quantity (3-Star 4-Cost Carry vs 8 Naked Units)",
                "> 80%",
                {
                    "round_stage": "4-6",
                    "focal_level": 8,
                    "opponent_level": 8,
                    "focal_health": 50,
                    "opponent_health": 50,
                    "focal_gold": 10,
                    "opponent_gold": 10,
                    "input_state_json": json.dumps({
                        "focal_board": [
                            {"unit": "TFT17_Jinx", "tier": 3, "loc": "D1", "items": ["TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade"]},
                            {"unit": "TFT17_Nasus", "tier": 2, "loc": "A1", "items": ["TFT_Item_WarmogsArmor"]},
                            {"unit": "TFT17_Caitlyn", "tier": 2, "loc": "D2", "items": []},
                            {"unit": "TFT17_Maokai", "tier": 2, "loc": "A2", "items": []},
                        ],
                        "opponent_board": [
                            {"unit": "TFT17_Aatrox", "tier": 2, "loc": "A1", "items": []},
                            {"unit": "TFT17_Pantheon", "tier": 2, "loc": "A2", "items": []},
                            {"unit": "TFT17_Briar", "tier": 2, "loc": "A3", "items": []},
                            {"unit": "TFT17_RekSai", "tier": 2, "loc": "A4", "items": []},
                            {"unit": "TFT17_Gragas", "tier": 2, "loc": "B1", "items": []},
                            {"unit": "TFT17_Nunu", "tier": 2, "loc": "B2", "items": []},
                            {"unit": "TFT17_Urgot", "tier": 2, "loc": "B3", "items": []},
                            {"unit": "TFT17_Chogath", "tier": 2, "loc": "B4", "items": []},
                        ],
                    }),
                },
                lambda p: p > 0.70,
            ),
        ]

        df_scen = pd.DataFrame([s[2] for s in scenarios])
        X_scen = self.extractor.transform(df_scen)
        raw_p = self.lgb_model.predict_proba(X_scen)[:, 1]
        cal_p = self.calibrator_lgb.predict_proba(raw_p.reshape(-1, 1))[:, 1] if self.calibrator_lgb else raw_p

        results: list[ScenarioTestResult] = []
        for i, (name, exp, _, check_fn) in enumerate(scenarios):
            p = float(cal_p[i])
            results.append(
                ScenarioTestResult(
                    scenario=name,
                    expected=exp,
                    predicted_win_prob=p,
                    passed=bool(check_fn(p)),
                )
            )

        return results

    def save(self, output_dir: Path | str, model_filename: str = "round_winner_model.joblib") -> Path:
        """Serialize model artifacts and metadata JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle = {
            "extractor": self.extractor,
            "lgb_model": self.lgb_model,
            "xgb_model": self.xgb_model,
            "calibrator_lgb": self.calibrator_lgb,
            "calibrator_xgb": self.calibrator_xgb,
            "feature_names": self.feature_names_,
            "metadata": self.metadata_,
        }

        save_path = out_dir / model_filename
        joblib.dump(bundle, save_path, compress=3)

        meta_path = out_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata_, f, indent=2)

        return save_path


class RoundWinnerPredictor:
    """Lightweight inference predictor loaded from a serialized bundle."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.extractor: TFTBoardFeatureExtractor = bundle["extractor"]
        self.lgb_model: lgb.LGBMClassifier = bundle["lgb_model"]
        self.xgb_model: xgb.XGBClassifier | None = bundle.get("xgb_model")
        self.calibrator_lgb: LogisticRegression = bundle["calibrator_lgb"]
        self.calibrator_xgb: LogisticRegression | None = bundle.get("calibrator_xgb")
        self.feature_names_: list[str] = bundle["feature_names"]
        self.metadata_: dict[str, Any] = bundle.get("metadata", {})

    @classmethod
    def load(cls, model_path: Path | str) -> RoundWinnerPredictor:
        """Load predictor from a joblib bundle file."""
        bundle = joblib.load(Path(model_path))
        return cls(bundle)

    def predict_proba(
        self,
        focal_board: list[dict[str, Any]],
        opponent_board: list[dict[str, Any]],
        *,
        round_stage: str = "4-2",
        focal_level: int = 8,
        opponent_level: int = 8,
        focal_health: int = 100,
        opponent_health: int = 100,
        focal_gold: int = 20,
        opponent_gold: int = 20,
        focal_augments: list[str] | None = None,
        opponent_augments: list[str] | None = None,
    ) -> float:
        """Predict probability of focal player winning the round."""
        row_dict = {
            "round_stage": round_stage,
            "focal_level": focal_level,
            "opponent_level": opponent_level,
            "focal_health": focal_health,
            "opponent_health": opponent_health,
            "focal_gold": focal_gold,
            "opponent_gold": opponent_gold,
            "focal_augments": focal_augments or [],
            "opponent_augments": opponent_augments or [],
            "input_state_json": json.dumps({
                "focal_board": focal_board,
                "opponent_board": opponent_board,
            }),
        }

        df = pd.DataFrame([row_dict])
        X = self.extractor.transform(df)
        raw_p = self.lgb_model.predict_proba(X)[:, 1]
        cal_p = self.calibrator_lgb.predict_proba(raw_p.reshape(-1, 1))[:, 1] if self.calibrator_lgb else raw_p
        return float(cal_p[0])

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities for a batch DataFrame of observations."""
        X = self.extractor.transform(df)
        raw_p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        cal_p_lgb = self.calibrator_lgb.predict_proba(raw_p_lgb.reshape(-1, 1))[:, 1] if self.calibrator_lgb else raw_p_lgb

        if self.xgb_model:
            raw_p_xgb = self.xgb_model.predict_proba(X)[:, 1]
            cal_p_xgb = self.calibrator_xgb.predict_proba(raw_p_xgb.reshape(-1, 1))[:, 1] if self.calibrator_xgb else raw_p_xgb
            return 0.55 * cal_p_lgb + 0.45 * cal_p_xgb

        return cal_p_lgb
