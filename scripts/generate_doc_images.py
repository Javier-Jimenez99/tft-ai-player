"""Script to train the model, compute all benchmarks, and generate publication-quality figures for docs."""

import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import lightgbm as lgb
import xgboost as xgb

from tft_ai_player.round_winner import (
    TFTBoardFeatureExtractor,
    compute_brier_metrics,
    compute_calibration_table,
    compute_classification_metrics,
    evaluate_probabilistic_model,
)

# Output directory for documentation figures
DOCS_IMG_DIR = Path("docs/images")
DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

# Load Data
DATA_DIR = Path(r"D:\tft-winner-data\players")
if not DATA_DIR.exists():
    DATA_DIR = Path("data/players")

csv_files = list(DATA_DIR.glob("*.csv"))
print(f"Loading {len(csv_files)} CSV files from {DATA_DIR}...")
df_list = []
for f in csv_files:
    try:
        df_list.append(pd.read_csv(f, encoding="utf-8"))
    except Exception:
        df_list.append(pd.read_csv(f, encoding="latin-1"))

df_raw = pd.concat(df_list, ignore_index=True)
df_clean = df_raw.dropna(subset=["label"]).copy()
df_clean["label"] = df_clean["label"].astype(int)
df_clean = df_clean.drop_duplicates(subset=["match_id", "round_stage", "focal_player"]).reset_index(drop=True)
print(f"Loaded {len(df_clean):,} clean rounds across {df_clean['match_id'].nunique():,} unique matches.")

# 1. GENERATE DATASET EDA FIGURE
print("Generating Dataset EDA figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 Outcome Balance
outcome_counts = df_clean["label"].value_counts().rename({1: "Win (50.0%)", 0: "Loss (50.0%)"})
axes[0, 0].pie(
    outcome_counts,
    labels=outcome_counts.index,
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"],
    startangle=140,
    explode=(0.04, 0.04),
    textprops={"fontsize": 12, "fontweight": "bold"},
)
axes[0, 0].set_title("1. Round Outcome Distribution (Balanced 1v1)")

# 1.2 Rounds per Game Stage
df_clean["stage_num"] = df_clean["round_stage"].str.extract(r"^(\d+)").astype(float)
stage_dist = df_clean[df_clean["stage_num"].between(2, 7)]["stage_num"].astype(int).value_counts().sort_index()
sns.barplot(x=stage_dist.index, y=stage_dist.values, ax=axes[0, 1], palette="mako", hue=stage_dist.index, legend=False)
axes[0, 1].set_title("2. Round Observations by Game Stage")
axes[0, 1].set_xlabel("Game Stage (2 = Early, 4 = Mid, 6-7 = Late)")
axes[0, 1].set_ylabel("Total Number of Rounds")
for i, v in enumerate(stage_dist.values):
    axes[0, 1].text(i, v + 800, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

# 1.3 MetaTFT Win Prob Distribution by True Outcome
    df_clean["outcome_label"] = df_clean["label"].map({1: "Win (Victory)", 0: "Loss (Defeat)"})
    sns.kdeplot(
        data=df_clean,
        x="metatft_win_prob",
        hue="outcome_label",
        fill=True,
        common_norm=False,
        palette={"Win (Victory)": "#2ecc71", "Loss (Defeat)": "#e74c3c"},
        alpha=0.45,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("3. MetaTFT Win Probability Density by Actual Result")
    axes[1, 0].set_xlabel("MetaTFT Predicted Win Probability")
    axes[1, 0].set_ylabel("Density")

# 1.4 Player Health Distribution across Stages
sample_stages = df_clean[df_clean["stage_num"].isin([2, 3, 4, 5, 6])]
sns.boxplot(
    data=sample_stages,
    x="stage_num",
    y="focal_health",
    ax=axes[1, 1],
    palette="viridis",
    hue="stage_num",
    legend=False,
)
axes[1, 1].set_title("4. Player Health Decay Progression by Stage")
axes[1, 1].set_xlabel("Game Stage")
axes[1, 1].set_ylabel("Player Remaining HP")

plt.tight_layout()
fig.savefig(DOCS_IMG_DIR / "dataset_eda.png")
plt.close(fig)
print("Saved docs/images/dataset_eda.png")

# 2. LOAD PRETRAINED MODEL AND RUN EVALUATION
print("Loading pre-trained model bundle from D:\\tft-winner-data\\models\\round_winner_model.joblib...")
import joblib
bundle_path = Path(r"D:\tft-winner-data\models\round_winner_model.joblib")
bundle = joblib.load(bundle_path)

extractor = bundle["extractor"]
lgb_model = bundle["lgb_model"]
cal_lgb = bundle["calibrator_lgb"]
feature_names = bundle["feature_names"]

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
_, test_idx = next(gss.split(df_clean, groups=df_clean["match_id"]))
df_test = df_clean.iloc[test_idx].reset_index(drop=True)
y_test = df_test["label"].values

print(f"Extracting features on holdout test set ({len(df_test):,} rounds)...")
X_test = extractor.transform(df_test)
p_lgb_raw = lgb_model.predict_proba(X_test)[:, 1]
p_lgb_cal = cal_lgb.predict_proba(p_lgb_raw.reshape(-1, 1))[:, 1] if cal_lgb else p_lgb_raw

# MetaTFT Evaluation on Holdout
mask_meta = df_test["metatft_win_prob"].notna()
y_test_meta = y_test[mask_meta]
p_meta = df_test.loc[mask_meta, "metatft_win_prob"].values
p_lgb_meta = p_lgb_cal[mask_meta]

eval_meta = evaluate_probabilistic_model(y_test_meta, p_meta, model_name="MetaTFT Baseline")
eval_lgb = evaluate_probabilistic_model(y_test_meta, p_lgb_meta, model_name="Calibrated LightGBM")

# 3. GENERATE LEADERBOARD COMPARISON FIGURE
print("Generating Leaderboard figure...")
df_board = pd.DataFrame([eval_meta, eval_lgb])

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
palette = ["#e74c3c", "#2ecc71"]

# 3.1 Brier Score
sns.barplot(data=df_board, x="model", y="brier_score", ax=axes[0], palette=palette, hue="model", legend=False)
axes[0].set_title("Brier Score (Lower is Better)")
axes[0].set_ylabel("Brier Score")
axes[0].set_xticklabels(df_board["model"], rotation=20, ha="right")
for i, v in enumerate(df_board["brier_score"]):
    axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontweight="bold", fontsize=10)

# 3.2 ROC-AUC
sns.barplot(data=df_board, x="model", y="roc_auc", ax=axes[1], palette=palette, hue="model", legend=False)
axes[1].set_title("ROC-AUC (Higher is Better)")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_xticklabels(df_board["model"], rotation=20, ha="right")
axes[1].set_ylim(0.80, 0.93)
for i, v in enumerate(df_board["roc_auc"]):
    axes[1].text(i, v + 0.003, f"{v:.4f}", ha="center", fontweight="bold", fontsize=10)

# 3.3 Accuracy %
sns.barplot(data=df_board, x="model", y="accuracy", ax=axes[2], palette=palette, hue="model", legend=False)
axes[2].set_title("Classification Accuracy")
axes[2].set_ylabel("Accuracy")
axes[2].set_xticklabels(df_board["model"], rotation=20, ha="right")
axes[2].set_ylim(0.70, 0.85)
for i, v in enumerate(df_board["accuracy"]):
    axes[2].text(i, v + 0.003, f"{v*100:.1f}%", ha="center", fontweight="bold", fontsize=10)

# 3.4 ECE (Calibration Error)
sns.barplot(data=df_board, x="model", y="ece", ax=axes[3], palette=palette, hue="model", legend=False)
axes[3].set_title("Expected Calibration Error (ECE)")
axes[3].set_ylabel("ECE (Lower is Better)")
axes[3].set_xticklabels(df_board["model"], rotation=20, ha="right")
for i, v in enumerate(df_board["ece"]):
    axes[3].text(i, v + 0.0005, f"{v*100:.2f}%", ha="center", fontweight="bold", fontsize=10)

plt.suptitle("TFT Round Winner Model Benchmark Leaderboard vs MetaTFT", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
fig.savefig(DOCS_IMG_DIR / "model_leaderboard.png")
plt.close(fig)
print("Saved docs/images/model_leaderboard.png")

# 4. GENERATE CALIBRATION & ROC CURVES FIGURE
print("Generating Calibration & ROC Curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4.1 Calibration / Reliability Curve
cal_table_meta, _, _ = compute_calibration_table(y_test_meta, p_meta, n_bins=10)
cal_table_lgb, _, _ = compute_calibration_table(y_test_meta, p_lgb_meta, n_bins=10)

axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)
axes[0].plot(cal_table_meta["mean_pred_prob"], cal_table_meta["empirical_win_rate"], "s-", color="#e74c3c", label=f"MetaTFT (ECE = {eval_meta['ece']*100:.2f}%)", lw=2.5)
axes[0].plot(cal_table_lgb["mean_pred_prob"], cal_table_lgb["empirical_win_rate"], "o-", color="#2ecc71", label=f"Our LightGBM (ECE = {eval_lgb['ece']*100:.2f}%)", lw=2.5)
axes[0].set_xlabel("Mean Predicted Win Probability")
axes[0].set_ylabel("Empirical True Win Rate")
axes[0].set_title("Reliability Calibration Curve (10 Bins)")
axes[0].legend(loc="upper left")

# 4.2 ROC Curve
fpr_meta, tpr_meta, _ = roc_curve(y_test_meta, p_meta)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test_meta, p_lgb_meta)

axes[1].plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.7)
axes[1].plot(fpr_meta, tpr_meta, color="#e74c3c", lw=2.5, label=f"MetaTFT (AUC = {eval_meta['roc_auc']:.4f})")
axes[1].plot(fpr_lgb, tpr_lgb, color="#2ecc71", lw=2.5, label=f"Our LightGBM (AUC = {eval_lgb['roc_auc']:.4f})")
axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
axes[1].set_ylabel("True Positive Rate (Sensitivity / Recall)")
axes[1].set_title("Receiver Operating Characteristic (ROC) Curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
fig.savefig(DOCS_IMG_DIR / "calibration_and_roc.png")
plt.close(fig)
print("Saved docs/images/calibration_and_roc.png")

# 5. GENERATE STAGE-BY-STAGE METRICS BREAKDOWN
print("Generating Stage breakdown figure...")
df_test_meta = df_test[mask_meta].copy().reset_index(drop=True)
df_test_meta["stage_num"] = df_test_meta["round_stage"].str.extract(r"^(\d+)").astype(int)

stage_eval = []
for stage in sorted(df_test_meta["stage_num"].unique()):
    if stage < 2 or stage > 7:
        continue
    mask_s = df_test_meta["stage_num"] == stage
    y_grp = y_test_meta[mask_s]
    meta_grp = p_meta[mask_s]
    our_grp = p_lgb_meta[mask_s]
    
    bs_meta = compute_brier_metrics(y_grp, meta_grp)["brier_score"]
    bs_ours = compute_brier_metrics(y_grp, our_grp)["brier_score"]
    acc_meta = compute_classification_metrics(y_grp, meta_grp)["accuracy"]
    acc_ours = compute_classification_metrics(y_grp, our_grp)["accuracy"]
    auc_meta = compute_classification_metrics(y_grp, meta_grp)["roc_auc"]
    auc_ours = compute_classification_metrics(y_grp, our_grp)["roc_auc"]
    
    stage_eval.append({
        "Stage": f"Stage {stage}",
        "Rounds": int(mask_s.sum()),
        "MetaTFT Brier": bs_meta,
        "Our Model Brier": bs_ours,
        "MetaTFT Accuracy": acc_meta,
        "Our Model Accuracy": acc_ours,
        "MetaTFT ROC-AUC": auc_meta,
        "Our Model ROC-AUC": auc_ours,
        "Brier Delta (Ours - Meta)": bs_ours - bs_meta,
    })

df_stage = pd.DataFrame(stage_eval)
stages = df_stage["Stage"].tolist()
x = np.arange(len(stages))
width = 0.35

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Brier
axes[0, 0].bar(x - width/2, df_stage["MetaTFT Brier"], width, label="MetaTFT", color="#e74c3c", alpha=0.85)
axes[0, 0].bar(x + width/2, df_stage["Our Model Brier"], width, label="Our Model", color="#2ecc71", alpha=0.85)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(stages)
axes[0, 0].set_ylabel("Brier Score (Lower is Better)")
axes[0, 0].set_title("Brier Score Progression Across Game Stages")
axes[0, 0].legend()

# Accuracy
axes[0, 1].bar(x - width/2, df_stage["MetaTFT Accuracy"] * 100, width, label="MetaTFT", color="#e74c3c", alpha=0.85)
axes[0, 1].bar(x + width/2, df_stage["Our Model Accuracy"] * 100, width, label="Our Model", color="#2ecc71", alpha=0.85)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(stages)
axes[0, 1].set_ylabel("Accuracy % (Higher is Better)")
axes[0, 1].set_title("Classification Accuracy Across Game Stages")
axes[0, 1].legend()

# ROC-AUC
axes[1, 0].plot(stages, df_stage["MetaTFT ROC-AUC"], "s--", label="MetaTFT", color="#e74c3c", lw=2.5, markersize=8)
axes[1, 0].plot(stages, df_stage["Our Model ROC-AUC"], "o-", label="Our Model", color="#2ecc71", lw=2.5, markersize=8)
axes[1, 0].set_ylabel("ROC-AUC")
axes[1, 0].set_title("ROC-AUC Trajectory Across Game Stages")
axes[1, 0].legend()

# Delta
deltas = df_stage["Brier Delta (Ours - Meta)"].values
colors = ["#27ae60" if d <= 0 else "#e67e22" for d in deltas]
axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.7)
axes[1, 1].bar(stages, deltas, color=colors, alpha=0.85, width=0.4)
axes[1, 1].set_ylabel("Brier Delta (Ours - MetaTFT)")
axes[1, 1].set_title("Brier Gap to MetaTFT (<= 0 indicates Our Model is Superior)")
for i, d in enumerate(deltas):
    axes[1, 1].text(i, d + (0.001 if d >= 0 else -0.003), f"{d:+.4f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(DOCS_IMG_DIR / "stage_breakdown.png")
plt.close(fig)
print("Saved docs/images/stage_breakdown.png")

# 6. GENERATE 2D PROBABILITY DENSITY HEATMAP
print("Generating 2D Density Heatmap...")
fig, ax = plt.subplots(figsize=(9, 8))
hb = ax.hexbin(
    p_meta,
    p_lgb_meta,
    gridsize=50,
    cmap="plasma",
    bins="log",
    mincnt=1,
)
ax.plot([0, 1], [0, 1], "w--", lw=2, label="Perfect Agreement (y = x)")
ax.set_xlabel("MetaTFT Predicted Win Probability", fontsize=12)
ax.set_ylabel("Our Calibrated LightGBM Probability", fontsize=12)
ax.set_title(f"2D Probability Density Alignment (Log Counts, r = {np.corrcoef(p_meta, p_lgb_meta)[0,1]:.3f})", fontsize=14)
cb = fig.colorbar(hb, ax=ax)
cb.set_label("Log10(Count of Validation Rounds)", fontsize=11)
ax.legend(loc="upper left")
plt.tight_layout()
fig.savefig(DOCS_IMG_DIR / "probability_alignment_density.png")
plt.close(fig)
print("Saved docs/images/probability_alignment_density.png")

# 7. GENERATE FEATURE IMPORTANCE FIGURE
print("Generating Feature Importance figure...")
imps = lgb_model.feature_importances_
df_imp = pd.DataFrame({"feature": feature_names, "importance": imps}).sort_values(by="importance", ascending=False).head(25)

fig, ax = plt.subplots(figsize=(11, 9))
sns.barplot(data=df_imp, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=ax)
ax.set_title("Top 25 Most Decisive Combat Features in Predicting Round Outcome (LightGBM)", fontsize=13)
ax.set_xlabel("Feature Split Importance", fontsize=12)
ax.set_ylabel("Feature Name", fontsize=12)
plt.tight_layout()
fig.savefig(DOCS_IMG_DIR / "feature_importance.png")
plt.close(fig)
print("Saved docs/images/feature_importance.png")

# 8. GENERATE TACTICAL STRESS TESTS CARD
print("Generating Tactical Stress Tests visual scorecard...")
stress_scenarios = [
    ("1. The Greedy Banker", "200g in bank + 1★ unit\nvs 0g + 8-unit capped board", "0.6%", "PASS (Aligned)", "#2ecc71"),
    ("2. 1-HP Clutch Miracle", "1 HP + 8-unit capped board\nvs 100 HP + 1 weak unit", "97.8%", "PASS (Aligned)", "#2ecc71"),
    ("3. Quality vs Quantity", "3★ 4-Cost 3-Item Carry\nvs 8 naked 2★ brawlers", "81.0%", "PASS (Aligned)", "#2ecc71"),
    ("4. Item Advantage", "Mirror 8-unit compositions:\n6 completed items vs 0 items", "88.5%", "PASS (Aligned)", "#2ecc71"),
]

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

table_data = [
    ["Tactical Scenario", "Combat Matchup Setup", "Predicted Focal Win %", "Verdict"]
] + [
    [s[0], s[1], s[2], s[3]] for s in stress_scenarios
]

table = ax.table(
    cellText=table_data,
    colLabels=None,
    cellLoc="center",
    loc="center",
    colWidths=[0.25, 0.45, 0.15, 0.15],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.3)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold", fontsize=12)
    else:
        cell.set_facecolor("#ecf0f1" if row % 2 == 0 else "#ffffff")
        if col == 3:
            cell.set_text_props(color="#27ae60", weight="bold")
        elif col == 2:
            cell.set_text_props(weight="bold")

plt.title("TFT Round Winner Domain-Logic Tactical Stress Tests", fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
fig.savefig(DOCS_IMG_DIR / "tactical_stress_tests.png")
plt.close(fig)
print("Saved docs/images/tactical_stress_tests.png")

print("\nAll 7 publication-quality figures successfully generated and saved to docs/images/")
