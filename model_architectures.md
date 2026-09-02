# Model Architectures & Neural System Specification

This document provides the technical specification of the two machine learning systems powering the TFT AI Player:
1. **Combat Round Winner & Damage Estimation System:** Fast tabular probabilistic battle outcome and player damage predictor.
2. **Multi-Modal State Representation Trunk & Embeddings:** Robust, simplified PyTorch Transformer representation learning system producing dense, permutation-invariant embeddings for the Reinforcement Learning agent.

---

## 1. Combat Round Winner & Damage Estimation System

### 1.1 Purpose & Role in Architecture
During both offline simulation (rollout simulation) and live game evaluation, computing full combat physics for 8 simultaneous players is computationally prohibitive. The Combat Round Winner model serves as a **calibrated, high-throughput probabilistic oracle** ($>100,000 \text{ evaluations/sec}$) that predicts:
1. **Round Victory Probability:** $P(\text{Focal Player Wins} \mid \mathbf{x}_{\text{focal}}, \mathbf{x}_{\text{opp}}) \in [0.0, 1.0]$
2. **Damage Magnitude (HP Loss):** $\mathbb{E}[\Delta \text{HP} \mid \text{Defeat}] \in [1, 45]$

```
                      Focal Board State               Opponent Board State
                    (Units, Items, Traits)           (Units, Items, Traits)
                               │                               │
                               └───────────────┬───────────────┘
                                               ▼
                              TFTBoardFeatureExtractor (1,107 dims)
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
          LightGBM Classifier + Platt Scaling            LGBM Damage Regressor
                       │                                               │
                       ▼                                               ▼
             P(Victory) ∈ [0.0, 1.0]                     Expected Damage Loss (HP)
```

---

### 1.2 Model Specifications

* **Package Location:** [`src/tft_ai_player/round_winner/`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/round_winner/)
* **Classifier:** Calibrated Gradient Boosted Decision Trees ([`LightGBM`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/round_winner/pipeline.py) / [`HistGradientBoostingClassifier`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/round_winner/trainer.py)) optimized for multi-class trait synergy density and combinatorial item interactions.
* **Probability Calibrator:** Platt Scaling / Isotonic Logistic Regression mapping raw tree margin outputs into well-calibrated posterior probabilities minimizing Brier score.
* **Damage Regressor:** Gradient Boosted Regressor ([`LGBMRegressor`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/round_winner/trainer.py)) predicting empirical HP loss ($[1, 45]$ HP) conditioned on defeat.

---

### 1.3 Empirical Results & Benchmark Comparison

Trained on 330k rounds and evaluated on 65,980 held-out high-Elo test rounds:

| Metric | Calibrated LightGBM (Ours) | MetaTFT Proprietary Baseline | Improvement |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.8503** | 0.8348 | **+1.55%** |
| **Classification Accuracy** | **76.53%** | 75.43% | **+1.10%** |
| **F1 Score** | **0.7893** | 0.7739 | **+1.54%** |
| **Precision** | **0.7720** | 0.7584 | **+1.36%** |
| **Recall** | **0.8074** | 0.7899 | **+1.75%** |
| **Brier Score (lower is better)** | **0.1577** | 0.1659 | **-0.0082** |
| **Brier Skill Score (BSS)** | **0.3640** | 0.3335 | **+9.15%** |
| **Log Loss (lower is better)** | **0.4788** | 0.5318 | **-0.0530** |
| **Damage Regressor MAE** | **1.657 HP** | — | — |

---

## 2. Multi-Modal State Representation Trunk & Embeddings

### 2.1 Overview & Architecture
The Multi-Modal Fusion Trunk ([`src/tft_ai_player/embeddings/`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/)) uses standard, native PyTorch modules (`nn.TransformerEncoder`, `nn.Embedding`, `nn.Linear`, `nn.LayerNorm`) for rock-solid stability and zero runtime crashes.

```
Snapshot Observation s_t
  ├── Friendly Board (4x7 Grid) ──> Champ2Vec (32D) ──> BoardTransformer (256D) ──┐
  │                                   (Zero-Gated)        (Masked Mean Pool)       │
  │                                 TraitEncoder (64D) ─────────────────────────┘  │
  │                                                                                ▼
  └── Economy & State Scalars   ──────────────────────> StateMLP (64D) ─────> Fusion Trunk (320D)
                                                                                   │
                                                                                   ▼
                                                                            fused_state (320D)
                                                                              ├── [Macro Head] ───> Top-4 Cross-Entropy Loss (λ=1.0)
                                                                              └── [Micro Head] ───> PVP Combat Win BCE Loss (λ=0.5)
  [Isolated Board Feature (256D)] ──────────────────────────────────────────────> [Flow Head] ────> Time-Contrastive InfoNCE (λ=0.15)
```

---

### 2.2 Core Components

#### 1. Permutation-Invariant Unit Representation: [`Champ2Vec`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L38)
* **Embeddings:** Champion ID (`32D`), Star Level (`8D`), Sum-Pooled Items (`16D`).
* **Fusion:** `Linear(56, 32) -> LayerNorm(32) -> ReLU()`.
* **Zero-Gating:** Multiplying output by `(champ_ids > 0)` guarantees exact `0.0` for empty hexes.

#### 2. Board Transformer Encoder: [`BoardHexTransformer`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L118)
* **Unit Projection:** `Linear(32, 128) + PositionalEmbeddings(1, 28, 128)`.
* **Standard PyTorch Transformer:** 2 layers of `nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, activation='relu', batch_first=True)`.
* **Masked Mean Pooling:** Mean-pools active unit tokens down to `128D`.
* **Trait Synergy Fusion:** Fuses pooled units (`128D`) + TraitEncoder (`64D`) $\to$ `Linear(192, 256) -> LayerNorm(256) -> ReLU()` = `board_feat` (`256D`).

#### 3. State MLP: [`StateMLP`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L182)
* 8 continuous scalars (HP, Gold, Level, Streak, Stage Num, Round in Stage, Unit Count, Item Count) $\to$ `Linear(8, 64) -> LayerNorm(64) -> ReLU()` = `state_feat` (`64D`).

#### 4. Multi-Modal Fusion Trunk: [`MultiModalFusionTrunk`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L196)
* Fuses `board_feat` (`256D`) + `state_feat` (`64D`) = `320D` $\to$ `Linear(320, 320) -> LayerNorm(320) -> ReLU() -> Linear(320, 320)` = `fused_state` (`320D`).

#### 5. Tri-Objective Pre-training Heads: [`TrunkPretrainModel`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L286)
* **Macro:** `Linear(320, 64) -> ReLU() -> Linear(64, 2)` (Top-4 Cross-Entropy, $\lambda_1 = 1.0$).
* **Micro:** `Linear(320, 64) -> ReLU() -> Linear(64, 1)` (PVP Combat Win Probability BCE, $\lambda_2 = 0.5$, masked on PVP rounds).
* **Flow:** `Linear(256, 128) -> ReLU() -> Linear(128, 128)` (Time-Contrastive InfoNCE on isolated `board_feat`, $\lambda_3 = 0.15$, $\tau = 0.07$).

---

### 2.3 Training & Execution

* **Match Grouped Partitioning:** Training/Validation split partitioned strictly by `match_id` to prevent temporal data leakage.
* **Gradient Clipping:** Strict `clip_grad_norm_(max_norm=1.0)` prevents gradient spikes and numerical instability.

```powershell
uv run python -m tft_ai_player.embeddings.train `
    --data-dir "D:\tft-winner-data\set18\players" `
    --epochs 10 `
    --batch-size 128 `
    --device cuda `
    --lr 1e-3 `
    --val-weight 1.0 `
    --micro-weight 0.5 `
    --contrast-weight 0.15 `
    --log-interval 10 `
    --wandb-project "tft-embeddings" `
    --run-name "trunk_robust_v1" `
    --output-dir "D:\tft-winner-data\set18\models\trunk"
```
