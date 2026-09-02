# Model Architectures & Neural System Specification

This document provides the technical specification of the core machine learning and representation systems powering the TFT AI Player:
1. **Combat Round Winner & Damage Estimation System:** Fast tabular probabilistic battle outcome and player damage predictor.
2. **Multi-Modal State Representation Trunk & Embeddings:** Robust PyTorch Transformer representation learning system producing dense, permutation-invariant embeddings.
3. **Composition Archetype Extraction & Latent Clustering (Z-Index):** Unsupervised latent geometric clustering distilling meta board archetypes into canonical reference centroids ($Z$-Index) to anchor goal-directed strategic decision making.

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

#### 1. Permutation-Invariant Unit Representation: [`Champ2Vec`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L33)
* **Embeddings:** Champion ID (`32D`), Star Level (`8D`), Sum-Pooled Items (`16D`).
* **Fusion:** `Linear(56, 32) -> LayerNorm(32) -> ReLU()`.
* **Zero-Gating:** Multiplying output by `(champ_ids > 0)` guarantees exact `0.0` for empty hexes.

#### 2. Board Transformer Encoder: [`BoardHexTransformer`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L153)
* **Unit Projection:** `Linear(32, 128) + PositionalEmbeddings(1, 28, 128)`.
* **Standard PyTorch Transformer:** 2 layers of `nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, activation='relu', batch_first=True)`.
* **Masked Mean Pooling:** Mean-pools active unit tokens down to `128D`.
* **Trait Synergy Fusion:** Fuses pooled units (`128D`) + TraitEncoder (`64D`) $\to$ `Linear(192, 256) -> LayerNorm(256) -> ReLU()` = `board_feat` (`256D`).

#### 3. State MLP: [`StateMLP`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L246)
* 8 continuous scalars (HP, Gold, Level, Streak, Stage Num, Round in Stage, Unit Count, Item Count) $\to$ `Linear(8, 64) -> LayerNorm(64) -> ReLU()` = `state_feat` (`64D`).

#### 4. Multi-Modal Fusion Trunk: [`MultiModalFusionTrunk`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L261)
* Fuses `board_feat` (`256D`) + `state_feat` (`64D`) = `320D` $\to$ `Linear(320, 320) -> LayerNorm(320) -> ReLU() -> Linear(320, 320)` = `fused_state` (`320D`).

#### 5. Tri-Objective Pre-training Heads: [`TrunkPretrainModel`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L407)
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

---

## 3. Composition Archetype Extraction & Latent Clustering (Z-Index)

### 3.1 Overview & Motivation: The Latent "Compass"
In complex auto-battlers like Teamfight Tactics, the combinatorial state and action space is astronomical. Asking a downstream decision-making agent to discover viable competitive endgame compositions purely via random exploration requires millions of compute hours and often results in catastrophic policy collapse or suboptimal local minima.

This phase establishes the **critical mathematical bridge between Deep Learning representation and strategic execution**:
* Instead of unstructured exploration, we leverage the dense, contrastively pre-trained geometric latent space of the [`BoardHexTransformer`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L153).
* We distill the canonical meta compositions into a discrete set of geometric centroids—the **$Z$-Index** ($Z = \{z_1, z_2, \dots, z_K\} \subset \mathbb{R}^{256}$).
* Each centroid $z_k$ serves as a **mathematical compass**, defining a specific high-tier composition archetype that downstream systems can target and navigate toward.

```
       Canonical Top-4 End-Game Boards (Stage ≥ 5-1)
                             │
                             ▼
             Pre-Trained MultiModalFusionTrunk
               (Frozen via .freeze() / Eval)
                             │
              Extract Isolated board_feat (256D)
              (Champ2Vec + BoardHexTransformer)
               [Bypassing StateMLP / Economy]
                             │
                             ▼
              Clustering Engine (K-Means / HDBSCAN)
                             │
                             ▼
     ┌─────────────────────────────────────────────────┐
     │           Z-Index Centroid Set Z                │
     │   Z = {z_1, z_2, ..., z_K} ∈ R^(K x 256)        │
     └───────────────────────┬─────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   WandB Visualizations & Metrics     Exported Z-Index Artifacts
   (UMAP / t-SNE Projections,         (z_index.pt,
    Silhouette, Archetype Profiles)    cluster_profiles.json)
```

---

### 3.2 Target Dataset Curation (Defining the End-State)
To discover authentic meta compositions, early-game and mid-game transition boards must be strictly filtered out:
* **The Transition Noise Problem:** Boards from Stages 2 to 4 frequently contain placeholder item-holders, partial 1-star pairs on bench/board, and incomplete synergy transitions that dilute composition semantics.
* **Top-4 Filtering Criterion:** Only final boards from players placing in the **Top 4** ($\text{rank} \le 4$) in high-Elo matches are retained.
* **Stage Threshold ($\ge 5\text{-}1$):** By Stage 5-1 and beyond, surviving competitive players have stabilized their primary carries, completed 3-item cores, and activated their capstone trait thresholds.
* This curation guarantees that clustered points represent fully realized, canonical winning board states.

---

### 3.3 Isolated Latent Extraction (`board_feat` 256D)
Feature extraction is performed by querying only the spatial board encoding pathway of the pre-trained trunk:

1. **Model Freezing:** The pre-trained [`MultiModalFusionTrunk`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L261) is loaded from checkpoint and placed into frozen evaluation mode via [`.freeze()`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L364).
2. **Dedicated Extraction Pathway:** Each curated board snapshot is passed into [`encode_board()`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L321):
   $$\mathbf{h}_{\text{board}} = \text{BoardHexTransformer}(\text{Champ2Vec}(\text{Tokens}), \text{TraitVec}) \in \mathbb{R}^{256}$$
3. **Intentional Omission of State Scalars ([`StateMLP`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L246)):
   * Player economy (Gold, Level, Streak) and HP scalars are **deliberately bypassed**.
   * **Theoretical Justification:** A specific strategic composition (e.g., *Fast-8 4-Cost Caster Flex* or *Vertical Rebel Reroll*) maintains an identical board identity whether the player has 50 gold or 0 gold, 90 HP or 12 HP.
   * Including scalar state features would conflate economic fortune with unit/trait synergy topology.

---

### 3.4 Clustering & Z-Index Formulation
Unsupervised geometric partitioning is applied to the $N$ extracted 256D latent vectors:

* **Clustering Algorithm:** K-Means (or density-based HDBSCAN) partitions the continuous 256D space into $K$ distinct clusters by minimizing within-cluster variance:
  $$\min_{\{z_1, \dots, z_K\}} \sum_{i=1}^N \min_{k \in \{1, \dots, K\}} \|\mathbf{h}_{\text{board}}^{(i)} - z_k\|_2^2$$
* **The Z-Index Definition:** The resulting set of $K$ cluster centroids forms the **$Z$-Index**:
  $$Z = \{z_1, z_2, \dots, z_K\}, \quad z_k \in \mathbb{R}^{256}$$
* **Archetype Manifold Projection:** Each vector $z_k$ represents the mathematical distillation of a specific TFT composition archetype. Any arbitrary board state $\mathbf{h}_{\text{board}}^t$ can be scored against archetype $k$ via cosine similarity:
  $$\text{CosineSimilarity}(\mathbf{h}_{\text{board}}^t, z_k) = \frac{\mathbf{h}_{\text{board}}^t \cdot z_k}{\|\mathbf{h}_{\text{board}}^t\|_2 \, \|z_k\|_2}$$

---

### 3.5 Evaluation, Archetype Profiling & WandB Integration
To make evaluation clear, interpretable, and reproducible, the clustering pipeline automatically logs quantitative metrics, archetype profiles, and dimensional reduction plots to **Weights & Biases (WandB)**:

#### 1. Quantitative Clustering Metrics
* **Silhouette Score:** Quantifies intra-cluster cohesion versus nearest-cluster separation ($[-1.0, +1.0]$).
* **Davies-Bouldin Index:** Measures average similarity between each cluster and its most similar counterpart (lower indicates superior clustering).
* **Calinski-Harabasz Score:** Ratio of between-cluster dispersion to within-cluster dispersion.
* **Inertia & Elbow Sweep:** Sweeping $K \in [6, 25]$ to identify the optimal number of meta archetypes.

#### 2. Dimensionality Reduction & WandB Visualizations
* **UMAP & t-SNE 2D/3D Scatter Plots:** High-dimensional board embeddings projected into 2D/3D coordinate space, color-coded by archetype cluster ID, and logged as interactive WandB plots.
* **Centroid Distance Heatmap:** Pairwise cosine distance matrix between all $K$ centroids logged to verify diversity across archetypes.

#### 3. Archetype Interpretability & Profiling
For each centroid $z_k$, the system generates an archetype profile summary:
* **Top Champions & Star Levels:** Frequency distribution of units appearing on boards belonging to cluster $k$.
* **Core Item Trios:** Most frequent 3-item combinations equipped on primary carries/tanks.
* **Dominant Trait Tiers:** Activated trait breakpoints (e.g. 6 Invoker / 4 Preserver).
* **Nearest Canonical Boards:** The top real game boards closest to centroid $z_k$ in Euclidean and cosine distance.

#### 4. Exported Artifacts
* `z_index.pt`: Serialized PyTorch dictionary containing the centroid matrix $Z \in \mathbb{R}^{K \times 256}$ and normalizer constants.
* `cluster_profiles.json`: Machine-readable mapping of cluster IDs to archetype names, trait distributions, and champion frequencies.

---

### 3.6 Execution Pipeline

```powershell
uv run python -m tft_ai_player.embeddings.cluster `
    --data-dir "D:\tft-winner-data\set18\players" `
    --trunk-checkpoint "D:\tft-winner-data\set18\models\trunk\best_model.pt" `
    --min-stage 5 `
    --min-placement 4 `
    --n-clusters 15 `
    --device cuda `
    --wandb-project "tft-clustering" `
    --run-name "z_index_k15_set18" `
    --output-dir "models/clustering"
```
