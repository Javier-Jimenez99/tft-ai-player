# Model Architectures & Neural System Specification

This document provides the technical specification of the core machine learning and representation systems powering the TFT AI Player:
1. **Combat Round Winner & Damage Estimation System:** Fast tabular probabilistic battle outcome and player damage predictor.
2. **Multi-Modal State Representation Trunk & Embeddings:** Robust PyTorch Transformer representation learning system producing dense, permutation-invariant embeddings.
3. **Composition Archetype Extraction & Latent Clustering (Z-Index):** Unsupervised latent geometric clustering distilling meta board archetypes into canonical reference centroids ($Z$-Index) to anchor goal-directed strategic decision making.
4. **State Transition Predictor (World Model):** Latent macro-trajectory dynamics model.
5. **Reinforcement Learning & AlphaStar League Pipeline (v6 Baseline):** Multi-agent self-play league and prioritized fictitious play.
6. **AlphaStar v6.1: Trace-Driven Shadow Match Reinforcement Learning:** PPO fine-tuning directly embedded in real-world high-Elo human matches.
7. **AlphaStar v6.2: Adaptive Ranked Ladder RL (Curriculum MMR Progression):** Multi-tier competitive climbing from Oro to Challenger with empirical survival tables and LP rewards.

---

## 0. Model Versioning & Pipeline Nomenclature Standard

To ensure absolute traceability across multi-phase training regimes, the project adheres to a strict hierarchical semantic versioning convention:

$$\mathbf{v}\langle \text{Phase 1: Foundation / Self-Play} \rangle \,.\, \langle \text{Phase 2: Human Trace / Ranked Ladder} \rangle \,.\, \langle \text{Phase 3: Superhuman League / Self-Play Hardening} \rangle$$

### Version Breakdown:
1. **Major Version (`vX` — Phase 1: Foundation Gym / Self-Play League):**
   * Represents the fundamental agent architecture, state encoding trunk, action space factorizations, and the multi-agent Self-Play baseline.
   * *Example:* **`v6`** (AlphaStar Multi-Agent League with 15 Exploiters, Tree-Search Lookahead Planner, and Z-Centroid conditioning).
2. **Minor Version (`vX.Y` — Phase 2: Human Trace-Driven / Ranked Ladder branches):**
   * Represents parallel or distinct methodology branches grounded in real human matches, **all initialized directly from the base vX checkpoint** (`v6/gen_0300`).
   * *Examples:*
     * **`v6.1`**: Shadow Match RL directly embedded against high-Elo Challenger match traces (branched from `v6`).
     * **`v6.2`**: Adaptive Ranked Ladder RL (dynamic MMR/ELO system climbing through Bronze $\to$ Challenger with league-specific empirical round death distributions and LP rewards, branched from `v6`).
3. **Patch Version (`vX.Y.Z` — Phase 3: Superhuman Hardening / Full Self-Play):**
   * Represents post-human self-play specialization, where an agent that has mastered human play plays against copies of itself to surpass human meta ceilings.
   * *Example:* **`v6.2.1`** (Full Self-Play League initialized from the `v6.2` Master/Challenger-ranked policy to discover strategies beyond human play).

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

---

## 4. State Transition Predictor (World Model)

### 4.1 Overview & Motivation
The **State Transition Predictor** acts as an offline behavioral cloning and macro-trajectory world model. Rather than forcing downstream decision-making agents to explore unbounded state transitions blindly, we map strategic round-to-round planning ($s_t \to \hat{s}_{t+1}$) directly within the 320D latent manifold of the frozen [`MultiModalFusionTrunk`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L261).

```
Current Round State s_t (320D)
              │
              ▼
   Linear(320, 512) -> LayerNorm -> ReLU -> Dropout
              │
              ▼
   Residual MLP Blocks (3 Layers x 512D)
              │
              ▼
   Linear(512, 320) -> Residual Delta Δs_t
              │
              ▼
   Predicted Next State: ŝ_{t+1} = s_t + Δs_t (320D)
```

* **Deep Residual MLP:** Preserves core board identity via residual delta mapping ($\hat{s}_{t+1} = s_t + \Delta s_t$).
* **Dual-Objective Loss:** $\mathcal{L}_{\text{Transition}} = \mathcal{L}_{\text{Huber}}(\hat{s}_{t+1}, s_{t+1}) + \lambda \cdot \mathcal{L}_{\text{Cosine}}(\hat{s}_{t+1}, s_{t+1})$.

---

## 5. Reinforcement Learning & AlphaStar League Pipeline

### 5.1 System Architecture & Model Interaction Topology

The training loop orchestrates five specialized modules across offline guidance, physics simulation, and policy optimization:

```
                                      ROUND PLANNING PHASE (t)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. Initial State Extraction                                                                  │
 │    • Board + Scalars ──> Frozen MultiModalFusionTrunk ───────────────────────> s_t (320D)    │
 │    • 5 Shop Slots    ──> Frozen Champ2Vec + Linear(160, 64) ─────────────────> shop_feat     │
 │    • 9 Bench Slots   ──> Frozen Champ2Vec + Permutation Invariant Pooling ───> bench_feat    │
 │    • Policy Condition: Exploiters receive Target z_k (256D); Main receives 0_256             │
 ├──────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 2. Strategic Oracle Projection                                                               │
 │    • Initial State s_t ──> Frozen World Model (Transition MLP) ──────────────> ŝ_{t+1} (320D)│
 ├──────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 3. Sequential Decision Execution (Gymnasium Loop)                                            │
 │    • Observation: [s_t (320D), shop_feat (64D), bench_feat (64D), z_target (256D)] (704D)    │
 │    • Action Mask: Pre-Softmax pruning of illegal transactions (-1e9 logit penalty)           │
 │    • PPO Actor selects discrete action a_k ∈ {0..110} until `PASS_ROUND` (0) emitted        │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                        COMBAT RESOLUTION
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 4. Tabular Combat Oracle (LightGBM)                                                          │
 │    • Focal Board + Matched Opponent Board ──> TFTBoardFeatureExtractor (1,107 dims)          │
 │    • Classifier ──────> P(Victory) ∈ [0.0, 1.0] via Platt Scaling                            │
 │    • Regressor  ──────> E[ΔHP | Defeat] ∈ [1, 45]                                            │
 │    • Environment updates Player HP, Gold interest, Win/Loss streaks, and victory gold        │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                     ROUND EVALUATION PHASE (t+1)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 5. End-of-Round Transition & Multi-Objective Reward                                          │
 │    • Realized Post-Combat State ──> Frozen MultiModalFusionTrunk ────────────> s_{t+1} (320D)│
 │    • Micro Macro Alignment: CosineSimilarity(s_{t+1}, ŝ_{t+1})                               │
 │    • Specialist Comps Alignment: CosineSimilarity(h_{board}^{t+1}, z_k)                      │
 │    • Game Victory / Survival: ΔHP, Gold economy, Top-4 threshold                             │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 5.2 Gymnasium Decision MDP & Action Masking

#### State Representation ($o_t \in \mathbb{R}^{704}$)
1. **Core State ($320\text{D}$):** Output of frozen [`MultiModalFusionTrunk.encode_board()`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L321) ($256\text{D}$) combined with [`StateMLP`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L246) ($64\text{D}$).
2. **Shop State ($64\text{D}$):** 5 shop units pass through frozen `Champ2Vec` ($5 \times 32 = 160\text{D}$) and project through `Linear(160, 64) -> LayerNorm(64) -> ReLU()`.
3. **Bench State ($64\text{D}$):** 9 bench slots pass through frozen `Champ2Vec` ($9 \times 32\text{D}$). An order-invariant DeepSets reduction (Sum-Pool + Mean-Pool) projects through `Linear(64, 64) -> LayerNorm(64) -> ReLU()`.
4. **Target Conditioning ($256\text{D}$):** For Exploiter agents, assigned Z-Index centroid $z_k$. For Main Agent, strictly $\mathbf{0}_{256}$.

#### Factorized Discrete Action Space ($|\mathcal{A}| = 111$)

| Action Index | Category | Semantic Description |
| :--- | :--- | :--- |
| `0` | **`PASS_ROUND`** | Finalizes planning phase; triggers combat resolution. |
| `1 .. 5` | **`BUY_SHOP_SLOT`** | Purchases unit from shop index $0 \dots 4$. |
| `6` | **`REROLL_SHOP`** | Spends 2 gold to refresh shop offers. |
| `7` | **`BUY_XP`** | Spends 4 gold to purchase 4 experience points. |
| `8 .. 16` | **`SELL_BENCH`** | Sells unit residing in bench slot $0 \dots 8$. |
| `17 .. 44` | **`SELL_BOARD`** | Sells unit residing on board hex $0 \dots 27$. |
| `45 .. 72` | **`DEPLOY_UNIT`** | Moves targeted bench unit to board hex $0 \dots 27$. |
| `73 .. 100` | **`MOVE_BOARD`** | Swaps positions between targeted hex and hex $0 \dots 27$. |
| `101 .. 110` | **`EQUIP_ITEM`** | Equips targeted item component/completed item to unit. |

#### Strict Pre-Softmax Action Masking
$$\pi(a \mid o_t) = \text{Softmax}(\mathbf{z}_{\text{logits}} + (1 - \mathbf{m}) \cdot (-10^{9}))$$

---

### 5.3 Multi-Objective Reward Formulation

$$R_{\text{step}} = R_{\text{env}} + \alpha \cdot R_{\text{macro}} + \beta \cdot R_{\text{micro}}$$

1. **Environment Reward ($R_{\text{env}}$):**
   * $R_{\text{combat}} = +0.5$ for PVP win, $-0.02 \times \Delta \text{HP}$ for defeat.
   * $R_{\text{interest}} = +0.05$ per round if gold is maintained at $\ge 50\text{g}$.
   * $R_{\text{terminal}} = +2.0$ (Rank 1), $+1.0$ (Rank 2..4), $-1.0$ (Rank 5..6), $-2.0$ (Rank 7..8).
2. **Micro Alignment ($R_{\text{micro}}$):** $\text{CosineSimilarity}(s_{t+1}, \hat{s}_{t+1})$.
3. **Macro Alignment ($R_{\text{macro}}$):** $\text{CosineSimilarity}(\mathbf{h}_{\text{board}}^{t+1}, z_k)$.

| Policy Type | Macro Weight ($\alpha$) | Micro Weight ($\beta$) | Strategic Behavior |
| :--- | :---: | :---: | :--- |
| **Main Agent (Generalist)** | **$0.0$** | **$0.3$** | Optimizes global lobby survival using World Model guidance without composition bias. |
| **Exploiter Agent (Specialist)** | **$0.8$** | **$0.2$** | Highly penalized if diverting from target archetype $z_k$; masters execution of one line. |

---

### 5.4 PPO Actor-Critic Architecture

```
State Observation o_t (704D)
               │
               ▼
   Linear(704, 512) -> LayerNorm -> Orthogonal(gain=sqrt(2)) -> GELU
               │
               ▼
   Linear(512, 512) -> LayerNorm -> Orthogonal(gain=sqrt(2)) -> GELU
               │
       ┌───────┴────────────────────────────────────────┐
       ▼                                                ▼
  Actor Branch                                     Critic Branch
Linear(512, 256) -> GELU                         Linear(512, 256) -> GELU
       │                                                │
Linear(256, 111) -> Action Logits                Linear(256, 1) -> Value Estimate V(s)
       │
Action Mask Injection: logits.masked_fill(~mask, -1e9)
       │
Categorical Distribution Sampling
```

* **Clip Ratio ($\epsilon$):** $0.2$
* **Generalized Advantage Estimation ($\lambda_{\text{GAE}}$):** $0.95$
* **Discount Factor ($\gamma$):** $0.99$
* **Entropy Regularization ($c_1$):** $0.01 \to 0.001$
* **Value Loss Coefficient ($c_2$):** $0.5$
* **Max Gradient Norm:** $0.5$
* **Learning Rate:** $2.5 \times 10^{-4}$ with linear decay across iterations

---

### 5.5 AlphaStar League Engine & Prioritized Fictitious Self-Play (PFSP)

* **Matchmaker Distribution:** 35% Active Self-Play, 40% Exploiter Pool ($K=15$ specialists), 15% Historical Snapshots, 10% Deterministic Benchmark Bots.
* **PFSP Payoff Sampling:**
  $$P(\text{Match Opponent } j) = \frac{f(1 - M_{\text{Main}, j})}{\sum_{k} f(1 - M_{\text{Main}, k})}, \quad f(x) = x^2$$
* **Deterministic Benchmark Bots (Evaluated every 25 generations across 100 matches):**
  * **Bot Alpha (Standard 4-Cost Fast-8):** Rushes level 8 at Stage 4-2; plays standard front-to-back meta comps.
  * **Bot Beta (Hyperroll 1-Cost):** Spends all economy at Stage 3-1 to 3-star early game units.
  * **Bot Gamma (Greedy Economy Open-Fort):** Holds 50g at all costs to push fast level 9.

---

### 5.6 Diagnostics, Telemetry & Policy Collapse Detection

#### The "Principal" Dashboard (5 Essential Plots)

| Plot Name | Metric Logged | Normal Target Range | Interpretation / Health Signal | Kill / Adjustment Rule |
| :--- | :--- | :---: | :--- | :--- |
| **`Principal/1_Average_Placement`** | `avg_placement` / `eval_avg_placement` | $4.5 \to \le 3.5$ | **Overall Performance:** Tracks tournament finish rank ($1.0 = 1\text{st}, 8.0 = 8\text{th}$). If trending downward, the agent is legitimately improving. | If flat at $4.5$ or drifting $\ge 5.5$ for $>100$ gens: check action masking and reward scaling. |
| **`Principal/2_Critic_Explained_Variance`** | `explained_variance` | $0.50 \to 0.95$ | **Critic Value Accuracy:** Measures how accurately the value network predicts discounted returns ($1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$). | **KILL / ROLLBACK if $< 0.0$** for $>5$ consecutive iterations. Value network is corrupted. |
| **`Principal/3_Policy_Entropy`** | `policy_entropy` | $2.5 \to 0.8$ | **Exploration Balance:** Action distribution randomness. Should gradually and smoothly decay. | **KILL if $< 0.15$ (Premature Collapse):** Agent is permanently frozen into a single repetitive action. If $>4.0$, agent is random. |
| **`Principal/4_Policy_Stability_KL`** | `approx_kl` | $0.005 \to 0.020$ | **PPO Step Stability:** Quantifies policy drift $D_{\text{KL}}(\pi_{\text{old}} \parallel \pi_{\text{new}})$ per batch. | **ADJUST if $> 0.05$ (Policy Explosion):** Triggers epoch early-stopping. If continuous, reduce learning rate by 50%. |
| **`Principal/5_Benchmark_WinRate`** | `(Alpha_WR + Beta_WR + Gamma_WR)/3` | $50\% \to 85\%$ | **Absolute Competence:** Win rate against fixed, deterministic benchmark bots (Fast-8, Hyperroll, Economy). | If $< 25\%$, the agent is not mastering standard fundamentals (economy thresholds & level tempo). |

#### Health Telemetry & Collapse Thresholds

| Metric Name | Normal Expected Range | Collapse Warning Threshold | Pathology / Failure Mode |
| :--- | :---: | :---: | :--- |
| **`policy_entropy`** | $2.5 \to 0.8$ | $< 0.15$ or $> 4.0$ | **Entropy Collapse:** Agent fixates prematurely on a single micro-action. |
| **`explained_variance`** | $0.50 \to 0.92$ | $< 0.0$ (Negative) | **Critic Failure:** Value network fails; advantages corrupted. |
| **`approx_kl`** | $0.005 \to 0.02$ | $> 0.05$ | **Policy Explosion:** Triggers early stopping for current PPO epoch. |
| **`action_mask_rejection_rate`** | $0.0\%$ (Strict) | $> 0.00001$ | **Masking Leak:** Invalid actions leaking past action mask. |
| **`macro_z_cosine`** | $0.65 \to 0.92$ | $< 0.40$ (Stage 5) | **Specialist Divergence:** Exploiter agent ignoring target archetype. |
| **`micro_world_model_cosine`** | $0.70 \to 0.88$ | $< 0.50$ | **Macro Regression:** Divergence from high-Elo human planning trajectory. |

> **Safety Rollback:** If `explained_variance` remains negative for $>5$ consecutive iterations, the learning rate drops by $50\%$ and policy weights roll back to the previous stable league generation.

---

### 5.7 Combat Prediction Engine: Deep Learning vs. LightGBM Benchmark

We profiled both surrogate combat engines across single 1v1 matchups, batched parallel environments, and the complete end-to-end RL training loop (RTX 3070 GPU vs. Multi-core CPU):

#### 1. End-to-End RL System Impact (512 Rollout Steps + 4 PPO Epochs)
* **Deep Learning (GPU Trunk):** **`6.47 s`** per generation (**`79.1 steps/s`**) — **33.8% faster total training time** (1.51x speedup).
* **LightGBM ML (1,107 Features):** **`9.77 s`** per generation (**`52.4 steps/s`**).
* **Pure Heuristic Baseline:** **`8.16 s`** per generation (**`62.7 steps/s`**).

#### 2. Microbenchmark Latency & Throughput Scaling
* **In-RL Latent Reuse:** Reusing the already computed $s_t / \mathbf{h}_{\text{board}}$ embeddings on GPU takes **`0.324 ms`** per match (**`3,089 matches/s`**), achieving a **`16.1x speedup`** over LightGBM ($5.20\text{ ms}$).
* **Batched Parallel Scaling ($N = 4,096$):** Deep Learning GPU processes **`162,040 matches/s`** vs LightGBM's **`3,565 matches/s`** (**`45.5x throughput gain`**).

---

## 6. AlphaStar v6.1: Trace-Driven Shadow Match Reinforcement Learning

### 6.1 Concept & Motivation
In standard self-play reinforcement learning (v6 baseline), the agent spars against copies of itself, exploiters, and historical league snapshots. While effective for learning combinatorial synergies, synthetic bots can develop non-human eccentricities or "meta bubbles" that differ from real human lobbies.

**AlphaStar v6.1 (Shadow Match RL)** solves this by embedding the PPO agent as a **"Shadow Player"** directly inside real, historical Challenger/Grandmaster matches:
* Rather than simulating 7 bot opponents, we load real chronological timeline trajectories from high-Elo human matches.
* In each round, the AI receives authentic shop rolls, player health states, and economic resources, and fights against the **actual opponent board** that the human player faced in that exact round.
* The agent makes its own decisions (buying, selling, positioning, leveling, rolling, equipping items), but combats are resolved against authentic human boards using the Deep Learning Combat Resolver.

```
                    Historical Challenger Match Trace (t = 1 .. T)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        Focal Player Seed Conditions          Opponent Board at Stage t
        (Starting Gold, Items, Level)         (Authentic Challenger Board)
                    │                                   │
                    ▼                                   ▼
      PPO Actor-Critic + Lookahead Planner         DL Combat Resolver
           (AlphaStar v6.1 Policy)             (GPU Siamese / Trunk Net)
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                         Victory / Damage Outcome
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        Focal Agent State Updates               PPO Rollout Buffer
        (HP, Economy, Streak, Bench)        (s_t, a_t, r_t, v_t, log_pi)
```

---

### 6.2 Shadow Environment Mechanics ([`ShadowMatchEnv`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/rl/shadow_match/shadow_env.py))

1. **Replay Repository & Trace Sampling:**
   * Replays are parsed and cached from high-Elo player timeline data into [`ShadowMatchRepository`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/rl/shadow_match/replay_loader.py) ($>2,500$ top-tier games).
   * At `reset()`, a match is sampled, and the agent begins from the human player's starting state.
2. **Sequential Phase Loop:**
   * **Planning Phase:** The agent executes factorized discrete actions ($a \in \{0 \dots 110\}$) with pre-softmax valid action masking until it selects `PASS_ROUND` ($a = 0$).
   * **Combat Resolution:** The agent's real-time board is evaluated against the historical opponent's board for that round.
   * **Damage & Economy:** Standard TFT rules apply (combat win/loss streak bonuses, base round income $+5$, interest capped at $50\text{g}$, player damage calculation).
3. **Termination & Survival:**
   * If the agent's HP drops to $\le 0$, the episode terminates.
   * If the agent survives past the final round recorded in the trace, it earns a victory terminal reward.

---

### 6.3 Reward Decomposition & Alignment

AlphaStar v6.1 uses a decomposed multi-objective reward structure identical in logging format to the v6 baseline:

$$R_{\text{step}} = R_{\text{combat}} + R_{\text{interest}} + R_{\text{terminal}} + R_{\text{env}}$$

* **Combat Reward ($R_{\text{combat}}$):** $+0.5$ per combat victory; $-0.02 \times \text{Damage}$ on defeat.
* **Interest Reward ($R_{\text{interest}}$):** $+0.05$ per round when holding $\ge 50\text{g}$ to encourage strong economic habits.
* **Terminal Reward ($R_{\text{terminal}}$):** Scaled by survival relative to the original Challenger player ($+2.0$ for outlasting the match, $-1.0$ to $-2.0$ for early elimination).
* **Environment Reward ($R_{\text{env}}$):** Total environment transition reward.

---

### 6.4 Training Architecture & CLI Integration

* **Trainer Class:** [`ShadowTrainer`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/rl/shadow_match/train_shadow.py), subclassing [`LeagueTrainer`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/rl/train.py).
* **Warm-Start Initialization:** Initializes directly from the baseline v6 Gen 300 checkpoint (`training_state.pt`), preserving learned tactical knowledge while adapting to authentic Challenger boards.
* **Command:**
  ```powershell
  .venv\Scripts\python.exe -m tft_ai_player.cli rl-train-shadow `
      --pretrained-checkpoint D:/tft-winner-data/set18/models/rl/checkpoints/ppo_alphastar_v6/gen_0300/training_state.pt `
      --checkpoint-dir D:/tft-winner-data/set18/models/rl/checkpoints/ppo_alphastar_v6_1_shadow `
      --generations 300 `
      --device cuda `
      --run-name ppo_alphastar_v6_1
  ```
* **Periodic Evaluation ([`ShadowMatchEvaluator`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/rl/shadow_match/evaluator.py)):**
  Evaluates 30 held-out Challenger matches every 25 generations, reporting estimated placement, top-4 rate, win rate, and combat win rate against authentic human opponents.

---

## 7. AlphaStar v6.2: Adaptive Ranked Ladder RL (Curriculum MMR Progression)

### 7.1 Concept & Branching Architecture
* **Branching Root:** In accordance with the versioning standard, **AlphaStar v6.2 branches directly from the base `v6` Gen 300 checkpoint** (`ppo_alphastar_v6/gen_0300/training_state.pt`). It represents an alternative Phase 2 regime to `v6.1`.
* **The Problem with Direct High-Elo Ingestion (`v6.1`):** In `v6.1`, dropping the agent exclusively into Challenger lobbies creates an overly steep penalty cliff: at early generations, the agent survives comfortably through Stage 4 (~20-25 rounds) but gets out-scaled in Stage 5, collapsing the policy into heavy reroll panic instead of learning active XP pacing.
* **The Solution (`v6.2`):** Rather than a static environment, `v6.2` introduces an **Adaptive Ranked Matchmaking Ladder**:
  1. The agent starts with an initial MMR / Division (e.g. **Gold IV** / $1,200\text{ Elo}$).
  2. The environment samples exclusively from **1st-place real human matches** matching the agent's current rank tier.
  3. When the agent finishes an episode, its final survival round is translated into an empirical lobby placement according to that specific tier's survival curve.
  4. The agent gains or loses League Points (LP / MMR). If it accumulates $+100\text{ LP}$, it promotes to the next division; if it falls below $0\text{ LP}$, it demotes.
  5. The next rollout dynamically samples from the new tier's replay pool.

```
                   Current Agent MMR / Tier (e.g. Gold -> Platinum -> Emerald -> Challenger)
                                                │
                                                ▼
                         Sample 1st-Place Human Match from Current Tier
                                                │
                                                ▼
                                PPO Actor-Critic Rollout Step
                               (Shop, Buy, Level, Position, Equip)
                                                │
                                                ▼
                           Empirical Placement Determination (1st to 8th)
                               (Rank-Specific Stage Death Distribution)
                                                │
                                ┌───────────────┴───────────────┐
                                ▼                               ▼
                         LP Delta (±10 to ±40)           PPO Reward Signal
                         Promotion / Demotion          R_step = R_combat + R_LP
```

### 7.2 Tier Progression & Promotion Logic
* **Tiers Supported:** Bronze $\to$ Silver $\to$ Gold $\to$ Platinum $\to$ Emerald $\to$ Diamond $\to$ Master $\to$ Grandmaster $\to$ Challenger.
* **Tier Partitions:** Mapped directly to disk paths (`D:/tft-winner-data/tiers/{tier}/players/`).
* **LP Economy:**
  * **1st Place:** $+40\text{ LP}$
  * **2nd Place:** $+30\text{ LP}$
  * **3rd Place:** $+20\text{ LP}$
  * **4th Place:** $+10\text{ LP}$
  * **5th Place:** $-10\text{ LP}$
  * **6th Place:** $-20\text{ LP}$
  * **7th Place:** $-30\text{ LP}$
  * **8th Place:** $-40\text{ LP}$
* **Promotions:** Reaching $100\text{ LP}$ advances division (IV $\to$ III $\to$ II $\to$ I $\to$ Next Tier).
* **Demotions:** Falling below $0\text{ LP}$ demotes with a grace buffer.





