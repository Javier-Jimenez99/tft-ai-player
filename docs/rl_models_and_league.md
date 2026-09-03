# Reinforcement Learning & AlphaStar League Pipeline Specification

This document provides a comprehensive technical guide to the **Reinforcement Learning (RL)** models, **Deep Learning architecture**, and **AlphaStar-style Multi-Agent League** implemented for Teamfight Tactics (TFT) in [`src/tft_ai_player/rl/`](../src/tft_ai_player/rl).

---

## Table of Contents
1. [Overview & Problem Framing](#1-overview--problem-framing)
2. [Gymnasium Decision MDP & State Representation (704D)](#2-gymnasium-decision-mdp--state-representation-704d)
3. [Factorized Discrete Action Space (|A| = 111) & Action Masking](#3-factorized-discrete-action-space-a--111--action-masking)
4. [Calibrated Combat Simulation via LightGBM (1,107 dims)](#4-calibrated-combat-simulation-via-lightgbm-1107-dims)
5. [Multi-Objective Reward Formulation](#5-multi-objective-reward-formulation)
6. [PPO Actor-Critic Neural Architecture](#6-ppo-actor-critic-neural-architecture)
7. [The AlphaStar League Engine & Prioritized Fictitious Self-Play (PFSP)](#7-the-alphastar-league-engine--prioritized-fictitious-self-play-pfsp)
8. [Deterministic Benchmark Bot Validation](#8-deterministic-benchmark-bot-validation)
9. [Diagnostics, Telemetry & Policy Collapse Detection](#9-diagnostics-telemetry--policy-collapse-detection)
10. [CLI Commands & Usage Guide](#10-cli-commands--usage-guide)

---

## 1. Overview & Problem Framing

Auto-battlers like Teamfight Tactics combine challenges from multiple domains of artificial intelligence:
- **Multi-Agent 8-Player Non-Zero Sum Game**: 8 autonomous players compete simultaneously for ordinal tournament placements ($1^{\text{st}}$ through $8^{\text{th}}$). In competitive TFT, finishing in the **Top 4** is considered a win.
- **High Combinatorial State & Action Spaces**: Fielded unit positions across a $4 \times 7$ hex grid, item assignments, bench holdings, and shop card transactions.
- **Non-Transitivity & Meta-Cycles**: Self-play without diverse counter-strategies causes cyclic forgetting.

To solve this, Phase 3 integrates:
1. Pre-trained representation grounding from the frozen **MultiModalFusionTrunk** (320D) and canonical **Z-Index** centroids (256D).
2. Offline guidance from the **State Transition Predictor** (World Model).
3. Fast tabular combat physics via calibrated **LightGBM** (1,107 dims).
4. **Maskable PPO** with pre-softmax invalid action pruning.
5. An **AlphaStar Multi-Agent League** with Prioritized Fictitious Self-Play (PFSP) and deterministic benchmark bots.

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

## 2. Gymnasium Decision MDP & State Representation (704D)

The policy network receives an invariant concatenated state vector $o_t \in \mathbb{R}^{704}$ constructed at each decision step:

1. **Core State ($320\text{D}$):** Output of frozen [`MultiModalFusionTrunk.encode_board()`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L321) ($256\text{D}$) combined with [`StateMLP`](file:///c:/Users/javij/Desktop/Proyectos/tft-ai-player/src/tft_ai_player/embeddings/model.py#L246) ($64\text{D}$).
2. **Shop State ($64\text{D}$):** 5 shop units pass through frozen `Champ2Vec` ($5 \times 32 = 160\text{D}$) and project through `Linear(160, 64) -> LayerNorm(64) -> ReLU()`.
3. **Bench State ($64\text{D}$):** 9 bench slots pass through frozen `Champ2Vec` ($9 \times 32\text{D}$). An order-invariant DeepSets reduction (Sum-Pool + Mean-Pool) projects through `Linear(64, 64) -> LayerNorm(64) -> ReLU()`.
4. **Target Conditioning ($256\text{D}$):** For Exploiter agents, assigned Z-Index centroid $z_k$. For the Main Agent, strictly zero-filled ($\mathbf{0}_{256}$).

---

## 3. Factorized Discrete Action Space ($|\mathcal{A}| = 111$) & Action Masking

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

### Strict Pre-Softmax Action Masking
To prevent policy entropy collapse and illegal actions, the environment exposes a boolean validity mask $\mathbf{m} \in \{0, 1\}^{111}$:

$$\pi(a \mid o_t) = \text{Softmax}(\mathbf{z}_{\text{logits}} + (1 - \mathbf{m}) \cdot (-10^{9}))$$

---

## 4. Calibrated Combat Simulation via LightGBM (1,107 dims)

When `PASS_ROUND` executes:
1. `TFTBoardFeatureExtractor` transforms the focal board and opponent board into differential features $\mathbf{x}_{1107}$.
2. Outcome sampling:
   $$p_{\text{win}} = \text{PlattScale}(\text{LGBMClassifier}(\mathbf{x}_{1107})) \in [0.0, 1.0]$$
   $$y_{\text{combat}} \sim \text{Bernoulli}(p_{\text{win}})$$
3. Damage application:
   * If $y = 1$: Win gold $+1\text{g}$, win streak incremented.
   * If $y = 0$: Damage $\Delta \text{HP} = \max(1, \text{Round}(\text{LGBMRegressor}(\mathbf{x}_{1107})))$, $\text{HP}_{t+1} = \text{HP}_t - \Delta \text{HP}$.

---

## 5. Multi-Objective Reward Formulation

$$R_{\text{step}} = R_{\text{env}} + \alpha \cdot R_{\text{macro}} + \beta \cdot R_{\text{micro}}$$

### 5.1 Reward Terms
1. **$R_{\text{env}} = R_{\text{combat}} + R_{\text{interest}} + R_{\text{terminal}}$**:
   * $R_{\text{combat}} = +0.5$ (win), $-0.02 \times \Delta \text{HP}$ (defeat).
   * $R_{\text{interest}} = +0.05$ (if gold $\ge 50\text{g}$).
   * $R_{\text{terminal}} = +2.0$ (Rank 1), $+1.0$ (Rank 2..4), $-1.0$ (Rank 5..6), $-2.0$ (Rank 7..8).
2. **$R_{\text{micro}} = \text{CosineSimilarity}(s_{t+1}, \hat{s}_{t+1})$**: Alignment with World Model trajectory.
3. **$R_{\text{macro}} = \text{CosineSimilarity}(\mathbf{h}_{\text{board}}^{t+1}, z_k)$**: Alignment with assigned Z-Index centroid.

### 5.2 Agent Parameterization
| Policy Type | Macro Weight ($\alpha$) | Micro Weight ($\beta$) | Strategic Behavior |
| :--- | :---: | :---: | :--- |
| **Main Agent (Generalist)** | **$0.0$** | **$0.3$** | Optimizes global lobby survival without composition bias. |
| **Exploiter Agent (Specialist)** | **$0.8$** | **$0.2$** | Masters execution of one targeted archetype $z_k$. |

---

## 6. PPO Actor-Critic Neural Architecture

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

### Hyperparameters
* **Clip Ratio ($\epsilon$):** $0.2$
* **Generalized Advantage Estimation ($\lambda_{\text{GAE}}$):** $0.95$
* **Discount Factor ($\gamma$):** $0.99$
* **Entropy Regularization ($c_1$):** $0.01 \to 0.001$
* **Value Loss Coefficient ($c_2$):** $0.5$
* **Max Gradient Norm:** $0.5$
* **Learning Rate:** $2.5 \times 10^{-4}$ with linear decay
* **PPO Epochs per Batch:** $4$
* **Mini-batch Size:** $512$ across rollout steps

---

## 7. The AlphaStar League Engine & Prioritized Fictitious Self-Play (PFSP)

### 7.1 Matchmaker Distribution
* **Active Self-Play (35%):** Current Main Agent policy.
* **Exploiter Pool (40%):** $K=15$ Specialist Exploiters ($z_1 \dots z_{15}$).
* **Historical Snapshots (15%):** Frozen historical Main Agent checkpoints.
* **Deterministic Benchmark Bots (10%):** Hardcoded baselines.

### 7.2 PFSP Payoff Sampling
The matchmaker maintains the empirical win-rate matrix $M_{i, j}$. When the Main Agent searches for an opponent:

$$P(\text{Match Opponent } j) = \frac{f(1 - M_{\text{Main}, j})}{\sum_{k} f(1 - M_{\text{Main}, k})}, \quad f(x) = x^2$$

---

## 8. Deterministic Benchmark Bot Validation

Every 25 generations, gradient updates freeze while the Main Agent plays 100 evaluation matches against:
1. **Bot Alpha (Standard 4-Cost Fast-8):** Rushes level 8 at Stage 4-2; standard front-to-back meta comps.
2. **Bot Beta (Hyperroll 1-Cost):** Spends all economy at Stage 3-1 to 3-star early game units.
3. **Bot Gamma (Greedy Economy Open-Fort):** Holds 50g at all costs to push fast level 9.

---

## 9. Diagnostics, Telemetry & Policy Collapse Detection

### 9.1 The "Principal" Dashboard (5 Essential Plots)
To avoid metric overload on Weights & Biases, the top-level section **`Principal`** provides exactly **5 executive charts** for immediate go/no-go assessment:

| Plot Name | Metric Logged | Normal Target Range | Interpretation / Health Signal | Kill / Adjustment Rule |
| :--- | :--- | :---: | :--- | :--- |
| **`Principal/1_Average_Placement`** | `avg_placement` (or `eval_avg_placement`) | $4.5 \to \le 3.5$ | **Overall Performance:** Tracks tournament finish rank ($1.0 = 1\text{st}, 8.0 = 8\text{th}$). If trending downward, the agent is legitimately improving. | If flat at $4.5$ or drifting $\ge 5.5$ for $>100$ gens: check action masking and reward scaling. |
| **`Principal/2_Critic_Explained_Variance`** | `explained_variance` | $0.50 \to 0.95$ | **Critic Value Accuracy:** Measures how accurately the value network predicts discounted returns ($1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$). | **KILL / ROLLBACK if $< 0.0$** for $>5$ consecutive iterations. Value network is corrupted. |
| **`Principal/3_Policy_Entropy`** | `policy_entropy` | $2.5 \to 0.8$ | **Exploration Balance:** Action distribution randomness. Should gradually and smoothly decay. | **KILL if $< 0.15$ (Premature Collapse):** Agent is permanently frozen into a single repetitive action. If $>4.0$, agent is random. |
| **`Principal/4_Policy_Stability_KL`** | `approx_kl` | $0.005 \to 0.020$ | **PPO Step Stability:** Quantifies policy drift $D_{\text{KL}}(\pi_{\text{old}} \parallel \pi_{\text{new}})$ per batch. | **ADJUST if $> 0.05$ (Policy Explosion):** Triggers epoch early-stopping. If continuous, reduce learning rate by 50%. |
| **`Principal/5_Benchmark_WinRate`** | `(Alpha_WR + Beta_WR + Gamma_WR)/3` | $50\% \to 85\%$ | **Absolute Competence:** Win rate against fixed, deterministic benchmark bots (Fast-8, Hyperroll, Economy). | If $< 25\%$, the agent is not mastering standard fundamentals (economy thresholds & level tempo). |

### 9.2 Comprehensive Diagnostics Table

| Metric Name | Normal Range | Warning Threshold | Pathology / Failure Mode |
| :--- | :---: | :---: | :--- |
| **`policy_entropy`** | $2.5 \to 0.8$ | $< 0.15$ or $> 4.0$ | **Entropy Collapse:** Agent fixates on single action. |
| **`explained_variance`** | $0.50 \to 0.92$ | $< 0.0$ (Negative) | **Critic Failure:** Advantage corruption. |
| **`approx_kl`** | $0.005 \to 0.02$ | $> 0.05$ | **Policy Explosion:** Triggers early stopping for current epoch. |
| **`action_mask_rejection_rate`** | $0.0\%$ (Strict) | $> 0.00001$ | **Masking Leak:** Invalid actions leaking past mask. |
| **`macro_z_cosine`** | $0.65 \to 0.92$ | $< 0.40$ (Stage 5) | **Specialist Divergence:** Exploiter drifting off archetype. |
| **`micro_world_model_cosine`** | $0.70 \to 0.88$ | $< 0.50$ | **Macro Regression:** Divergence from human planning. |

> **Rollback Protection:** If `explained_variance` remains negative for $>5$ consecutive iterations, the learning rate drops by $50\%$ and policy weights roll back to the previous stable league generation checkpoint.

---

## 10. Combat Engine Comparison & RL Speed Impact

We evaluated the real-world system performance of the surrogate combat engines (LightGBM vs. Deep Learning on GPU vs. Heuristic Baseline):

### 10.1 End-to-End RL System Impact (512 Steps + 4 PPO Epochs)

| Combat Engine | Full Gen Time | Rollout Time | PPO Train Time | Combat Time | RL Throughput | Overall System Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM ML (1,107 Features)** | **`9.77 s`** | `9.61 s` | `0.15 s` | `3.02 s` | **`52.4 steps/s`** | Baseline |
| **Deep Learning (GPU Trunk)** | **`6.47 s`** | `6.33 s` | `0.13 s` | `0.68 s` | **`79.1 steps/s`** | **`1.51x faster` (33.8% time saved)** 🚀 |
| **Pure Heuristic Baseline** | **`8.16 s`** | `7.76 s` | `0.28 s` | `0.04 s` | **`62.7 steps/s`** | $1.20\times$ faster |

### 10.2 Microbenchmark Latency & Throughput Scaling

* **In-RL Latent Reuse:** Reusing already computed $s_t / \mathbf{h}_{\text{board}}$ embeddings on GPU takes **`0.324 ms`** per match (**`3,089 matches/s`**), achieving a **`16.1x speedup`** over LightGBM ($5.20\text{ ms}$).
* **Batched Parallel Scaling ($N = 4,096$):** Deep Learning GPU processes **`162,040 matches/s`** vs LightGBM's **`3,565 matches/s`** (**`45.5x throughput gain`**).

---

## 11. CLI Commands & Usage Guide

### 1. Training with PPO and AlphaStar League
```powershell
uv run python -m tft_ai_player.cli rl-train `
    --generations 50 `
    --rollout-steps 4096 `
    --batch-size 512 `
    --epochs 4 `
    --lr 2.5e-4 `
    --eval-every 25 `
    --snapshot-every 50 `
    --checkpoint-dir "checkpoints/league" `
    --wandb-project "tft-ai-league" `
    --run-name "ppo_alphastar_v1"
```
*(Note: Automatically defaults to the high-throughput Deep Learning GPU combat resolver `DLCombatResolver` for 33.8% faster training).*


### 2. League Tournament Evaluation
```powershell
uv run python -m tft_ai_player.cli rl-league `
    --matches 20 `
    --checkpoint-dir "checkpoints/league" `
    --markdown-out "dashboards/league_standings.md"
```

