# Reinforcement Learning & AlphaStar League System Architecture

This document provides a comprehensive technical guide to the **Reinforcement Learning (RL)** models, **Deep Learning architecture**, and **AlphaStar-style Multi-Agent League** implemented for Teamfight Tactics (TFT) in [`src/tft_ai_player/rl/`](../src/tft_ai_player/rl).

---

## Table of Contents
1. [Overview & Problem Framing](#1-overview--problem-framing)
2. [Deep Learning Model Architecture: `TFTActorCritic`](#2-deep-learning-model-architecture-tftactorcritic)
   - [Entity Embeddings for Champions & Items](#21-entity-embeddings-for-champions--items)
   - [Spatial Board & Bench Encoders](#22-spatial-board--bench-encoders)
   - [Permutation-Invariant Opponent Pooling (DeepSets)](#23-permutation-invariant-opponent-pooling-deepsets)
   - [Trunk & Dual Actor-Critic Heads](#24-trunk--dual-actor-critic-heads)
3. [Invalid Action Masking (`TOTAL_DISCRETE_ACTIONS = 1721`)](#3-invalid-action-masking-total_discrete_actions--1721)
4. [Reinforcement Learning Algorithm: Maskable PPO](#4-reinforcement-learning-algorithm-maskable-ppo)
   - [Generalized Advantage Estimation (GAE-$\lambda$)](#41-generalized-advantage-estimation-gae-lambda)
   - [Clipped Surrogate Objective](#42-clipped-surrogate-objective)
   - [Total Loss Function](#43-total-loss-function)
5. [Reward Shaping & Placement Optimization](#5-reward-shaping--placement-optimization)
6. [Handling TFT's Luck & Stochasticity Factor](#6-handling-tfts-luck--stochasticity-factor)
7. [AlphaStar Multi-Agent League & 8-Player Elo](#7-alphastar-multi-agent-league--8-player-elo)
   - [League Member Ecosystem](#71-league-member-ecosystem)
   - [Prioritized Fictitious Self-Play (PFSP)](#72-prioritized-fictitious-self-play-pfsp)
   - [8-Player Multilateral Elo Rating System](#73-8-player-multilateral-elo-rating-system)
   - [Hall of Fame & Checkpoint Persistence](#74-hall-of-fame--checkpoint-persistence)
8. [CLI Commands & Usage Guide](#8-cli-commands--usage-guide)
9. [Verification & Test Coverage](#9-verification--test-coverage)

---

## 1. Overview & Problem Framing

Auto-battlers like Teamfight Tactics combine challenges from multiple domains of artificial intelligence:
- **Multi-Agent 8-Player Non-Zero Sum Game**: Rather than a 1v1 zero-sum game, 8 autonomous players compete simultaneously for ordinal tournament placements ($1^{\text{st}}$ through $8^{\text{th}}$). In competitive TFT, finishing in the **Top 4** is considered a win (LP gain).
- **High Stochasticity / Imperfect Information**: Shop draws draw from a shared finite champion pool, item drops vary by creep rounds, and round combat matchmaking is randomized.
- **Astronomical Action Space**: Over 1,700 micro-actions per step (shop purchases, rerolls, level-ups, repositioning across a $4 \times 7$ hex grid, item equipping, item combining, and passing turns).
- **Non-Transitivity & Meta-Cycles**: Simple self-play suffers from cyclical forgetting (e.g. learning fast-9 legendaries, forgetting how to defend against fast reroll tempo, and regressing).

To resolve these complexities, the framework combines a **multi-modal Actor-Critic neural network**, **Maskable PPO with Generalized Advantage Estimation**, and an **AlphaStar-inspired League system with 8-player multilateral Elo**.

---

## 2. Deep Learning Model Architecture: `TFTActorCritic`

The model implemented in [`TFTActorCritic`](../src/tft_ai_player/rl/models/networks.py) processes multi-modal game observations through dedicated sub-networks:

```
                                  TFT OBSERVATION
  ┌───────────────────────┬────────────────────────┬─────────────────────────┐
  │ Player Stats (9,)     │ Board & Bench Grid     │ Opponent Boards (7x28)  │
  │ (HP, Gold, Level...)  │ (Champs, Stars, Items) │ (Public Scouting)       │
  └──────────┬────────────┴───────────┬────────────┴────────────┬────────────┘
             ▼                        ▼                         ▼
   ┌───────────────────┐    ┌───────────────────┐     ┌───────────────────┐
   │    Stats MLP      │    │ Entity Embedding  │     │  Shared Opponent  │
   │  (Linear + ReLU)  │    │ (Champs & Items)  │     │  Encoder + Pooling│
   └─────────┬─────────┘    └─────────┬─────────┘     └─────────┬─────────┘
             │                        │                         │
             └────────────────────────┼─────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   Multi-Modal Fusion      │
                        │   LayerNorm + Trunk MLP   │
                        └─────────────┬─────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
          ┌─────────────────────┐           ┌─────────────────────┐
          │     ACTOR HEAD      │           │     CRITIC HEAD     │
          │ 1,721 Action Logits │           │ State Value V(s)    │
          │  + Invalid Masking  │           │ Expected Placement  │
          └─────────────────────┘           └─────────────────────┘
```

### 2.1 Entity Embeddings for Champions & Items
Passing champion and item IDs as raw numerical scalars introduces a false ordinal bias (e.g. implying champion #50 is "greater" than champion #12). 

Instead, the network uses trainable dense lookup embeddings:
- `champ_embedding = nn.Embedding(num_champs + 1, embedding_dim=16)`
- `item_embedding = nn.Embedding(num_items + 1, embedding_dim=8)`

Every unit on the board or bench is encoded as a combined dense vector:
$$\mathbf{u} = [\mathbf{e}_{\text{champ}} \parallel \text{star} / 3 \parallel \mathbf{e}_{\text{item}_1} \parallel \mathbf{e}_{\text{item}_2} \parallel \mathbf{e}_{\text{item}_3}] \in \mathbb{R}^{16 + 1 + 24} = \mathbb{R}^{41}$$

### 2.2 Spatial Board & Bench Encoders
- **Board Grid ($4 \times 7 = 28$ hexes)**: Flattened entity embeddings ($28 \times 41 = 1148$) are projected through a two-layer feedforward network with ReLU activations down to a 128-dimensional representation, learning positional synergies (frontline vanguards vs backline carries).
- **Bench (9 slots)**: Projected down to a 64-dimensional feature representation.
- **Item Bench (10 slots) & Shop (5 slots)**: Processed via entity embeddings and dedicated linear projection layers.

### 2.3 Permutation-Invariant Opponent Pooling (DeepSets)
There are 7 active opponents in the lobby. If two opponents swap player seat indices, the strategic state is identical. To guarantee permutation invariance, all 7 opponent boards are passed through a shared linear encoder and aggregated via **symmetric mean-pooling**:
$$\mathbf{h}_{\text{opponents}} = \frac{1}{7} \sum_{k=1}^{7} \text{ReLU}\left(\mathbf{W}_{\text{opp}} \mathbf{u}_{\text{board}}^{(k)} + \mathbf{b}_{\text{opp}}\right)$$

### 2.4 Trunk & Dual Actor-Critic Heads
1. **Shared Fusion Trunk**:
   $$\mathbf{z} = \text{MLP}\left(\text{LayerNorm}\left([\mathbf{h}_{\text{board}} \parallel \mathbf{h}_{\text{bench}} \parallel \mathbf{h}_{\text{items}} \parallel \mathbf{h}_{\text{shop}} \parallel \mathbf{h}_{\text{stats}} \parallel \mathbf{h}_{\text{opp\_summary}} \parallel \mathbf{h}_{\text{opponents}}]\right)\right) \in \mathbb{R}^{256}$$
2. **Actor Head (Policy $\pi_\theta$)**: Linear layer $\mathbf{z} \to \mathbb{R}^{1721}$ producing logits for all discrete micro-actions.
3. **Critic Head (Value $V_\phi$)**: Linear layer $\mathbf{z} \to \mathbb{R}^{1}$ estimating expected cumulative return and final tournament finish.

---

## 3. Invalid Action Masking (`TOTAL_DISCRETE_ACTIONS = 1721`)

TFT features 1,721 discrete micro-actions per step:
| Action ID Range | Action Type | Description |
| :--- | :--- | :--- |
| `0` | PASS | Conclude micro-actions and advance to combat phase |
| `1 .. 5` | BUY SHOP | Buy champion card in shop slot 0 to 4 |
| `6` | REROLL SHOP | Refresh shop cards (cost: 2 gold) |
| `7` | BUY EXP | Purchase 4 experience points (cost: 4 gold) |
| `8` | TOGGLE LOCK | Lock or unlock shop for next round |
| `9 .. 17` | SELL BENCH | Sell unit on bench slot 0 to 8 |
| `18 .. 45` | SELL BOARD | Sell fielded unit on board hex 0 to 27 |
| `46 .. 297` | BENCH $\to$ BOARD | Move unit from bench slot to board hex ($9 \times 28 = 252$) |
| `298 .. 549` | BOARD $\to$ BENCH | Move unit from board hex to bench slot ($28 \times 9 = 252$) |
| `550 .. 1305` | BOARD $\to$ BOARD | Reposition fielded unit between board hexes ($28 \times 27 = 756$) |
| `1306 .. 1585` | EQUIP $\to$ BOARD | Equip item from item bench to board unit ($10 \times 28 = 280$) |
| `1586 .. 1675` | EQUIP $\to$ BENCH | Equip item from item bench to bench unit ($10 \times 9 = 90$) |
| `1676 .. 1720` | COMBINE ITEMS | Combine two component items on item bench ($\binom{10}{2} = 45$) |

### Masked Softmax Implementation
At any tick, over 95% of actions are illegal. Penalizing illegal actions with negative rewards wastes policy capacity learning rules instead of strategy. 

We use [`MaskedCategorical`](../src/tft_ai_player/rl/models/distributions.py):
$$\pi(a_i | s) = \frac{\exp(z_i) \cdot m_i}{\sum_{j} \exp(z_j) \cdot m_j}$$
where $m_i \in \{0, 1\}$ is the boolean validity mask computed dynamically by `get_action_mask()`. Invalid action logits are set to $-\infty$, ensuring an exact probability of **$0.0$** and completely eliminating illegal explorations.

---

## 4. Reinforcement Learning Algorithm: Maskable PPO

The agent is trained using Proximal Policy Optimization (PPO) in [`MaskablePPO`](../src/tft_ai_player/rl/algorithms/ppo.py).

### 4.1 Generalized Advantage Estimation (GAE-$\lambda$)
Rollouts are stored in [`RolloutBuffer`](../src/tft_ai_player/rl/algorithms/ppo.py). To reduce variance over multi-stage matches, advantages are calculated using GAE ($\gamma = 0.99, \lambda = 0.95$):
$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

### 4.2 Clipped Surrogate Objective
To prevent destabilizing policy shifts, the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is clipped within $[1 - \epsilon, 1 + \epsilon]$ ($\epsilon = 0.2$):
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

### 4.3 Total Loss Function
$$L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 S[\pi_\theta]$$
- $L^{\text{VF}}(\phi) = \frac{1}{2} (V_\phi(s) - R_t)^2$: Mean squared error critic loss.
- $S[\pi_\theta]$: Policy entropy regularization bonus ($c_2 = 0.01$) preventing premature collapse to rigid compositions.

---

## 5. Reward Shaping & Placement Optimization

Because games last 30 to 45 rounds, sparse terminal placement rewards are complemented by round-level signals:
1. **Round HP Preservation**: Penalizes damage taken from combat:
   $$r_{\text{combat}} = \frac{\Delta \text{HP}}{100} \times 0.5$$
2. **PvP Win Bonus**: $+0.1$ for winning a PvP combat round.
3. **Survival Tick**: $+0.02$ per round survived.
4. **Tournament Placement Reward**: Awarded upon elimination or game end:
   - $1^{\text{st}}$: `+1.0` | $2^{\text{nd}}$: `+0.6` | $3^{\text{rd}}$: `+0.4` | $4^{\text{th}}$: `+0.2`
   - $5^{\text{th}}$: `-0.2` | $6^{\text{th}}$: `-0.4` | $7^{\text{th}}$: `-0.6` | $8^{\text{th}}$: `-1.0`

---

## 6. Handling TFT's Luck & Stochasticity Factor

| Stochastic Factor | Challenge | Framework Solution | Implementation Reference |
| :--- | :--- | :--- | :--- |
| **Shop & Item RNG** | Lucky high-rolls skew evaluations | **Common Random Numbers (CRN) / Paired Seeds**: Candidate models compete across identical game seeds with mirrored card draws and item drops. | [`TournamentEvaluator.run_paired_benchmark()`](../src/tft_ai_player/rl/evaluation/evaluator.py) |
| **High Return Variance** | Single-game noise destabilizes gradient steps | **GAE-$\lambda$ + Advantage Normalization**: Mini-batch advantage normalization $\frac{\hat{A} - \mu}{\sigma}$ removes game-to-game baseline shifts. | [`MaskablePPO.train_epoch()`](../src/tft_ai_player/rl/algorithms/ppo.py) |
| **Meta Rigidity** | Forcing one build fails when shop doesn't roll it | **Entropy Regularization ($c_2 = 0.01$)**: Keeps action exploration active so the policy pivots to open traits rather than dead-ends. | [`MaskablePPO`](../src/tft_ai_player/rl/algorithms/ppo.py) |
| **Value Estimation** | Disentangling good play from bad luck | **Placement Expectancy Critic**: Learns expected placement from current board and econ, penalizing greedy bleedouts even if lucky later. | [`TFTActorCritic.critic_head`](../src/tft_ai_player/rl/models/networks.py) |

---

## 7. AlphaStar Multi-Agent League & 8-Player Elo

To prevent cyclical non-transitive meta shifts, the system implements an AlphaStar-inspired League framework in [`LeagueManager`](../src/tft_ai_player/rl/league/league_manager.py).

### 7.1 League Member Ecosystem
- **Main Agent(s)**: Continuously learning agents updated via Maskable PPO.
- **Exploiter Bots**: Specialized policies targeting specific playstyles (e.g. aggressive hyper-roll to punish greedy bankers, fast-9 legendary greed to punish passive tempo).
- **Hall of Fame Checkpoints**: Frozen historical snapshots of the Main Agent.
- **Baseline Anchors**: Deterministic bots (`StandardTempoBot`, `GreedyBankerBot`, `RandomBot`) acting as ground-truth skill anchors.

### 7.2 Prioritized Fictitious Self-Play (PFSP)
Lobby opponents are sampled with probability weighted by their historical difficulty against the focal agent:
$$P(\text{Opponent}_i) \propto (1 - \text{WinRate}(i, \text{Main Agent}))^\tau$$
Opponents with $\approx 50\%$ win rates or those causing losses are prioritized, forcing the learning policy to patch strategic blind spots.

### 7.3 8-Player Multilateral Elo Rating System
In [`MultilateralEloSystem`](../src/tft_ai_player/rl/league/elo.py), each 8-player match is decomposed into $\binom{8}{2} = 28$ pairwise matchups:
- For pair $(i, j)$, actual score $S_{ij} = 1.0$ if player $i$ placed above player $j$, $0.5$ if tied, $0.0$ if below.
- Expected score under logistic Elo:
  $$E_{ij} = \frac{1}{1 + 10^{(R_j - R_i) / 400}}$$
- Multilateral rating update across all 7 opponents:
  $$\Delta R_i = \frac{K}{7} \sum_{j \neq i} (S_{ij} - E_{ij})$$

In addition to Elo ratings, the system tracks:
- **Top-4 Rate** ($\% \text{ of finishes } \le 4^{\text{th}}$)
- **Average Placement** ($\sum \text{placements} / N$)
- **Win Rate** ($\% \text{ of } 1^{\text{st}} \text{ place finishes}$)

### 7.4 Hall of Fame & Checkpoint Persistence
Model checkpoints and profile metadata are serialized to disk via [`CheckpointManager`](../src/tft_ai_player/rl/league/checkpoints.py), enabling tournament resumption and historical replay evaluations.

---

## 8. CLI Commands & Usage Guide

The CLI exposes high-level subcommands for training and league evaluation:

```bash
# 1. Run 10 tournament matches across league bots and print Elo standings:
tft-ai-player rl-league --matches 10

# 2. Export markdown leaderboard report:
tft-ai-player rl-league --matches 20 --markdown-out docs/leaderboard.md

# 3. Train the RL agent using Maskable PPO with League evaluation:
tft-ai-player rl-train --generations 20 --rollout-steps 512 --batch-size 64 --eval-every 2
```

---

## 9. Verification & Test Coverage

The test suite in [`tests/test_rl_league.py`](../tests/test_rl_league.py) validates:
- **Masked Softmax & Distribution**: $0\%$ probability on invalid actions and strict argmax deterministic selection.
- **Actor-Critic Forward Pass**: Tensor dimensions `(batch, 1721)` and `(batch, 1)`.
- **Multilateral Elo Conservation**: Winning increases Elo, $8^{\text{th}}$ place decreases Elo, and pairwise updates sum symmetrically.
- **PFSP Opponent Sampling**: Sampling distribution shifts toward challenging opponents.
- **Paired-Seed CRN Benchmark**: Identical seeds produce bitwise deterministic placement outcomes.
- **RLBot Simulation Turn Execution**: Successfully completes planning turns in 8-player simulations.
- **PPO Mini-Iteration**: Buffer collection, advantage calculation, and backpropagation gradient updates.

Run all tests:
```bash
pytest tests/test_rl_league.py -v
```
Output:
```
tests/test_rl_league.py::test_numpy_masked_softmax PASSED                [ 12%]
tests/test_rl_league.py::test_pytorch_masked_categorical PASSED          [ 25%]
tests/test_rl_league.py::test_actor_critic_forward PASSED                [ 37%]
tests/test_rl_league.py::test_multilateral_elo_system PASSED             [ 50%]
tests/test_rl_league.py::test_league_manager_and_pfsp PASSED             [ 62%]
tests/test_rl_league.py::test_paired_seed_benchmark PASSED               [ 75%]
tests/test_rl_league.py::test_rlbot_adapter_turn PASSED                  [ 87%]
tests/test_rl_league.py::test_league_trainer_mini_iteration PASSED       [100%]
============================= 8 passed in 15.14s ==============================
```
